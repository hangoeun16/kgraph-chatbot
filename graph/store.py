"""Neo4j graph store for character and relationship persistence.

Replaces the in-memory NetworkX graph with a real graph database.
All reads and writes go through this module, keeping Cypher queries
centralized in `queries.py`.
"""

from neo4j import AsyncGraphDatabase, AsyncDriver

from models.character import Character, CharacterAttribute
from models.relationship import FamilyEdge, SocialEdge
from models.enums import FamilyRelation
from graph import queries


# Symmetric family relations have no distinct reverse; they are stored once
# with a deterministic orientation so (A, B) and (B, A) collapse to one edge.
_SYMMETRIC_FAMILY = {FamilyRelation.SPOUSE_OF, FamilyRelation.SIBLING_OF}

# How a family edge reads from the perspective of a given node, depending on
# whether the stored (canonical) edge points away from it ("out") or toward
# it ("in"). The "in" rows are the inferred reverse relations.
_FAMILY_PHRASE = {
    ("parent_of", "out"): "is parent of",
    ("parent_of", "in"): "is child of",
    ("spouse_of", "out"): "is spouse of",
    ("spouse_of", "in"): "is spouse of",
    ("sibling_of", "out"): "is sibling of",
    ("sibling_of", "in"): "is sibling of",
    # Legacy safety net for any child_of edges left by older data.
    ("child_of", "out"): "is child of",
    ("child_of", "in"): "is parent of",
}


def _canonical_family_edge(
    source: str, target: str, relation: FamilyRelation
) -> tuple[str, str, FamilyRelation]:
    """Reduce a family relation to the single form actually stored.

    Only two primitive edges are ever persisted:
    - parent_of (parent -> child)
    - a symmetric spouse_of / sibling_of, oriented alphabetically by name

    child_of is folded into a reversed parent_of, and symmetric relations are
    given a deterministic direction so duplicates collapse under MERGE. The
    reverse of every stored edge is inferred at read time, never stored.
    """
    if relation is FamilyRelation.CHILD_OF:
        # "source is child of target" == "target is parent of source"
        return target, source, FamilyRelation.PARENT_OF
    if relation is FamilyRelation.PARENT_OF:
        return source, target, FamilyRelation.PARENT_OF
    # Symmetric: orient by name so (A, B) and (B, A) become the same edge.
    if source.lower() <= target.lower():
        return source, target, relation
    return target, source, relation


class GraphStore:
    """Async Neo4j wrapper for the knowledge graph.

    Usage:
        store = GraphStore("bolt://localhost:7687", "neo4j", "password")
        await store.connect()
        await store.add_character(Character(name="Kim"))
        await store.close()
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Open the Neo4j connection."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            await self._driver.close()

    async def _run(self, query: str, **params):
        """Execute a single Cypher query and return all records."""
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return await result.data()

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------

    async def add_character(self, character: Character) -> None:
        """Create or merge a character node."""
        await self._run(
            queries.CREATE_CHARACTER,
            name=character.name,
            sex=character.sex.value,
            is_placeholder=character.is_placeholder,
            exists_from=character.exists_from,
            exists_until=character.exists_until,
        )

    async def rename_placeholder(self, parent_name: str, child_name: str) -> bool:
        """Rename a placeholder child node. Returns True if successful."""
        records = await self._run(
            queries.RENAME_PLACEHOLDER,
            parent_name=parent_name,
            child_name=child_name,
        )
        return len(records) > 0

    async def set_attribute(self, name: str, attr: CharacterAttribute) -> None:
        """Set or update an attribute on a character."""
        await self._run(
            queries.SET_ATTRIBUTE,
            name=name,
            key=attr.key,
            value=attr.value,
            valid_from=attr.valid_from,
            valid_until=attr.valid_until,
        )

    # ------------------------------------------------------------------
    # Family relationships
    # ------------------------------------------------------------------

    async def add_family_edge(self, edge: FamilyEdge) -> None:
        """Create a single canonical family edge.

        Only the canonical direction is stored (parent_of, or a
        deterministically-oriented spouse_of / sibling_of). The reverse
        relation is inferred when reading rather than duplicated in the
        graph, keeping one source of truth per relationship.
        """
        source, target, relation = _canonical_family_edge(
            edge.source, edge.target, edge.relation
        )
        await self._run(
            queries.CREATE_FAMILY_EDGE,
            source=source,
            target=target,
            relation=relation.value,
            valid_from=edge.valid_from,
            valid_until=edge.valid_until,
        )

    async def add_placeholder_children(
        self,
        parents: list[str],
        count: int,
        exists_from: int | None = None,
    ) -> None:
        """Create placeholder children shared by one or more co-parents.

        A single set of placeholder nodes is created and linked to every
        parent, so a couple's children are shared rather than duplicated
        per parent. The node key is derived from the sorted parent names,
        making the operation idempotent regardless of the order parents
        are supplied. A single parent (list of length one) yields the same
        keys and names as before.
        """
        if not parents:
            return
        key_prefix = "+".join(sorted(p.lower() for p in parents))
        name_prefix = " & ".join(parents)
        await self._run(
            queries.CREATE_PLACEHOLDER_CHILDREN,
            parents=parents,
            key_prefix=key_prefix,
            name_prefix=name_prefix,
            count=count,
            exists_from=exists_from,
        )

    # ------------------------------------------------------------------
    # Social relationships
    # ------------------------------------------------------------------

    async def add_social_edge(self, edge: SocialEdge) -> None:
        """Create a social relationship edge."""
        if edge.target_is_person:
            await self._run(
                queries.CREATE_SOCIAL_EDGE_TO_PERSON,
                source=edge.source,
                target=edge.target,
                relation=edge.relation.value,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            )
        else:
            await self._run(
                queries.CREATE_SOCIAL_EDGE,
                source=edge.source,
                target=edge.target,
                relation=edge.relation.value,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            )

    # ------------------------------------------------------------------
    # Temporal subgraph retrieval (GraphRAG core)
    # ------------------------------------------------------------------

    async def get_subgraph_at_time(self, event_order: int) -> list[dict]:
        """Retrieve all characters and relations valid at a given time point.

        This is the main GraphRAG retrieval method — instead of dumping
        the entire graph, it returns only what exists at the current
        narrative time point.
        """
        return await self._run(
            queries.FIND_SUBGRAPH_AT_TIME,
            event_order=event_order,
        )

    async def get_character_subgraph(
        self, name: str, event_order: int
    ) -> list[dict]:
        """Retrieve a character's local neighborhood at a given time point."""
        return await self._run(
            queries.FIND_CHARACTER_SUBGRAPH_AT_TIME,
            name=name,
            event_order=event_order,
        )

    # ------------------------------------------------------------------
    # Inferred relations (FHKB-inspired)
    # ------------------------------------------------------------------

    async def find_grandparents(self, event_order: int) -> list[dict]:
        """Infer grandparent-grandchild pairs via path traversal."""
        return await self._run(
            queries.FIND_GRANDPARENT, event_order=event_order
        )

    async def find_uncles_aunts(self, event_order: int) -> list[dict]:
        """Infer uncle/aunt-niece/nephew pairs via path traversal."""
        return await self._run(
            queries.FIND_UNCLE_AUNT, event_order=event_order
        )

    async def find_cousins(self, event_order: int) -> list[dict]:
        """Infer cousin pairs via path traversal."""
        return await self._run(
            queries.FIND_COUSIN, event_order=event_order
        )

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    async def get_all_context(self) -> list[dict]:
        """Get all named characters with their relations and attributes.

        Used to build the context string for the LLM prompt.
        """
        return await self._run(queries.GET_ALL_CONTEXT)

    async def build_context_string(self, event_order: int) -> str:
        """Build a human-readable context string for the current time point.

        This replaces the old get_graph_context() method. Instead of dumping
        everything, it filters by the active timeline position.
        """
        records = await self.get_subgraph_at_time(event_order)

        if not records:
            return "No story information recorded yet (knowledge graph is empty)."

        lines: list[str] = []
        for record in records:
            person = record["p"]
            name = person.get("name", "Unknown")

            # Attributes valid at this time point (age, occupation, ...).
            # Without these, conflict detection has no prior value to
            # compare against (e.g. age 4 -> 6 goes undetected).
            for attr in record.get("attributes", []):
                key = attr.get("key")
                value = attr.get("value")
                if key and value is not None:
                    lines.append(f"{name}'s {key} is {value}.")

            # Outgoing family edges: the canonical stored direction.
            for rel in record.get("out_relations", []):
                phrase = _FAMILY_PHRASE.get((rel.get("relation"), "out"))
                target = rel.get("target")
                if phrase and target:
                    lines.append(f"{name} {phrase} {target}.")

            # Incoming family edges: the reverse relation, inferred here
            # rather than stored (e.g. a stored parent_of reads as child_of
            # from the child's perspective).
            for rel in record.get("in_relations", []):
                phrase = _FAMILY_PHRASE.get((rel.get("relation"), "in"))
                source = rel.get("source")
                if phrase and source:
                    lines.append(f"{name} {phrase} {source}.")

            # Unnamed children summarized as a count. Their placeholder names
            # are deliberately never emitted, so the LLM cannot echo them back
            # as if they were real characters.
            unnamed = record.get("unnamed_children", 0) or 0
            if unnamed > 0:
                noun = "child" if unnamed == 1 else "children"
                lines.append(f"{name} has {unnamed} unnamed {noun}.")

        if not lines:
            return "No story information recorded yet (knowledge graph is empty)."

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization data
    # ------------------------------------------------------------------

    async def get_graph_data(self) -> dict:
        """Get nodes and edges formatted for frontend visualization."""
        records = await self._run(queries.GET_GRAPH_DATA)

        nodes = []
        edges = []

        for record in records:
            person = record["p"]
            node_id = person.get("name_lower", "")
            name = person.get("name", "Unknown")

            attributes = {}
            skip = {"name", "name_lower", "placeholder"}
            for key, value in person.items():
                if key not in skip and value is not None:
                    attributes[key] = str(value)

            nodes.append({
                "id": node_id,
                "name": name,
                "attributes": attributes,
            })

            for edge in record.get("edges", []):
                target_id = edge.get("target_name_lower")
                if target_id:
                    edges.append({
                        "from": node_id,
                        "to": target_id,
                        "relation": edge.get("relation", "related"),
                    })

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """Delete all nodes and relationships (memory reset)."""
        await self._run("MATCH (n) DETACH DELETE n")

    async def person_exists(self, name: str) -> bool:
        """Check if a named (non-placeholder) person exists."""
        records = await self._run(queries.CHECK_PERSON_EXISTS, name=name)
        return bool(records and records[0].get("exists"))

    async def count_placeholder_children(self, parent_name: str) -> int:
        """Count unnamed placeholder children of a parent."""
        records = await self._run(
            queries.COUNT_PLACEHOLDER_CHILDREN, parent_name=parent_name
        )
        if records:
            return records[0].get("placeholder_count", 0)
        return 0
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


# Reverse mapping for bidirectional family edges
_REVERSE_FAMILY = {
    FamilyRelation.PARENT_OF: FamilyRelation.CHILD_OF,
    FamilyRelation.CHILD_OF: FamilyRelation.PARENT_OF,
    FamilyRelation.SPOUSE_OF: FamilyRelation.SPOUSE_OF,
    FamilyRelation.SIBLING_OF: FamilyRelation.SIBLING_OF,
}


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
        """Create a family relationship with its reverse edge."""
        reverse = _REVERSE_FAMILY.get(edge.relation)
        if reverse:
            await self._run(
                queries.CREATE_FAMILY_EDGE_PAIR,
                source=edge.source,
                target=edge.target,
                relation=edge.relation.value,
                reverse_relation=reverse.value,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            )
        else:
            await self._run(
                queries.CREATE_FAMILY_EDGE,
                source=edge.source,
                target=edge.target,
                relation=edge.relation.value,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            )

    async def add_placeholder_children(
        self, parent_name: str, count: int, exists_from: int | None = None
    ) -> None:
        """Create placeholder child nodes for a parent."""
        await self._run(
            queries.CREATE_PLACEHOLDER_CHILDREN,
            parent_name=parent_name,
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

            # Relations valid at this time point
            for rel in record.get("relations", []):
                rel_type = rel.get("relation")
                target = rel.get("target")
                if rel_type and target:
                    if rel_type == "parent_of":
                        lines.append(f"{name} is parent of {target}.")
                    elif rel_type == "child_of":
                        lines.append(f"{name} is child of {target}.")
                    else:
                        lines.append(f"{name} is {rel_type} {target}.")

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
                if edge.get("target_name"):
                    edges.append({
                        "from": node_id,
                        "to": edge["target_name"].lower(),
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

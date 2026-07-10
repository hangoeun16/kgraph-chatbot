"""Graph applier: writes extraction results to the graph store.

This module bridges the extraction layer and the graph layer.
The ExtractionService returns typed results; the GraphApplier
translates those results into graph operations (creating nodes,
edges, renaming placeholders, setting attributes).

Keeping this separate from both extraction and storage means:
- Extraction can be tested without a database
- Storage can be tested without an LLM
- The mapping logic between the two is explicit and auditable
"""

from models.character import Character, CharacterAttribute
from models.relationship import FamilyEdge, SocialEdge
from models.enums import FamilyRelation
from models.extraction import (
    AttributeExtractionResult,
    FamilyExtractionResult,
    SocialExtractionResult,
)
from graph.store import GraphStore
from graph.temporal import TimelineManager


class GraphApplier:
    """Applies LLM extraction results to the Neo4j graph.

    Usage:
        applier = GraphApplier(store, timeline_manager)
        await applier.apply_family(family_result)
        await applier.apply_attributes(attribute_result)
        await applier.apply_social(social_result)
    """

    def __init__(self, store: GraphStore, timeline: TimelineManager) -> None:
        self._store = store
        self._timeline = timeline

    async def apply_family(self, result: FamilyExtractionResult) -> None:
        """Apply extracted family relationships to the graph.

        Handles three types of family extraction results:
        1. Named relationships (Kim is married to Jim)
        2. Child counts (they have 3 kids)
        3. Child naming (the oldest is Rick)
        """
        current_order = self._timeline.active_event_order

        # 1) Named relationships
        for rel in result.relationships:
            # Ensure both characters exist
            await self._store.add_character(
                Character(name=rel.person1, exists_from=current_order)
            )
            await self._store.add_character(
                Character(name=rel.person2, exists_from=current_order)
            )

            await self._store.add_family_edge(
                FamilyEdge(
                    source=rel.person1,
                    target=rel.person2,
                    relation=rel.relation,
                    valid_from=current_order,
                )
            )

        # 2) Child counts → shared placeholder nodes
        for cc in result.child_counts:
            for parent in cc.parents:
                await self._store.add_character(
                    Character(name=parent, exists_from=current_order)
                )
            await self._store.add_placeholder_children(
                parents=cc.parents,
                count=cc.count,
                exists_from=current_order,
            )

        # 3) Child naming → rename placeholders
        for cn in result.child_namings:
            success = await self._store.rename_placeholder(
                parent_name=cn.parent,
                child_name=cn.child_name,
            )
            if not success:
                # No placeholder to rename; create as new character
                await self._store.add_character(
                    Character(name=cn.child_name, exists_from=current_order)
                )
                await self._store.add_family_edge(
                    FamilyEdge(
                        source=cn.parent,
                        target=cn.child_name,
                        relation=FamilyRelation.PARENT_OF,
                        valid_from=current_order,
                    )
                )

    async def apply_attributes(
        self, result: AttributeExtractionResult
    ) -> None:
        """Apply extracted attributes to character nodes."""
        current_order = self._timeline.active_event_order

        for attr in result.attributes:
            await self._store.add_character(
                Character(name=attr.person, exists_from=current_order)
            )
            await self._store.set_attribute(
                name=attr.person,
                attr=CharacterAttribute(
                    key=attr.key,
                    value=attr.value,
                    valid_from=current_order,
                ),
            )

    async def apply_social(self, result: SocialExtractionResult) -> None:
        """Apply extracted social relationships to the graph."""
        current_order = self._timeline.active_event_order

        for rel in result.relationships:
            await self._store.add_character(
                Character(name=rel.person, exists_from=current_order)
            )

            if rel.target_is_person:
                await self._store.add_character(
                    Character(name=rel.target, exists_from=current_order)
                )

            await self._store.add_social_edge(
                SocialEdge(
                    source=rel.person,
                    target=rel.target,
                    relation=rel.relation,
                    target_is_person=rel.target_is_person,
                    valid_from=current_order,
                )
            )
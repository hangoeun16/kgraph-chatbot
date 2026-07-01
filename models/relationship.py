"""Relationship edge models with temporal validity.

Every relationship in the graph carries a time range indicating when
it is active in the narrative. This allows queries like "who was Kim
married to during the honeymoon?" to correctly exclude relationships
that hadn't formed yet or had already ended.
"""

from pydantic import BaseModel

from models.enums import FamilyRelation, SocialRelation


class FamilyEdge(BaseModel):
    """A family relationship edge between two characters.

    Attributes:
        source: Name of the source character.
        target: Name of the target character.
        relation: The family relation type (parent_of, child_of, etc.).
        valid_from: Event order at which this relationship begins.
        valid_until: Event order at which this relationship ends
            (e.g. divorce for spouse_of). None means it persists.
    """

    source: str
    target: str
    relation: FamilyRelation
    valid_from: int | None = None
    valid_until: int | None = None


class SocialEdge(BaseModel):
    """A non-family relationship edge between a character and a target.

    The target can be another character or a concept (e.g. "pizza", "music").

    Attributes:
        source: Name of the source character.
        target: Name of the target (character or concept).
        relation: The social relation type (likes, knows, etc.).
        target_is_person: True if the target is another character, False if
            it is a concept node.
        valid_from: Event order at which this relationship begins.
        valid_until: Event order at which this relationship ends.
    """

    source: str
    target: str
    relation: SocialRelation
    target_is_person: bool = False
    valid_from: int | None = None
    valid_until: int | None = None

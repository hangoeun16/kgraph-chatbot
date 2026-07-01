"""Enumerations for the knowledge graph domain.

Family relation types are inspired by the FHKB (Family History Knowledge Base)
ontology design: only base relations are stored explicitly, and higher-order
relations (grandparent, uncle, cousin, etc.) are inferred via graph traversal.
"""

from enum import Enum


class Sex(str, Enum):
    """Biological sex, used for gendered relation inference (e.g. uncle vs aunt)."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class FamilyRelation(str, Enum):
    """Base family relations stored as graph edges.

    Following the FHKB principle, only these four primitive relations are
    persisted. All other kin relations (grandparent, uncle, cousin, etc.)
    are derived through Cypher path queries at retrieval time.
    """

    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SPOUSE_OF = "spouse_of"
    SIBLING_OF = "sibling_of"


class SocialRelation(str, Enum):
    """Non-family relations between characters or between a character and a concept."""

    LIKES = "likes"
    LOVES = "loves"
    DISLIKES = "dislikes"
    KNOWS = "knows"
    FRIENDS_WITH = "friends_with"
    WORKS_WITH = "works_with"
    OWNS = "owns"
    WANTS = "wants"

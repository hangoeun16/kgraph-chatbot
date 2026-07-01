"""Structured output schemas for LLM extraction steps.

Each schema defines the expected response format when the LLM is asked
to extract information from user messages. These are passed directly to
OpenAI's structured output API so the response is guaranteed to conform.

Replaces the old approach of prompting for raw JSON and manually parsing
with string slicing (find('[') / find('{')).
"""

from pydantic import BaseModel

from models.enums import FamilyRelation, SocialRelation


# --- Family extraction ---


class ExtractedFamilyRelation(BaseModel):
    """A single family relationship extracted from user text.

    Attributes:
        person1: Name of the first person.
        person2: Name of the second person.
        relation: The family relation from person1's perspective
            (e.g. parent_of means person1 is parent of person2).
    """

    person1: str
    person2: str
    relation: FamilyRelation


class ExtractedChildCount(BaseModel):
    """A count of unnamed children for a parent.

    Used when the user says something like "they have 3 kids" without
    providing names. Placeholder nodes will be created for each child.

    Attributes:
        parent: Name of the parent.
        count: Number of unnamed children.
    """

    parent: str
    count: int


class ExtractedChildNaming(BaseModel):
    """A name assignment for a previously unnamed placeholder child.

    Used when the user provides a name for an existing placeholder
    (e.g. "The oldest is called Rick").

    Attributes:
        parent: Name of the parent whose placeholder child is being named.
        child_name: The actual name to assign.
    """

    parent: str
    child_name: str


class FamilyExtractionResult(BaseModel):
    """Combined output of the family extraction step.

    Attributes:
        relationships: Newly identified family relationships.
        child_counts: Parents with unnamed children counts.
        child_namings: Placeholder children being given real names.
    """

    relationships: list[ExtractedFamilyRelation] = []
    child_counts: list[ExtractedChildCount] = []
    child_namings: list[ExtractedChildNaming] = []


# --- Attribute extraction ---


class ExtractedAttribute(BaseModel):
    """A personal attribute extracted from user text.

    Attributes:
        person: Name of the person this attribute belongs to.
        key: Attribute name (e.g. "age", "occupation", "hobby").
        value: Attribute value (e.g. "30", "engineer", "painting").
    """

    person: str
    key: str
    value: str


class AttributeExtractionResult(BaseModel):
    """Output of the attribute extraction step."""

    attributes: list[ExtractedAttribute] = []


# --- Social relationship extraction ---


class ExtractedSocialRelation(BaseModel):
    """A non-family relationship extracted from user text.

    Attributes:
        person: Name of the source character.
        relation: The social relation type.
        target: Name of the target (person or concept).
        target_is_person: True if the target is a character, False if concept.
    """

    person: str
    relation: SocialRelation
    target: str
    target_is_person: bool = False


class SocialExtractionResult(BaseModel):
    """Output of the social relationship extraction step."""

    relationships: list[ExtractedSocialRelation] = []


# --- Conflict detection ---


class ConflictCheckResult(BaseModel):
    """Output of the conflict detection step.

    A conflict is a direct factual contradiction (e.g. two different ages
    for the same person). Adding new information is not a conflict.

    Attributes:
        has_conflict: Whether a genuine conflict was found.
        subject: Who the conflict is about.
        existing_statement: The previously stored fact.
        new_statement: The contradicting new statement.
        explanation: Why this is a conflict.
    """

    has_conflict: bool = False
    subject: str = ""
    existing_statement: str = ""
    new_statement: str = ""
    explanation: str = ""


# --- Placeholder naming detection ---


class NamingDetectionResult(BaseModel):
    """Output of the placeholder naming detection step.

    Determines whether the user is providing names for previously
    unnamed placeholder characters, so that conflict detection
    can be skipped for those statements.

    Attributes:
        is_naming_unnamed: True if the message is naming placeholders.
    """

    is_naming_unnamed: bool = False

"""Character (person) node models with temporal awareness.

Each character tracks when they exist in the narrative timeline,
allowing the system to filter out characters who haven't been born yet
or are otherwise absent at a given point in the story.
"""

from pydantic import BaseModel, Field

from models.enums import Sex


class Character(BaseModel):
    """A person node in the knowledge graph.

    Attributes:
        name: Display name of the character.
        sex: Biological sex, used for gendered relation inference.
        is_placeholder: True if this is an unnamed character (e.g. "Kim's child #1")
            that can be renamed later when the user provides an actual name.
        exists_from: The event order at which this character first appears
            in the narrative. None means they exist from the very beginning.
        exists_until: The event order at which this character exits the
            narrative (e.g. death). None means they persist indefinitely.
    """

    name: str
    sex: Sex = Sex.UNKNOWN
    is_placeholder: bool = False
    exists_from: int | None = None
    exists_until: int | None = None


class CharacterAttribute(BaseModel):
    """A time-aware attribute attached to a character.

    Attributes can change across the timeline. For example, a character's
    occupation might differ between their youth and adulthood.

    Attributes:
        key: Attribute name (e.g. "age", "occupation", "personality").
        value: Attribute value as a string.
        valid_from: Event order from which this attribute holds.
        valid_until: Event order at which this attribute no longer holds.
    """

    key: str
    value: str
    valid_from: int | None = None
    valid_until: int | None = None

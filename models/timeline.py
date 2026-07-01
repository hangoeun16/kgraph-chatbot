"""Timeline tracking and temporal inference models.

These models support the core temporal reasoning feature: detecting when
the user shifts the narrative to a different point in time (explicitly or
implicitly) and maintaining a structured event log so the graph can be
queried at any point in the story.
"""

from pydantic import BaseModel, Field


class TimelineEvent(BaseModel):
    """A single event recorded on the narrative timeline.

    Events are ordered sequentially and serve as anchors for temporal
    filtering. Characters and relationships reference event orders via
    their exists_from / valid_from fields.

    Attributes:
        order: Sequential position on the timeline (1, 2, 3, ...).
        description: Brief description of what happened
            (e.g. "Kim and Jim got married").
        trigger_utterance: The user message that caused this event
            to be recorded.
    """

    order: int
    description: str
    trigger_utterance: str = ""


class TimelineState(BaseModel):
    """The full state of the narrative timeline.

    Tracks all recorded events and which point the conversation is
    currently focused on.

    Attributes:
        events: Ordered list of all recorded timeline events.
        active_event_order: The event order that the conversation is
            currently situated at. Graph queries filter by this value.
    """

    events: list[TimelineEvent] = Field(default_factory=list)
    active_event_order: int = 0


class TimeShiftDetection(BaseModel):
    """LLM output for detecting temporal shifts in user messages.

    On every user turn, the LLM determines whether the message implies
    a shift to a different point in the narrative timeline.

    Covers two levels of temporal expression:
    - Explicit: "Let's go back to 10 years ago"
    - Semi-implicit: "During their honeymoon" (requires cross-referencing
      the event log to determine that the honeymoon was before the
      children were born)

    Attributes:
        has_time_shift: Whether the message implies a timeline change.
        inferred_period: Human-readable description of the target period
            (e.g. "right after marriage, before children").
        target_event_order: The event order to jump to, based on the
            existing event log. None if no shift detected.
        reasoning: Brief explanation of how the time shift was inferred.
    """

    has_time_shift: bool = False
    inferred_period: str = ""
    target_event_order: int | None = None
    reasoning: str = ""

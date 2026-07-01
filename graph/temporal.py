"""Timeline manager for temporal narrative tracking.

Maintains the ordered event log and the active time position.
Works with the graph store to persist timeline events and with
the LLM to detect time shifts in user messages.
"""

from models.timeline import TimelineEvent, TimelineState, TimeShiftDetection
from graph.store import GraphStore
from graph import queries


class TimelineManager:
    """Manages the narrative timeline and temporal state.

    Responsibilities:
    - Recording new events as the story progresses.
    - Tracking which time point the conversation is currently at.
    - Providing the event log as context for time-shift detection.
    - Updating the active time point when a shift is detected.

    Usage:
        timeline = TimelineManager(graph_store)
        await timeline.load()

        # Record a new event
        await timeline.record_event("Kim and Jim got married", utterance)

        # Apply a detected time shift
        timeline.apply_shift(shift_detection)
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._state = TimelineState()

    @property
    def active_event_order(self) -> int:
        """The event order the conversation is currently situated at."""
        return self._state.active_event_order

    @property
    def events(self) -> list[TimelineEvent]:
        """All recorded timeline events in chronological order."""
        return self._state.events

    async def load(self) -> None:
        """Load existing timeline events from the graph database."""
        records = await self._store._run(queries.GET_ALL_TIMELINE_EVENTS)
        self._state.events = [
            TimelineEvent(
                order=r["e"].get("event_order", 0),
                description=r["e"].get("description", ""),
                trigger_utterance=r["e"].get("trigger_utterance", ""),
            )
            for r in records
        ]
        if self._state.events:
            self._state.active_event_order = self._state.events[-1].order

    async def record_event(
        self, description: str, trigger_utterance: str = ""
    ) -> TimelineEvent:
        """Record a new event and advance the timeline.

        Args:
            description: What happened (e.g. "Kim and Jim got married").
            trigger_utterance: The user message that caused this event.

        Returns:
            The newly created TimelineEvent.
        """
        next_order = self._state.active_event_order + 1

        event = TimelineEvent(
            order=next_order,
            description=description,
            trigger_utterance=trigger_utterance,
        )

        await self._store._run(
            queries.CREATE_TIMELINE_EVENT,
            order=event.order,
            description=event.description,
            trigger_utterance=event.trigger_utterance,
        )

        self._state.events.append(event)
        self._state.active_event_order = next_order
        return event

    def apply_shift(self, detection: TimeShiftDetection) -> None:
        """Apply a detected time shift to update the active time point.

        Args:
            detection: The LLM's time-shift detection result.
        """
        if detection.has_time_shift and detection.target_event_order is not None:
            self._state.active_event_order = detection.target_event_order

    def build_timeline_summary(self) -> str:
        """Build a human-readable summary of the timeline for LLM context.

        This is passed to the time-shift detection prompt so the LLM
        can cross-reference events when interpreting semi-implicit
        time expressions like "during their honeymoon".

        Returns:
            Formatted timeline string, or empty string if no events.
        """
        if not self._state.events:
            return ""

        lines = [f"Current timeline (active point: {self._state.active_event_order}):"]
        for event in self._state.events:
            marker = " <-- active" if event.order == self._state.active_event_order else ""
            lines.append(f"  Event {event.order}: {event.description}{marker}")

        return "\n".join(lines)

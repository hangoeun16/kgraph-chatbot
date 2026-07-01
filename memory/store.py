"""Conversation turn storage.

Keeps a sliding window of recent conversation turns. When the buffer
exceeds a configurable limit, the oldest turns are passed to the
summarizer for compression, and replaced by a running summary.

This solves the problem where the old system had no conversation
history at all — the LLM only saw graph facts, not what was actually
said. With this, "what did I say 3 turns ago?" has an answer.
"""

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the conversation.

    Attributes:
        role: Who sent the message ("user" or "assistant").
        content: The message text.
        event_order: The timeline event order at the time of this turn,
            enabling reconstruction of which time point the conversation
            was at during each exchange.
    """

    role: str
    content: str
    event_order: int = 0


class ConversationBuffer(BaseModel):
    """Sliding window buffer for conversation history.

    Stores the most recent turns in full, plus a compressed summary
    of older turns. The summary is updated each time the buffer
    overflows its window size.

    Attributes:
        turns: List of recent conversation turns (within the window).
        summary: Compressed summary of turns that have been evicted
            from the window.
        window_size: Maximum number of turns to keep in full before
            triggering summarization.
    """

    turns: list[ConversationTurn] = Field(default_factory=list)
    summary: str = ""
    window_size: int = 20

    @property
    def needs_summarization(self) -> bool:
        """True if the buffer has exceeded its window size."""
        return len(self.turns) > self.window_size

    @property
    def overflow_turns(self) -> list[ConversationTurn]:
        """The oldest turns that should be summarized and evicted."""
        if not self.needs_summarization:
            return []
        overflow_count = len(self.turns) - self.window_size
        return self.turns[:overflow_count]

    def add_turn(self, role: str, content: str, event_order: int = 0) -> None:
        """Add a new turn to the buffer."""
        self.turns.append(
            ConversationTurn(
                role=role, content=content, event_order=event_order
            )
        )

    def evict_overflow(self) -> None:
        """Remove overflow turns from the buffer after summarization."""
        if self.needs_summarization:
            overflow_count = len(self.turns) - self.window_size
            self.turns = self.turns[overflow_count:]

    def format_turns(self) -> str:
        """Format recent turns as a readable string for the LLM.

        Returns:
            A formatted conversation log like:
                User: Kim is married to Jim.
                Assistant: Got it! Kim and Jim are married.
                User: They have 3 kids.
        """
        if not self.turns:
            return ""

        lines = []
        for turn in self.turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the buffer and summary."""
        self.turns.clear()
        self.summary = ""

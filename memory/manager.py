"""Conversation memory manager.

Orchestrates the conversation buffer and summarizer to produce
a unified context string for the LLM. The context has three layers:

1. Graph context (from Neo4j) — structured facts about characters
   and relationships at the current time point.
2. Conversation summary — compressed history of older exchanges.
3. Recent turns — the last N turns in full.

This layered approach means the LLM always has:
- All known facts (graph)
- Awareness of what was discussed before (summary)
- Full detail of recent exchanges (buffer)
"""

from extraction.client import LLMClient
from memory.store import ConversationBuffer
from memory.summarizer import ConversationSummarizer


class MemoryManager:
    """Manages conversation memory with hybrid buffer/summary strategy.

    Usage:
        memory = MemoryManager(llm_client, window_size=20)

        # After each user message
        memory.add_user_turn("Kim is married to Jim", event_order=2)

        # After each assistant response
        memory.add_assistant_turn("Got it! Kim and Jim are married.", event_order=2)

        # Before generating a response
        context = await memory.build_context(graph_context="Kim is parent of Mia.")
    """

    def __init__(
        self,
        client: LLMClient,
        window_size: int = 20,
    ) -> None:
        self._buffer = ConversationBuffer(window_size=window_size)
        self._summarizer = ConversationSummarizer(client)

    def add_user_turn(self, content: str, event_order: int = 0) -> None:
        """Record a user message."""
        self._buffer.add_turn("user", content, event_order)

    def add_assistant_turn(self, content: str, event_order: int = 0) -> None:
        """Record an assistant response."""
        self._buffer.add_turn("assistant", content, event_order)

    async def compress_if_needed(self) -> None:
        """Summarize and evict overflow turns if the buffer is full.

        Call this periodically (e.g. after each exchange) to keep
        the buffer within its window size.
        """
        if not self._buffer.needs_summarization:
            return

        overflow = self._buffer.overflow_turns
        self._buffer.summary = await self._summarizer.summarize(
            existing_summary=self._buffer.summary,
            overflow_turns=overflow,
        )
        self._buffer.evict_overflow()

    async def build_context(self, graph_context: str) -> str:
        """Build the full context string for the LLM prompt.

        Combines graph facts, conversation summary, and recent turns
        into a single context block.

        Args:
            graph_context: The current graph state as a string
                (from GraphStore.build_context_string).

        Returns:
            Formatted context string with all three layers.
        """
        sections = []

        # Layer 1: Graph context (structured facts)
        if graph_context and "empty" not in graph_context.lower():
            sections.append(f"## Known facts\n{graph_context}")

        # Layer 2: Conversation summary (compressed older history)
        if self._buffer.summary:
            sections.append(
                f"## Earlier conversation summary\n{self._buffer.summary}"
            )

        # Layer 3: Recent turns (full detail)
        recent = self._buffer.format_turns()
        if recent:
            sections.append(f"## Recent conversation\n{recent}")

        if not sections:
            return "No previous conversation yet."

        return "\n\n".join(sections)

    def clear(self) -> None:
        """Reset all conversation memory."""
        self._buffer.clear()

    @property
    def turn_count(self) -> int:
        """Total number of turns in the buffer."""
        return len(self._buffer.turns)

    @property
    def has_summary(self) -> bool:
        """True if older turns have been compressed into a summary."""
        return bool(self._buffer.summary)

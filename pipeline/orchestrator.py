"""Pipeline orchestrator: the central chat processing flow.

Replaces the old GirlyChatbot class. Coordinates all components:
moderation → time-shift detection → conflict check → extraction →
graph application → response streaming.

Each stage yields SSE events so the frontend can show natural
language status messages while the user waits.
"""

from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from models.events import StreamEvent, StreamEventType
from models.extraction import ConflictCheckResult
from extraction.client import LLMClient
from extraction.service import ExtractionService
from graph.store import GraphStore
from graph.temporal import TimelineManager
from memory.manager import MemoryManager
from pipeline.moderator import check_content_safety
from pipeline.applier import GraphApplier


# Chatbot personality prompt (neutral tone, no longer "girly")
_SYSTEM_PROMPT = """You are a helpful AI assistant that tracks characters \
and relationships in conversations.

IMPORTANT RULES:
- The person chatting with you is always "user" — never confuse them \
with character names mentioned in the story.
- ONLY use information listed in the story context provided.
- If the context says the knowledge graph is empty or doesn't mention \
someone, you know NOTHING about them.
- NEVER fabricate, assume, or hallucinate relationships or facts \
not in the context.
- Always accept new information the user tells you — do not question \
or correct them.
- Family relationships beyond what is directly stated (grandparent, \
uncle, cousin, etc.) are inferred automatically by the system. \
You may reference them if they appear in the context.

When responding:
- Be concise and helpful.
- Reference known character details naturally.
"""


class ChatPipeline:
    """Async pipeline that processes a user message end-to-end.

    Usage:
        pipeline = ChatPipeline(
            openai_api_key="sk-...",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )
        await pipeline.start()

        async for event in pipeline.handle_message("Kim married Jim"):
            # StreamEvent(type="status", content="Reviewing content...")
            # StreamEvent(type="token", content="Got")
            # StreamEvent(type="token", content=" it!")
            # StreamEvent(type="done", content="")
            ...

        await pipeline.shutdown()
    """

    def __init__(
        self,
        openai_api_key: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        model: str = "gpt-4o",
    ) -> None:
        self._openai_client = AsyncOpenAI(api_key=openai_api_key)
        self._llm = LLMClient(api_key=openai_api_key, model=model)
        self._extractor = ExtractionService(self._llm)
        self._store = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
        self._timeline = TimelineManager(self._store)
        self._applier = GraphApplier(self._store, self._timeline)
        self._memory = MemoryManager(self._llm, window_size=20)

    async def start(self) -> None:
        """Initialize connections and load state."""
        await self._store.connect()
        await self._timeline.load()

    async def shutdown(self) -> None:
        """Close connections."""
        await self._store.close()

    # ------------------------------------------------------------------
    # Main message handler
    # ------------------------------------------------------------------

    async def handle_message(
        self, message: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process a user message through the full pipeline.

        Yields StreamEvent objects for the frontend:
        - STATUS events during pre-processing stages
        - TOKEN events during response streaming
        - DONE when complete
        - REJECTED if moderation or conflict blocks the message
        - ERROR on failure

        Args:
            message: The user's input text.

        Yields:
            StreamEvent instances.
        """
        try:
            # --- Stage 1: Content moderation ---
            yield _status("Reviewing content...")

            moderation = await check_content_safety(
                self._openai_client, message
            )
            if not moderation.is_safe:
                categories = ", ".join(moderation.flagged_categories)
                yield _rejected(f"Content flagged: {categories}")
                return

            # --- Stage 2: Time-shift detection ---
            yield _status("Checking the timeline...")

            context = await self._store.build_context_string(
                self._timeline.active_event_order
            )
            timeline_summary = self._timeline.build_timeline_summary()

            shift = await self._extractor.detect_time_shift(
                message=message,
                context=context,
                timeline_summary=timeline_summary,
            )

            if shift.has_time_shift:
                self._timeline.apply_shift(shift)
                # Rebuild context at the new time point
                context = await self._store.build_context_string(
                    self._timeline.active_event_order
                )
                yield _status(f"Moving to: {shift.inferred_period}...")

            # --- Stage 3: Conflict detection ---
            yield _status("Comparing with existing information...")

            conflict = await self._check_conflict_safe(message, context)
            if conflict and conflict.has_conflict:
                yield _rejected(
                    f"Conflict: {conflict.subject} — {conflict.explanation}"
                )
                return

            # --- Stage 4: Information extraction ---
            yield _status("Understanding relationships...")

            existing_people = await self._get_existing_people()

            family_result, attr_result, social_result = (
                await self._extract_all(
                    message=message,
                    context=context,
                    existing_people=existing_people,
                    timeline_summary=timeline_summary,
                )
            )

            # --- Stage 5: Apply to graph ---
            yield _status("Updating the story graph...")

            await self._applier.apply_family(family_result)
            await self._applier.apply_attributes(attr_result)
            await self._applier.apply_social(social_result)

            # Record a timeline event if new info was extracted
            if (
                family_result.relationships
                or family_result.child_counts
                or family_result.child_namings
            ):
                description = self._summarize_family_event(family_result)
                await self._timeline.record_event(description, message)

            # --- Stage 6: Stream response ---
            yield _status("")  # Clear status message

            # Record user turn in conversation memory
            self._memory.add_user_turn(
                message, self._timeline.active_event_order
            )

            # Build unified context: graph facts + summary + recent turns
            graph_context = await self._store.build_context_string(
                self._timeline.active_event_order
            )
            full_context = await self._memory.build_context(graph_context)

            full_response = ""
            async for token in self._llm.stream_chat(
                system=_SYSTEM_PROMPT,
                context=full_context,
                user_message=message,
            ):
                full_response += token
                yield StreamEvent(type=StreamEventType.TOKEN, content=token)

            # Record assistant turn and compress if needed
            self._memory.add_assistant_turn(
                full_response, self._timeline.active_event_order
            )
            await self._memory.compress_if_needed()

            yield StreamEvent(type=StreamEventType.DONE)

        except Exception as e:
            print(f"Pipeline error: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content="Something went wrong.",
            )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    async def clear_memory(self) -> str:
        """Clear all graph data, timeline, and conversation memory."""
        await self._store.clear()
        self._timeline = TimelineManager(self._store)
        self._memory.clear()
        return "Memory cleared!"

    async def get_graph_data(self) -> dict:
        """Get graph data for frontend visualization."""
        return await self._store.get_graph_data()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_conflict_safe(
        self, message: str, context: str
    ) -> ConflictCheckResult | None:
        """Run conflict detection, skipping if naming or no context."""
        if not context or "empty" in context.lower():
            return None

        # Check if user is naming placeholders → skip conflict check
        naming = await self._extractor.check_naming(
            message=message, context=context
        )
        if naming.is_naming_unnamed:
            return None

        return await self._extractor.check_conflict(
            message=message, context=context
        )

    async def _get_existing_people(self) -> list[str]:
        """Get list of named characters from the graph."""
        records = await self._store.get_all_context()
        return [r["name"] for r in records if r.get("name")]

    async def _extract_all(
        self,
        message: str,
        context: str,
        existing_people: list[str],
        timeline_summary: str,
    ):
        """Run all three extraction steps.

        Currently sequential. Could be parallelized with asyncio.gather
        if extraction steps are confirmed independent.
        """
        family_result = await self._extractor.extract_family(
            message=message,
            context=context,
            existing_people=existing_people,
            timeline_summary=timeline_summary,
        )

        attr_result = await self._extractor.extract_attributes(
            message=message,
            context=context,
            existing_people=existing_people,
        )

        social_result = await self._extractor.extract_social(
            message=message,
            context=context,
            existing_people=existing_people,
        )

        return family_result, attr_result, social_result

    @staticmethod
    def _summarize_family_event(result) -> str:
        """Create a brief description of extracted family changes."""
        parts = []
        for rel in result.relationships:
            parts.append(
                f"{rel.person1} is {rel.relation.value} {rel.person2}"
            )
        for cc in result.child_counts:
            parts.append(f"{cc.parent} has {cc.count} children")
        for cn in result.child_namings:
            parts.append(f"{cn.parent}'s child named {cn.child_name}")
        return "; ".join(parts) if parts else "Story updated"


# ------------------------------------------------------------------
# SSE event helpers
# ------------------------------------------------------------------


def _status(content: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.STATUS, content=content)


def _rejected(content: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.REJECTED, content=content)

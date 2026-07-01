"""Extraction service: structured LLM calls for information extraction.

This module replaces the five extraction/detection methods that were
inlined in the old knowledge_graph.py. Each method now:
1. Builds the prompt from templates (prompts/)
2. Calls the LLM with a Pydantic response schema (structured output)
3. Returns a typed result — no JSON string parsing needed

The service does NOT write to the graph directly. It returns extraction
results that the pipeline orchestrator applies to the graph store.
This separation makes each piece independently testable.
"""

from models.extraction import (
    AttributeExtractionResult,
    ConflictCheckResult,
    FamilyExtractionResult,
    NamingDetectionResult,
    SocialExtractionResult,
)
from models.timeline import TimeShiftDetection
from extraction.client import LLMClient
from prompts import family, attributes, social, conflict, timeline


class ExtractionService:
    """Stateless service for LLM-based information extraction.

    Each method corresponds to one extraction step in the pipeline.
    All methods are async and return Pydantic model instances.

    Usage:
        service = ExtractionService(llm_client)

        families = await service.extract_family(
            message="Kim is married to Jim",
            context="No story information yet.",
            existing_people=["Kim"],
            timeline_summary="",
        )
        # families.relationships -> [ExtractedFamilyRelation(...)]
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def extract_family(
        self,
        message: str,
        context: str,
        existing_people: list[str],
        timeline_summary: str,
    ) -> FamilyExtractionResult:
        """Extract family relationships from a user message.

        Args:
            message: The user's input text.
            context: Current graph context string.
            existing_people: List of known character names.
            timeline_summary: Current timeline state summary.

        Returns:
            FamilyExtractionResult with relationships, child counts,
            and child naming actions.
        """
        people_str = ", ".join(existing_people) if existing_people else "None yet"

        user_prompt = family.USER.format(
            context=context,
            existing_people=people_str,
            timeline_summary=timeline_summary or "No events recorded yet.",
            message=message,
        )

        return await self._client.extract(
            system=family.SYSTEM,
            user=user_prompt,
            response_model=FamilyExtractionResult,
        )

    async def extract_attributes(
        self,
        message: str,
        context: str,
        existing_people: list[str],
    ) -> AttributeExtractionResult:
        """Extract personal attributes from a user message.

        Args:
            message: The user's input text.
            context: Current graph context string.
            existing_people: List of known character names.

        Returns:
            AttributeExtractionResult with extracted attributes.
        """
        people_str = ", ".join(existing_people) if existing_people else "None yet"

        user_prompt = attributes.USER.format(
            context=context,
            existing_people=people_str,
            message=message,
        )

        return await self._client.extract(
            system=attributes.SYSTEM,
            user=user_prompt,
            response_model=AttributeExtractionResult,
        )

    async def extract_social(
        self,
        message: str,
        context: str,
        existing_people: list[str],
    ) -> SocialExtractionResult:
        """Extract non-family relationships from a user message.

        Args:
            message: The user's input text.
            context: Current graph context string.
            existing_people: List of known character names.

        Returns:
            SocialExtractionResult with extracted social relationships.
        """
        people_str = ", ".join(existing_people) if existing_people else "None yet"

        user_prompt = social.USER.format(
            context=context,
            existing_people=people_str,
            message=message,
        )

        return await self._client.extract(
            system=social.SYSTEM,
            user=user_prompt,
            response_model=SocialExtractionResult,
        )

    async def check_conflict(
        self,
        message: str,
        context: str,
    ) -> ConflictCheckResult:
        """Check if a message conflicts with existing information.

        Args:
            message: The user's input text.
            context: Current graph context string.

        Returns:
            ConflictCheckResult indicating whether a conflict was found.
        """
        user_prompt = conflict.CONFLICT_USER.format(
            context=context,
            message=message,
        )

        return await self._client.extract(
            system=conflict.CONFLICT_SYSTEM,
            user=user_prompt,
            response_model=ConflictCheckResult,
        )

    async def check_naming(
        self,
        message: str,
        context: str,
    ) -> NamingDetectionResult:
        """Check if the user is naming previously unnamed characters.

        Args:
            message: The user's input text.
            context: Current graph context string.

        Returns:
            NamingDetectionResult indicating whether naming is detected.
        """
        user_prompt = conflict.NAMING_USER.format(
            context=context,
            message=message,
        )

        return await self._client.extract(
            system=conflict.NAMING_SYSTEM,
            user=user_prompt,
            response_model=NamingDetectionResult,
        )

    async def detect_time_shift(
        self,
        message: str,
        context: str,
        timeline_summary: str,
    ) -> TimeShiftDetection:
        """Detect if the message implies a narrative time shift.

        This is the core temporal reasoning call. The LLM examines
        the message against the recorded timeline to determine whether
        the story has moved to a different point in time.

        Args:
            message: The user's input text.
            context: Current graph context at the active time point.
            timeline_summary: Formatted timeline event log.

        Returns:
            TimeShiftDetection with shift details if detected.
        """
        user_prompt = timeline.USER.format(
            timeline_summary=timeline_summary or "No events recorded yet.",
            context=context,
            message=message,
        )

        return await self._client.extract(
            system=timeline.SYSTEM,
            user=user_prompt,
            response_model=TimeShiftDetection,
        )

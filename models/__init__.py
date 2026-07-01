"""Pydantic models for the KGraph chatbot.

Organized by domain:
- enums: Relation types, sex
- character: Person nodes with temporal existence
- relationship: Family and social edges with temporal validity
- timeline: Event log and time-shift detection
- extraction: LLM structured output schemas
- events: SSE stream events and moderation
"""

from models.enums import FamilyRelation, Sex, SocialRelation
from models.character import Character, CharacterAttribute
from models.relationship import FamilyEdge, SocialEdge
from models.timeline import TimelineEvent, TimelineState, TimeShiftDetection
from models.extraction import (
    AttributeExtractionResult,
    ConflictCheckResult,
    ExtractedAttribute,
    ExtractedChildCount,
    ExtractedChildNaming,
    ExtractedFamilyRelation,
    ExtractedSocialRelation,
    FamilyExtractionResult,
    NamingDetectionResult,
    SocialExtractionResult,
)
from models.events import ModerationResult, StreamEvent, StreamEventType

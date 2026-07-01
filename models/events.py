"""Server-Sent Event and moderation models.

These models define the data structures flowing between the backend
and the frontend over the SSE connection.
"""

from enum import Enum

from pydantic import BaseModel


class StreamEventType(str, Enum):
    """Types of events sent to the frontend via SSE."""

    TOKEN = "token"
    STATUS = "status"
    DONE = "done"
    REJECTED = "rejected"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single SSE event sent to the frontend.

    Attributes:
        type: The event type (token, status, done, rejected, error).
        content: The payload. For TOKEN events this is a text chunk;
            for STATUS events it is a natural language progress message;
            for REJECTED/ERROR it is an explanation.
    """

    type: StreamEventType
    content: str = ""


class ModerationResult(BaseModel):
    """Output of the content safety check.

    Attributes:
        is_safe: True if the content passes moderation.
        flagged_categories: List of violated category names
            (e.g. "sexual content", "hate speech").
    """

    is_safe: bool = True
    flagged_categories: list[str] = []

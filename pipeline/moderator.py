"""Content safety moderation using OpenAI's Moderation API.

Async version of the original content_moderator.py.
"""

from openai import AsyncOpenAI

from models.events import ModerationResult


# Category mapping: API field name -> human-readable label
_CATEGORY_LABELS = {
    "sexual": "sexual content",
    "hate": "hate speech",
    "harassment": "harassment",
    "self_harm": "self-harm",
    "sexual_minors": "sexual content involving minors",
    "violence": "violence",
}


async def check_content_safety(
    client: AsyncOpenAI, text: str
) -> ModerationResult:
    """Check if user input contains inappropriate or harmful content.

    Args:
        client: Async OpenAI client instance.
        text: The user input text to analyze.

    Returns:
        ModerationResult with safety status and flagged categories.
    """
    try:
        response = await client.moderations.create(input=text)
        result = response.results[0]

        if not result.flagged:
            return ModerationResult(is_safe=True)

        flagged = [
            label
            for field, label in _CATEGORY_LABELS.items()
            if getattr(result.categories, field, False)
        ]

        return ModerationResult(is_safe=False, flagged_categories=flagged)

    except Exception as e:
        # Allow conversation to continue on moderation API failure
        print(f"Moderation error: {e}")
        return ModerationResult(is_safe=True)

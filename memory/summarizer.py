"""Conversation summarizer.

When the conversation buffer overflows, this module compresses the
oldest turns into a concise summary using the LLM. The summary is
appended to any existing summary, creating a rolling compression
of the full conversation history.

This is the token-cost vs. context trade-off:
- Keeping all turns: accurate but expensive, eventually exceeds context window.
- Summarizing everything: cheap but lossy.
- Hybrid (this approach): recent turns in full + compressed older history.
"""

from extraction.client import LLMClient
from memory.store import ConversationTurn

SUMMARIZE_SYSTEM = (
    "Summarize the conversation below into a concise paragraph. "
    "Preserve key facts, character details, relationship changes, "
    "and any timeline shifts. Do not add information that isn't present."
)

SUMMARIZE_USER = """Existing summary (if any):
{existing_summary}

New conversation turns to incorporate:
{turns}

Write a concise updated summary that merges the existing summary with the new turns. Focus on facts that would be important for continuing the conversation."""


class ConversationSummarizer:
    """Compresses conversation turns into summaries via the LLM.

    Usage:
        summarizer = ConversationSummarizer(llm_client)
        new_summary = await summarizer.summarize(
            existing_summary="Kim and Jim were introduced as a married couple.",
            overflow_turns=[...],
        )
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def summarize(
        self,
        existing_summary: str,
        overflow_turns: list[ConversationTurn],
    ) -> str:
        """Merge overflow turns into the existing summary.

        Args:
            existing_summary: The current running summary (may be empty).
            overflow_turns: Turns that are being evicted from the buffer.

        Returns:
            Updated summary string incorporating the new turns.
        """
        if not overflow_turns:
            return existing_summary

        turns_text = "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
            for t in overflow_turns
        )

        prompt = SUMMARIZE_USER.format(
            existing_summary=existing_summary or "(no previous summary)",
            turns=turns_text,
        )

        # Use raw completion instead of structured output since we just
        # need a plain text summary, not a typed object.
        response = await self._client._client.chat.completions.create(
            model=self._client._model,
            temperature=0.3,  # Lower temperature for factual summarization
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content.strip()

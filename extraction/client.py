"""Async OpenAI client with structured output support.

Wraps the OpenAI SDK to provide type-safe LLM calls. Instead of
prompting for raw JSON and slicing it with string operations, each
call specifies a Pydantic model as the response format. The API
guarantees the response conforms to the schema.

Replaces:
    response_text = response.content.strip()
    start = response_text.find('[')        # <-- this is gone
    end = response_text.rfind(']') + 1     # <-- this too
    result = json.loads(response_text[start:end])

With:
    result = await client.extract(prompt, ResponseModel)
    # result is already a validated Pydantic instance
"""

from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Async OpenAI client with structured output.

    Usage:
        client = LLMClient(api_key="sk-...")
        result = await client.extract(
            system="Extract family relationships.",
            user="Kim is married to Jim.",
            response_model=FamilyExtractionResult,
        )
        # result is a FamilyExtractionResult instance
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def extract(
        self,
        system: str,
        user: str,
        response_model: type[T],
    ) -> T:
        """Call the LLM with structured output and return a typed result.

        Args:
            system: System prompt describing the task.
            user: User prompt with context and message.
            response_model: Pydantic model class defining the expected
                response schema. Passed to the API as response_format.

        Returns:
            An instance of response_model populated by the LLM.

        Raises:
            openai.APIError: If the API call fails.
            pydantic.ValidationError: If the response doesn't match the schema
                (should not happen with structured output, but just in case).
        """
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_model,
        )

        return response.choices[0].message.parsed

    async def stream_chat(
        self,
        system: str,
        context: str,
        user_message: str,
    ):
        """Stream a chat response token by token.

        This is used for the final response generation (not extraction).
        Yields content strings as they arrive.

        Args:
            system: The chatbot personality prompt.
            context: Graph context from the current time point.
            user_message: The user's input message.

        Yields:
            Content string chunks from the model.
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "system", "content": f"Story context:\n{context}"},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

"""LLM-based information extraction layer.

Modules:
- client: Async OpenAI wrapper with structured output.
- service: Extraction methods for each pipeline step.
"""

from extraction.client import LLMClient
from extraction.service import ExtractionService

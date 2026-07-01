"""LLM prompt templates for each extraction and detection step.

Each module provides SYSTEM and USER prompt strings. The USER prompts
use Python format strings ({context}, {message}, etc.) that are filled
at call time with the current graph context and user message.
"""

from prompts import family, attributes, social, conflict, timeline

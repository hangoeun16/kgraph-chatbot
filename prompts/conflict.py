"""Prompt templates for conflict detection and placeholder naming detection."""

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

CONFLICT_SYSTEM = "Detect factual contradictions between new and existing information. Respond with the required structured format."

CONFLICT_USER = """Check whether the new statement directly contradicts any existing information.

## Existing story information
{context}

## New statement
"{message}"

## What counts as a conflict
A conflict is ONLY a direct factual contradiction:
- Same character has TWO DIFFERENT values for the same attribute (e.g. "Jim is 45" then "Jim is 30")
- Same character has TWO DIFFERENT spouses (e.g. "Jim married Mary" then "Jim married Sue")
- A stated fact is directly negated

## What is NOT a conflict
- Adding new characters or relationships
- Providing more details about existing characters
- Naming previously unnamed characters
- Expanding the story with new information

Default to no conflict if uncertain.
"""

# ---------------------------------------------------------------------------
# Placeholder naming detection
# ---------------------------------------------------------------------------

NAMING_SYSTEM = "Determine if the user is naming previously unnamed characters. Respond with the required structured format."

NAMING_USER = """Is the user providing names for previously unnamed characters?

## Existing story information
{context}

## New statement
"{message}"

## Instructions
Return true only if the message appears to be giving a real name to a character that currently exists as an unnamed placeholder (e.g. "the oldest is called Rick" when there are unnamed children).
"""

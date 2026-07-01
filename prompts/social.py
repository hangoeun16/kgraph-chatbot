"""Prompt template for non-family relationship extraction."""

SYSTEM = "Extract social relationships from the user's message. Respond with the required structured format."

USER = """Analyze the following message and extract any non-family relationships.

## Current story context
{context}

## Known characters
{existing_people}

## User message
"{message}"

## Instructions
Look for social connections such as: likes, loves, dislikes, knows, friends_with, works_with, owns, wants.

For each relationship, identify:
- The source character
- The type of relation
- The target (can be another character or a concept like "pizza", "music")
- Whether the target is a person or a concept

If no social relationships are found, return an empty list.
"""

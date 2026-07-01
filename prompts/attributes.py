"""Prompt template for personal attribute extraction."""

SYSTEM = "Extract personal attributes from the user's message. Respond with the required structured format."

USER = """Analyze the following message and extract any personal attributes mentioned.

## Current story context
{context}

## Known characters
{existing_people}

## User message
"{message}"

## Instructions
Extract attributes such as: age, occupation, location, personality, hobby, education, gender, nickname, appearance, or any other descriptive fact about a character.

Each attribute should specify which character it belongs to, the attribute name, and its value.

If no personal attributes are found, return an empty list.
"""

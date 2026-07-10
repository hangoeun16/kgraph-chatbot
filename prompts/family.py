"""Prompt template for family relationship extraction."""

SYSTEM = "Extract family relationships from the user's message. Respond with the required structured format."

USER = """Analyze the following message and extract any family relationships mentioned.

## Current story context
{context}

## Known characters
{existing_people}

## Current timeline
{timeline_summary}

## User message
"{message}"

## Instructions
Identify family relationships such as marriage, parent-child, and sibling connections.

For each relationship found, classify it as one of:
- parent_of: person1 is a parent of person2
- child_of: person1 is a child of person2
- spouse_of: person1 is married to person2
- sibling_of: person1 is a sibling of person2

Normalize variations: "father/mother" → parent_of, "son/daughter" → child_of, "husband/wife" → spouse_of, "brother/sister" → sibling_of.

If the message mentions a count of unnamed children (e.g. "they have 3 kids"), record it as a SINGLE child count listing every parent who shares those children. Resolve pronouns like "they" to the couple in context (e.g. parents: ["Kim", "Jim"]). Do not emit a separate entry per parent, and do not split shared children across parents.

If the message provides a name for a previously unnamed child (e.g. "the oldest is Rick" when a parent already has unnamed children), record it as a child naming action.

If no family relationships are found, return empty lists.
"""
"""Prompt template for temporal shift detection.

This is the core of the temporal reasoning feature. On every user turn,
the LLM examines the message against the existing timeline to determine
whether the narrative has shifted to a different point in time.

Handles two levels of temporal expression:
- Explicit: "Let's go back to 10 years ago", "Fast forward to when they're old"
- Semi-implicit: "During their honeymoon", "When Kim was still a student"
  (requires cross-referencing the event log to resolve)
"""

SYSTEM = "Detect temporal shifts in the user's message by cross-referencing the timeline. Respond with the required structured format."

USER = """Analyze whether this message implies a shift to a different point in the narrative timeline.

## Recorded timeline
{timeline_summary}

## Current story state at active time point
{context}

## User message
"{message}"

## Instructions
Determine if the message moves the story to a different time point.

Examples of time shifts:
- "Let's go back to before they met" → shift to before the meeting event
- "During their honeymoon" → shift to right after the marriage event
- "When the kids were young" → shift to shortly after children were born
- "10 years later" → shift forward from current point

Examples of NO time shift:
- "Tell me about Kim" → asking about current state, no shift
- "Kim likes pizza" → adding info at current time, no shift
- "What happened before?" → asking about past, but not shifting to it

Cross-reference the recorded timeline to find the closest matching event order.
If the message references a period between two events, use the earlier event's order.
If no matching event exists, suggest a new event order and describe the period.
"""

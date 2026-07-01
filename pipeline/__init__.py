"""Chat processing pipeline.

Modules:
- orchestrator: Main message handling flow.
- moderator: Content safety checks.
- applier: Writes extraction results to the graph.
"""

from pipeline.orchestrator import ChatPipeline

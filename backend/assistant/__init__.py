"""AI assistant for the Antarctic Navigation Decision Support System.

The assistant answers questions about sea-ice forecasts, iceberg drift
predictions, route recommendations, fuel estimates and distances. It:

* gathers real context from the DSS backend services (via ``tools.py``),
* formats a context report and system instructions (via ``prompts.py``),
* optionally asks a configurable LLM provider to phrase the answer
  (``assistant_service.py``), falling back to honest template answers when
  no provider/credentials are configured.

No API keys are ever exposed to the frontend. Credentials live in process
env/config only (see ``config.Settings``).
"""
from assistant.assistant_service import AssistantService, assistant_service
from assistant.prompts import SYSTEM_PROMPT
from assistant.tools import TOOLS, run_tool

__all__ = [
    "AssistantService",
    "assistant_service",
    "SYSTEM_PROMPT",
    "TOOLS",
    "run_tool",
]
"""Chat endpoint for the AI assistant.

Security notes:
* The frontend only sends the question + optional session history.
* LLM credentials are read inside the process (``config.Settings``) and are
  never part of any response or CORS-exposed header.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from assistant.assistant_service import assistant_service
from schemas.models import AssistantChatRequest, AssistantChatResponse

router = APIRouter(prefix="/assistant", tags=["assistant"])

# Limit session history kept server-side to bound prompt sizes.
MAX_HISTORY = 14


@router.post("/chat", response_model=AssistantChatResponse)
async def chat(req: AssistantChatRequest) -> AssistantChatResponse:
    """POST /api/assistant/chat

    Returns an answer grounded in the DSS backend plus the sources that were
    used, the LLM status, the demo flag and any warnings.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    history = [m.model_dump() for m in req.history[-MAX_HISTORY:]]
    try:
        out = assistant_service.answer(
            question=question,
            history=history,
            horizon_hours=req.horizon_hours,
        )
    except Exception as exc:  # noqa: BLE001 - never leak internals
        raise HTTPException(status_code=503, detail=f"Assistant unavailable: {exc}") from exc

    return AssistantChatResponse(**out)
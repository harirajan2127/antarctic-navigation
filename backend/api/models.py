"""Model registry endpoint — aggregate trained-model status across pipelines."""
from __future__ import annotations

from fastapi import APIRouter

from schemas.models import ModelRegistryResponse
from services.model_registry import model_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status", response_model=ModelRegistryResponse)
async def models_status() -> ModelRegistryResponse:
    """GET /api/models/status — trained model availability for the UI."""
    return ModelRegistryResponse(**model_registry())
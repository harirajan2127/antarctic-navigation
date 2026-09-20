"""Dataset inventory and status endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from services.dataset_service import from_database, real_data_available
from schemas.models import DatasetInfo, DatasetsResponse, DatasetStatusResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetsResponse)
async def list_datasets() -> DatasetsResponse:
    """GET /api/datasets"""
    items = [DatasetInfo(**row) for row in from_database()]
    return DatasetsResponse(
        datasets=items,
        demo_mode=False,
        warning=None,
    )


@router.get("/status", response_model=DatasetStatusResponse)
async def dataset_status() -> DatasetStatusResponse:
    """GET /api/datasets/status"""
    items = [DatasetInfo(**row) for row in from_database()]
    return DatasetStatusResponse(
        datasets=items,
        real_data_available=real_data_available(items),
        demo_mode=False,
        warning=None,
    )
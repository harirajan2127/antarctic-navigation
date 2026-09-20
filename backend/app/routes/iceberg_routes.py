"""Iceberg trajectory prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.iceberg.data_loader import IcebergDataLoader
from app.services.iceberg.predictor import IcebergTrajectoryService

router = APIRouter(prefix="/icebergs", tags=["icebergs"])

_predictor = IcebergTrajectoryService("persistence")
_data_loader = IcebergDataLoader()


@router.get("/predictions")
def predict_icebergs(
    horizon_hours: int = Query(24, ge=2, le=72),
    model: str = Query("persistence"),
):
    """Predict iceberg positions at the requested forecast horizon."""
    service = _predictor if model == "persistence" else IcebergTrajectoryService(model)
    icebergs = _data_loader.load()
    return {"icebergs": service.predict(icebergs, horizon_hours=horizon_hours)}


@router.get("/forecast-grid")
def forecast_grid():
    """Return iceberg predictions across all supported horizons."""
    icebergs = _data_loader.load()
    return {"icebergs": _predictor.predict_multi_horizon(icebergs)}


@router.get("/models")
def available_models():
    """List the available iceberg trajectory models."""
    return {"models": _predictor.available_models()}
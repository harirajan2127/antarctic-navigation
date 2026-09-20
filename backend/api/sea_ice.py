"""Sea-ice data and prediction endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from config import DEMO_WARNING
from schemas.models import (
    SeaIceCurrentResponse,
    SeaIceForecastResponse,
    SeaIcePredictRequest,
)
from services.sea_ice_service import sea_ice_service

router = APIRouter(prefix="/sea-ice", tags=["sea-ice"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_response(data: dict, forecast_time: datetime) -> SeaIceForecastResponse:
    valid_time = data.get("valid_time")
    if isinstance(valid_time, str):
        try:
            valid_time = datetime.fromisoformat(valid_time)
        except Exception:
            valid_time = None
    return SeaIceForecastResponse(
        forecast_time=forecast_time,
        valid_time=valid_time,
        horizon_hours=data.get("horizon_hours", 24),
        model=data.get("model", "persistence"),
        lat=data.get("lat", []),
        lon=data.get("lon", []),
        concentration=data.get("concentration", []),
        mean_concentration=data.get("mean_concentration", 0.0),
        max_concentration=data.get("max_concentration", 0.0),
        coverage_pct=data.get("coverage_pct", 0.0),
        classification=data.get("classification", "unknown"),
        demo=data.get("demo", True),
        model_used_real=data.get("model_used_real", False),
        skill_note=data.get("skill_note"),
        warning=data.get("warning"),
    )


@router.get("/current", response_model=SeaIceCurrentResponse)
async def sea_ice_current() -> SeaIceCurrentResponse:
    """GET /api/sea-ice/current"""
    data = sea_ice_service.current()
    return SeaIceCurrentResponse(
        timestamp=data["timestamp"],
        lat=data.get("lat", []),
        lon=data.get("lon", []),
        concentration=data.get("concentration", []),
        classification=data.get("classification", "unknown"),
        demo=data.get("demo", True),
        source=data.get("source"),
        warning=data.get("warning"),
    )


@router.get("/forecast", response_model=SeaIceForecastResponse)
async def sea_ice_forecast(
    horizon_hours: int = Query(24, ge=2, le=168, description="Forecast horizon in hours (2–168)"),
    model: str | None = Query(None, description="persistence | random_forest | convlstm (default: SEA_ICE_MODEL)"),
) -> SeaIceForecastResponse:
    """GET /api/sea-ice/forecast?horizon_hours=24&model=persistence"""
    data = sea_ice_service.forecast(horizon_hours=horizon_hours, model=model)
    return _to_response(data, data.get("forecast_time", _utcnow()))


@router.get("/models", tags=["sea-ice"])
async def sea_ice_models() -> dict:
    """GET /api/sea-ice/models — trained-model availability for the UI."""
    return sea_ice_service.model_status()


@router.post("/predict", response_model=SeaIceForecastResponse)
async def sea_ice_predict(payload: SeaIcePredictRequest) -> SeaIceForecastResponse:
    """POST /api/sea-ice/predict"""
    data = sea_ice_service.predict(
        concentration=payload.concentration,
        lat=payload.lat,
        lon=payload.lon,
        horizon_hours=payload.horizon_hours,
        model=payload.model,
        timestamp=payload.timestamp,
    )
    return _to_response(data, data.get("forecast_time", _utcnow()))
"""Sea-ice forecasting endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.services.sea_ice.data_loader import SeaIceDataLoader
from app.services.sea_ice.forecaster import SeaIceForecastService

router = APIRouter(prefix="/sea-ice", tags=["sea-ice"])

_forecaster = SeaIceForecastService("persistence")
_data_loader = SeaIceDataLoader()


@router.get("/forecast")
def forecast_sea_ice(
    horizon_hours: int = Query(24, ge=2, le=168),
    model: str = Query("persistence"),
):
    """Return a sea-ice concentration forecast for the Antarctic region."""
    service = _forecaster if model == "persistence" else SeaIceForecastService(model)
    data = _data_loader.load()
    return service.forecast(
        data["sea_ice_concentration"],
        data["lat"],
        data["lon"],
        horizon_hours=horizon_hours,
    )


@router.get("/models")
def available_models():
    """List the available sea-ice forecasting models."""
    return {"models": _forecaster.available_models()}
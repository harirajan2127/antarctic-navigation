"""Vessel catalog endpoint (fleet configuration used by route planning)."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import load_vessels
from schemas.models import VesselInfo, VesselsResponse

router = APIRouter(prefix="/vessels", tags=["vessels"])


@router.get("", response_model=VesselsResponse)
async def list_vessels() -> VesselsResponse:
    """GET /api/vessels"""
    raw = load_vessels()
    vessels = [
        VesselInfo(
            vessel_id=str(v.get("vessel_id", "")),
            name=str(v.get("name", "")),
            vessel_type=str(v.get("type", "")),
            ice_class=v.get("ice_class"),
            cruise_speed_knots=v.get("cruise_speed_knots"),
            max_speed_knots=v.get("max_speed_knots"),
            fuel_rate_lph=v.get("fuel_rate_lph"),
            fuel_capacity_tons=v.get("fuel_capacity_tons"),
            safety_distance_nm=v.get("safety_distance_nm"),
        )
        for v in raw
        if v.get("vessel_id")
    ]
    return VesselsResponse(vessels=vessels, count=len(vessels))
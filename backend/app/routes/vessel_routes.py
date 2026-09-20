"""Vessel endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import load_vessels

router = APIRouter(prefix="/vessels", tags=["vessels"])


@router.get("/")
def list_vessels():
    """List all configured vessels."""
    return load_vessels()


@router.get("/{vessel_id}")
def get_vessel(vessel_id: str):
    """Return a single vessel by id."""
    vessels = load_vessels()
    for vessel in vessels:
        if vessel["vessel_id"] == vessel_id:
            return vessel
    raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")
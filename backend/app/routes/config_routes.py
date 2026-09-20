"""Port and research center endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import load_research_centers, load_ports
from app.models.schemas import PortResponse, ResearchCenterResponse

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get("/ports", response_model=list[PortResponse])
def list_ports():
    """List all configured departure ports."""
    return load_ports()


@router.get("/research-centers", response_model=list[ResearchCenterResponse])
def list_research_centers():
    """List all configured Antarctic research center destinations."""
    return load_research_centers()


@router.get("/simulation")
def get_simulation_config():
    """Return the simulation configuration for the frontend."""
    from app.config import load_simulation_config

    return load_simulation_config()
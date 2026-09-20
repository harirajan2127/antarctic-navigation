"""Route-optimization endpoint for the Antarctic navigation DSS.

``POST /api/v1/routes/optimize`` generates a recommended route plus
alternative (shortest / safest / fuel-efficient) routes on a live risk grid
built from the configured sea-ice and iceberg data sources at the requested
forecast time.

All routing output is decision support; no route is described as safe.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import load_navigation_config, load_ports, load_research_centers, load_vessels
from app.models.schemas import (
    RiskLevel,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
    RouteResult,
    RouteWaypoint,
)
from navigation.approach import resolve_endpoint
from navigation.astar import NoPathFoundError
from navigation.grid import AntarcticGrid
from navigation.route_optimizer import RouteOptimizer
from config import settings
from services.route_engine import build_debug, build_route_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/routes", tags=["navigation"])

DISCLAIMER = (
    "Decision support only. Route recommendations are not certified navigation "
    "advice; the master retains final authority. No route is guaranteed safe."
)


def _resolve_vessel(vessel_id: str) -> dict[str, Any]:
    vessels = {v["vessel_id"]: v for v in load_vessels()}
    if vessel_id not in vessels:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")
    return vessels[vessel_id]


def _utc(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _clamp_to_region(
    grid: AntarcticGrid, lat: float, lon: float
) -> tuple[float, float, str | None]:
    """Clamp an out-of-grid point to the nearest edge, returning a warning."""
    if grid.contains(lat, lon):
        return lat, lon, None
    lat_c = min(max(lat, grid.lat_min), grid.lat_max)
    lon_c = min(max(lon, grid.lon_min), grid.lon_max)
    warning = (
        f"Point ({lat:.2f}, {lon:.2f}) is outside the Antarctic grid; "
        f"snapped to ({lat_c:.2f}, {lon_c:.2f}). Decision support uses the "
        "snapped position."
    )
    return lat_c, lon_c, warning


def _route_to_result(route: dict[str, Any]) -> RouteResult:
    return RouteResult(
        preference=route["preference"],
        source=route["source"],
        waypoints=[RouteWaypoint(**wp) for wp in route["waypoints"]],
        distance_km=route["distance_km"],
        distance_nm=route["distance_nm"],
        travel_time_hours=route["travel_time_hours"],
        fuel_tons=route["fuel_tons"],
        risk_score=route["risk_score"],
        max_risk_level=RiskLevel(route["max_risk_level"]),
        warnings=route["warnings"],
        valid=route["valid"],
    )


def _build_debug(
    grid: AntarcticGrid, engine: RiskEngine, config: dict[str, Any]
) -> dict[str, Any]:
    """Debug metadata about the routing grid and hazard state."""
    return {
        "grid": {
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "resolution_degrees": grid.resolution,
            "nlat": grid.nlat,
            "nlon": grid.nlon,
        },
        "land_cells": int(grid.land_mask.sum()),
        "blocked_cells": int(engine.blocked_mask.sum()),
        "forecast_horizon_hours": int(config.get("forecast_horizon_hours", 24)),
        "risk_weights": dict(engine.config.risk_weights),
        "engine_notes": list(engine.notes),
    }


@router.post("/optimize", response_model=RouteOptimizationResponse)
def optimize_route(request: RouteOptimizationRequest) -> RouteOptimizationResponse:
    """Generate recommended and alternative routes for a vessel."""
    config = load_navigation_config()
    vessel = _resolve_vessel(request.vessel_id)
    ts = _utc(request.forecast_time)

    grid, engine, notes, _ = build_route_engine(vessel, ts, config)

    warnings: list[str] = list(notes)

    start_lat, start_lon, start_warn = _clamp_to_region(
        grid, request.start_latitude, request.start_longitude
    )
    goal_lat, goal_lon, goal_warn = _clamp_to_region(
        grid, request.destination_latitude, request.destination_longitude
    )
    warnings.extend(w for w in (start_warn, goal_warn) if w)

    max_approach = float(settings.RESEARCH_STATION_APPROACH_DISTANCE_KM or 100.0)
    research_centers = load_research_centers()
    ports = load_ports()

    try:
        start_ep = resolve_endpoint(
            grid, start_lat, start_lon, research_centers, ports, max_approach
        )
        goal_ep = resolve_endpoint(
            grid, goal_lat, goal_lon, research_centers, ports, max_approach
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if start_ep.note:
        warnings.append(start_ep.note)
    if goal_ep.note:
        warnings.append(goal_ep.note)
    req_start_lat, req_start_lon = start_lat, start_lon
    req_goal_lat, req_goal_lon = goal_lat, goal_lon
    start_lat, start_lon = start_ep.lat, start_ep.lon
    goal_lat, goal_lon = goal_ep.lat, goal_ep.lon

    optimizer = RouteOptimizer(grid, engine, vessel=vessel)
    preference = request.optimization_preference.value

    try:
        recommended = optimizer.optimize(
            start_lat, start_lon, goal_lat, goal_lon, preference=preference
        )
    except NoPathFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    alternative_routes: list[RouteResult] = []
    for alternative in optimizer.alternatives(
        start_lat, start_lon, goal_lat, goal_lon, excluded=preference
    ):
        if alternative is None:
            warnings.append(
                "An alternative route preference could not be calculated "
                "(no navigable path under that preference)."
            )
        else:
            alternative_routes.append(_route_to_result(alternative))

    recommended_result = _route_to_result(recommended)

    response_warnings = list(dict.fromkeys(warnings + recommended_result.warnings))

    return RouteOptimizationResponse(
        start_latitude=round(start_lat, 4),
        start_longitude=round(start_lon, 4),
        destination_latitude=round(goal_lat, 4),
        destination_longitude=round(goal_lon, 4),
        requested_start_latitude=round(req_start_lat, 4),
        requested_start_longitude=round(req_start_lon, 4),
        requested_destination_latitude=round(req_goal_lat, 4),
        requested_destination_longitude=round(req_goal_lon, 4),
        destination_name=goal_ep.place_name,
        ocean_approach_distance_km=(
            round(goal_ep.approach_distance_km, 1) if goal_ep.on_land else None
        ),
        vessel_id=request.vessel_id,
        optimization_preference=preference,
        forecast_time=request.forecast_time,
        recommended=recommended_result,
        alternatives=alternative_routes,
        distance_km=recommended_result.distance_km,
        distance_nm=recommended_result.distance_nm,
        travel_time_hours=recommended_result.travel_time_hours,
        fuel_tons=recommended_result.fuel_tons,
        risk_score=recommended_result.risk_score,
        risk_level=recommended_result.max_risk_level,
        warnings=response_warnings,
        disclaimer=DISCLAIMER,
        hazard_config=engine.hazard_config(),
        explanation=engine.explanation(),
        debug=build_debug(grid, engine, config),
    )
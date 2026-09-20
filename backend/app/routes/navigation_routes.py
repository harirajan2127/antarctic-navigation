"""Navigation endpoints: journey creation, advancement, and route updates."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import load_ports, load_research_centers, load_vessels
from app.models.schemas import JourneyRequest, RouteResponse, SimulationStatus
from app.services.simulation.simulator import JourneyManager
from app.utils.geo import haversine_distance_nm

router = APIRouter(prefix="/navigation", tags=["navigation"])

_journey_manager = JourneyManager()
journeys: dict[str, dict[str, Any]] = {}


def _resolve_places(request: JourneyRequest) -> tuple[dict, dict]:
    """Resolve request ids to configured places."""
    ports = {p["port_id"]: p for p in load_ports()}
    centers = {c["center_id"]: c for c in load_research_centers()}

    if request.departure_port_id not in ports:
        raise HTTPException(status_code=404, detail="Departure port not found")
    if request.destination_id not in centers:
        raise HTTPException(status_code=404, detail="Destination research center not found")

    return ports[request.departure_port_id], centers[request.destination_id]


def _resolve_vessel(vessel_id: str | None) -> dict:
    vessels = {v["vessel_id"]: v for v in load_vessels()}
    if vessel_id and vessel_id not in vessels:
        raise HTTPException(status_code=404, detail=f"Vessel {vessel_id} not found")
    return vessels.get(vessel_id, vessels.get("polar_explorer", {}))


@router.post("/journey", response_model=RouteResponse)
def start_journey(request: JourneyRequest):
    """Start a new journey (outbound or return) and compute the first route."""
    port, center = _resolve_places(request)
    vessel = _resolve_vessel(request.vessel_id)

    journey = _journey_manager.create_journey(
        departure_port=port,
        destination=center,
        journey_mode=request.journey_mode,
        vessel=vessel,
    )

    # Live mode: start from an explicitly provided position if any.
    if request.position_mode == "live":
        if request.current_latitude is None or request.current_longitude is None:
            raise HTTPException(
                status_code=400,
                detail="Live mode requires current_latitude and current_longitude",
            )
        journey = _journey_manager.recalculate(
            journey,
            current_lat=request.current_latitude,
            current_lon=request.current_longitude,
        )

    journeys[journey["journey_id"]] = journey
    return _route_response(journey)


@router.post("/journey/{journey_id}/advance", response_model=RouteResponse)
def advance_journey(journey_id: str):
    """Advance the vessel two hours along the current route and recalculate."""
    journey = journeys.get(journey_id)
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey not found")

    journey = _journey_manager.advance(journey)
    journeys[journey_id] = journey
    return _route_response(journey)


@router.post("/journey/{journey_id}/recalculate", response_model=RouteResponse)
def recalculate_journey(
    journey_id: str, lat: float, lon: float, direction: str | None = None
):
    """Recalculate the route from an arbitrary position (live GPS)."""
    journey = journeys.get(journey_id)
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey not found")

    try:
        journey = _journey_manager.recalculate(
            journey, current_lat=lat, current_lon=lon, direction=direction
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    journeys[journey_id] = journey
    return _route_response(journey)


@router.get("/journey/{journey_id}/status", response_model=SimulationStatus)
def journey_status(journey_id: str):
    """Return the current simulation status."""
    journey = journeys.get(journey_id)
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey not found")

    remaining = haversine_distance_nm(
        journey["current_lat"],
        journey["current_lon"],
        journey["goal_lat"],
        journey["goal_lon"],
    )
    total = journey["route"]["total_distance_nm"] + remaining  # rough total for progress
    progress = (
        100.0 * (1.0 - remaining / total) if total > 0 else 100.0
    )

    return SimulationStatus(
        journey_id=journey_id,
        current_time_hours=journey["current_time_hours"],
        current_lat=journey["current_lat"],
        current_lon=journey["current_lon"],
        progress_percent=round(progress, 1),
        remaining_distance_nm=round(remaining, 1),
        remaining_fuel_tons=round(remaining / 2000.0 * 1.0, 1),
        is_complete=journey["completed"],
        route_update_count=journey["route_update_count"],
    )


def _route_response(journey: dict[str, Any]) -> RouteResponse:
    route = journey["route"]
    waypoints = []
    for wp in route["waypoints"]:
        from app.models.schemas import Waypoint

        waypoints.append(
            Waypoint(
                latitude=wp["latitude"],
                longitude=wp["longitude"],
                distance_nm=wp["distance_nm"],
                heading_deg=wp.get("heading_deg") or 0.0,
                speed_knots=wp["speed_knots"],
                risk_level=wp["risk_level"],
                estimated_fuel_tons=0.0,
                timestamp_hours=0.0,
            )
        )

    return RouteResponse(
        journey_id=journey["journey_id"],
        origin=journey["origin"],
        destination=journey["destination"],
        journey_mode=journey["mode"],
        waypoints=waypoints,
        total_distance_nm=route["total_distance_nm"],
        total_fuel_tons=route["estimated_fuel_tons"],
        estimated_duration_hours=route["estimated_duration_hours"],
        max_risk_level=_max_risk(waypoints),
        route_update_count=journey["route_update_count"],
        current_vessel_lat=journey["current_lat"],
        current_vessel_lon=journey["current_lon"],
    )


def _max_risk(waypoints) -> str:
    ranking = {"low": 0, "moderate": 1, "high": 2, "extreme": 3}
    return max((wp.risk_level for wp in waypoints), key=lambda r: ranking[r])
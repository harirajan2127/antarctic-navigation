"""Route-optimisation and route-detail endpoints."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from schemas.models import (
    RouteDetailResponse,
    RouteOptimizationRequest,
    RouteOptimizationPreference,
    RouteResult,
    RoutesOptimizeResponse,
)
from services.navigation_service import navigation_service

router = APIRouter(prefix="/routes", tags=["routes"])
logger = logging.getLogger("dss.api.routes")


def _result_from_dict(d: dict) -> RouteResult:
    return RouteResult(
        waypoints=d.get("waypoints", []),
        coordinates=d.get("coordinates", []),
        distance_km=d.get("distance_km", 0.0),
        distance_nm=d.get("distance_nm", 0.0),
        travel_time_hours=d.get("travel_time_hours", 0.0),
        fuel_tons=d.get("fuel_tons"),
        risk_score=d.get("risk_score"),
        risk_level=d.get("risk_level"),
        warnings=d.get("warnings", []),
    )


@router.post("/optimize", response_model=RoutesOptimizeResponse)
async def optimize_route(payload: RouteOptimizationRequest) -> RoutesOptimizeResponse:
    """POST /api/routes/optimize"""
    started = time.perf_counter()
    logger.info(
        "ROUTE_REQUEST_RECEIVED vessel_id=%s preference=%s",
        payload.vessel_id,
        payload.preference.value,
    )
    try:
        result, classification = navigation_service.optimize(
            start_lat=payload.start_latitude,
            start_lon=payload.start_longitude,
            dest_lat=payload.destination_latitude,
            dest_lon=payload.destination_longitude,
            vessel_id=payload.vessel_id,
            preference=payload.preference.value,
            forecast_time=payload.forecast_time,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    recommended = _result_from_dict(result["recommended"])
    alternatives = [_result_from_dict(a) for a in result.get("alternatives", [])]

    response = RoutesOptimizeResponse(
        route_id=result["route_id"],
        start_latitude=result["start_latitude"],
        start_longitude=result["start_longitude"],
        destination_latitude=result["destination_latitude"],
        destination_longitude=result["destination_longitude"],
        requested_start_latitude=result.get("requested_start_latitude"),
        requested_start_longitude=result.get("requested_start_longitude"),
        requested_destination_latitude=result.get("requested_destination_latitude"),
        requested_destination_longitude=result.get("requested_destination_longitude"),
        destination_name=result.get("destination_name"),
        ocean_approach_distance_km=result.get("ocean_approach_distance_km"),
        vessel_id=result["vessel_id"],
        preference=result["preference"],
        recommended=recommended,
        alternatives=alternatives,
        demo=result["demo"],
        disclaimer=result["disclaimer"],
    )
    recommended_points = len(response.recommended.coordinates)
    logger.info(
        "ROUTE_RESPONSE_SENT route_id=%s points=%d distance_km=%.2f elapsed_ms=%.1f",
        response.route_id,
        recommended_points,
        response.recommended.distance_km,
        (time.perf_counter() - started) * 1000,
    )
    logger.info("ROUTE_POINTS_COUNT count=%d", recommended_points)
    logger.info("ROUTE_DISTANCE distance_km=%.2f", response.recommended.distance_km)
    return response


@router.get("/{route_id}", response_model=RouteDetailResponse)
async def get_route(route_id: str) -> RouteDetailResponse:
    """GET /api/routes/{route_id}"""
    result = navigation_service.get_route(route_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Route '{route_id}' not found.")
    recommended = _result_from_dict(result)
    alternatives = [_result_from_dict(a) for a in result.get("alternatives", [])]
    return RouteDetailResponse(
        route_id=result["route_id"],
        created_at=result["created_at"],
        start_latitude=result["start_latitude"],
        start_longitude=result["start_longitude"],
        destination_latitude=result["destination_latitude"],
        destination_longitude=result["destination_longitude"],
        vessel_id=result["vessel_id"],
        preference=result["preference"],
        recommended=recommended,
        alternatives=alternatives,
        demo=result["demo"],
        disclaimer=result["disclaimer"],
        waypoints_total=result.get("waypoints_total", 0),
    )
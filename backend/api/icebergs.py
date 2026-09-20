"""Iceberg data, prediction, trajectory and distance endpoints."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query

from schemas.models import (
    IcebergDetailResponse,
    IcebergDistanceResponse,
    IcebergPrediction,
    IcebergPredictRequest,
    IcebergPredictionsResponse,
    IcebergTrajectoryResponse,
    IcebergsListResponse,
    TrajectoryPoint,
)
from services.iceberg_service import iceberg_service

router = APIRouter(prefix="/icebergs", tags=["icebergs"])
logger = logging.getLogger("dss.api.icebergs")


def _to_traj_point(p: dict) -> TrajectoryPoint:
    return TrajectoryPoint(
        timestamp=p.get("timestamp"),
        step=p["step"],
        latitude=p["latitude"],
        longitude=p["longitude"],
        horizon_hours=p.get("horizon_hours"),
    )


def _404(msg: str) -> HTTPException:
    raise HTTPException(status_code=404, detail=msg)


@router.get("", response_model=IcebergsListResponse)
async def list_icebergs() -> IcebergsListResponse:
    """GET /api/icebergs"""
    data = iceberg_service.list_icebergs()
    return IcebergsListResponse(**data)


@router.get("/distance", response_model=IcebergDistanceResponse)
async def iceberg_distance(
    iceberg_a: str = Query(..., description="First iceberg ID"),
    iceberg_b: str = Query(..., description="Second iceberg ID"),
) -> IcebergDistanceResponse:
    """GET /api/icebergs/distance?iceberg_a=DEMO-B000&iceberg_b=DEMO-B001"""
    result = iceberg_service.distance(iceberg_a, iceberg_b)
    if result is None:
        _404(f"Iceberg '{iceberg_a}' or '{iceberg_b}' not found.")
    return IcebergDistanceResponse(**result)


@router.post("/predict", response_model=IcebergPredictionsResponse)
async def iceberg_predict(payload: IcebergPredictRequest) -> IcebergPredictionsResponse:
    """POST /api/icebergs/predict"""
    data = iceberg_service.predict(
        iceberg_ids=payload.iceberg_ids,
        horizon_hours=payload.horizon_hours,
        model=payload.model,
    )
    return IcebergPredictionsResponse(**data)


@router.get("/models", tags=["icebergs"])
async def iceberg_models() -> dict:
    """GET /api/icebergs/models — trained-model availability for the UI."""
    return iceberg_service.model_status()


@router.get("/{iceberg_id}/trajectory", response_model=IcebergTrajectoryResponse)
async def iceberg_trajectory(
    iceberg_id: str,
    model: str = Query("persistence", description="persistence | random_forest | lstm"),
) -> IcebergTrajectoryResponse:
    """GET /api/icebergs/{iceberg_id}/trajectory"""
    started = time.perf_counter()
    logger.info("ICEBERG_PREDICTION_STARTED iceberg_id=%s model=%s", iceberg_id, model)
    data = iceberg_service.trajectory(iceberg_id, model=model)
    if data is None:
        _404(f"Iceberg '{iceberg_id}' not found.")
    response = IcebergTrajectoryResponse(
        iceberg_id=data["iceberg_id"],
        observations=[_to_traj_point(p) for p in data.get("observations", [])],
        predictions=[_to_traj_point(p) for p in data.get("predictions", [])],
        count_observations=data.get("count_observations", 0),
        count_predictions=data.get("count_predictions", 0),
        classification=data.get("classification", "unknown"),
        demo=data.get("demo", True),
        prediction_model=data.get("prediction_model", "persistence"),
        warning=data.get("warning"),
    )
    logger.info(
        "ICEBERG_PREDICTION_COMPLETED iceberg_id=%s observations=%d predictions=%d elapsed_ms=%.1f",
        iceberg_id,
        response.count_observations,
        response.count_predictions,
        (time.perf_counter() - started) * 1000,
    )
    return response


@router.get("/{iceberg_id}", response_model=IcebergDetailResponse)
async def iceberg_detail(iceberg_id: str) -> IcebergDetailResponse:
    """GET /api/icebergs/{iceberg_id}"""
    item = iceberg_service.detail(iceberg_id)
    if item is None:
        _404(f"Iceberg '{iceberg_id}' not found.")
    return IcebergDetailResponse(**item)
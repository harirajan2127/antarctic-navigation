"""Analytics endpoints (summary + model metrics)."""
from __future__ import annotations

from fastapi import APIRouter

from schemas.models import (
    AnalyticsSummaryResponse,
    DatasetInfo,
    ModelAccuracyEntry,
    ModelMetricOut,
    ModelMetricsResponse,
    ModelStatusEntry,
)
from services.analytics_service import model_metrics, summary
from services.dataset_service import from_database

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
async def analytics_summary() -> AnalyticsSummaryResponse:
    """GET /api/analytics/summary"""
    datasets = [DatasetInfo(**row) for row in from_database()]
    data = summary(classifications=datasets)
    return AnalyticsSummaryResponse(
        demo_mode=data["demo_mode"],
        real_model_available=data["real_model_available"],
        model_accuracy={
            k: ModelAccuracyEntry(**v) for k, v in data.get("model_accuracy", {}).items()
        },
        model_status={
            k: ModelStatusEntry(**v) for k, v in data.get("model_status", {}).items()
        },
        evaluation_checks=data.get("evaluation_checks", {}),
        accuracy_summary=data.get("accuracy_summary", {}),
        real_data_available=data["real_data_available"],
        datasets=data["datasets"],
        routes_stored=data["routes_stored"],
        icebergs_tracked=data["icebergs_tracked"],
        sea_ice_forecasts_stored=data["sea_ice_forecasts_stored"],
        metrics_available=data["metrics_available"],
        model_metrics_count=data["model_metrics_count"],
        warnings=data["warnings"],
    )


@router.get("/model-metrics", response_model=ModelMetricsResponse)
async def analytics_model_metrics() -> ModelMetricsResponse:
    """GET /api/analytics/model-metrics"""
    data = model_metrics()
    return ModelMetricsResponse(
        metrics=[ModelMetricOut(**m) for m in data["metrics"]],
        count=data["count"],
        demo_mode=data["demo_mode"],
        warning=data["warning"],
    )
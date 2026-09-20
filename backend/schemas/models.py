"""Pydantic request/response schemas for the DSS REST API.

Kept in ``backend/schemas/`` so the public API surface is independent of the
legacy ``app.models.schemas`` used by the older ``/api/v1`` application.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SYNTHETIC_DEMO = "synthetic_demo"


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    demo_mode: bool = False
    real_data_mode: bool = True
    database: str
    timestamp: datetime
    sea_ice_data: bool = False
    ocean_data: bool = False
    weather_data: bool = False
    iceberg_data: bool = False
    land_mask: bool = False
    iceberg_model: bool = False
    route_engine: bool = False


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class DatasetInfo(BaseModel):
    dataset: str = Field(description="Dataset key, e.g. sea_ice, iceberg")
    description: str | None = None
    classification: str = Field(
        description="synthetic_demo (fake) / pipeline_data (processed real) / unclassified"
    )
    demo: bool = Field(description="True when the dataset is synthetic demo data")
    status: str = Field(description="ok / missing / demo")
    file_path: str | None = None
    records: int | None = None
    source_id: str | None = None
    source_name: str | None = None
    notes: str | None = None
    updated_at: datetime | None = None


class DatasetsResponse(BaseModel):
    datasets: list[DatasetInfo]
    demo_mode: bool
    warning: str | None = None


class DatasetStatusResponse(BaseModel):
    datasets: list[DatasetInfo]
    real_data_available: bool = Field(
        description="True when at least one registered dataset is not synthetic"
    )
    demo_mode: bool
    warning: str | None = None


# --------------------------------------------------------------------------
# Sea ice
# --------------------------------------------------------------------------
class SeaIceCurrentResponse(BaseModel):
    timestamp: datetime
    lat: list[float]
    lon: list[float]
    concentration: list[list[float | None]] = Field(
        description="2-D field, shape (len(lat), len(lon)); None marks cells with no observed data"
    )
    classification: str
    demo: bool
    source: str | None = None
    warning: str | None = None


class SeaIceForecastResponse(BaseModel):
    forecast_time: datetime
    valid_time: datetime | None = None
    horizon_hours: int
    model: str
    lat: list[float]
    lon: list[float]
    concentration: list[list[float | None]] = Field(
        description="2-D field, shape (len(lat), len(lon)); None marks cells with no observed data"
    )
    mean_concentration: float
    max_concentration: float
    coverage_pct: float = Field(description="% of valid grid cells in the field")
    classification: str
    demo: bool
    model_used_real: bool = Field(
        description="True only when the model was trained on real observations"
    )
    skill_note: str | None = None
    warning: str | None = None


class SeaIcePredictRequest(BaseModel):
    """Optional custom input for sea-ice forecasting.

    When ``concentration`` / ``lat`` / ``lon`` are omitted the service uses
    the currently loaded sea-ice field.
    """

    concentration: list[list[float]] | None = None
    lat: list[float] | None = None
    lon: list[float] | None = None
    horizon_hours: int = Field(default=24, ge=2, le=168)
    model: str = Field(default="persistence", description="persistence | random_forest | convlstm")
    timestamp: datetime | None = None


# --------------------------------------------------------------------------
# Icebergs
# --------------------------------------------------------------------------
class IcebergInfo(BaseModel):
    """An iceberg and its latest-known position.

    ``on_land`` and ``distance_to_coast_km`` are additive dashboard-only
    annotations provided by the data loader.  They let the map draw only the
    icebergs that are at sea and nearest to the Antarctic coast; everything
    else (positions, counts, predictions, analytics) is unchanged.
    """

    iceberg_id: str
    latitude: float
    longitude: float
    length_km: float | None = None
    width_km: float | None = None
    last_observed: datetime | None = None
    source: str | None = None
    demo: bool = True
    on_land: bool | None = None
    distance_to_coast_km: float | None = None


class IcebergsListResponse(BaseModel):
    icebergs: list[IcebergInfo]
    count: int
    classification: str
    demo: bool
    warning: str | None = None


class IcebergDetailResponse(IcebergInfo):
    observation_count: int = 0
    classification: str
    warning: str | None = None


class IcebergPredictRequest(BaseModel):
    iceberg_ids: list[str] | None = Field(
        default=None, description="Empty/None predicts for all tracked icebergs"
    )
    horizon_hours: int = Field(default=24, ge=2, le=72)
    model: str = Field(default="persistence", description="persistence | random_forest | lstm")


class IcebergPrediction(BaseModel):
    iceberg_id: str
    current_lat: float
    current_lon: float
    predicted_lat: float
    predicted_lon: float
    prediction_hours: int
    confidence: float | str | None = Field(
        default=None,
        description="Numeric confidence when derived from real model validation; "
        "or an explanatory text note from baselines that model no uncertainty.",
    )
    demo: bool = True


class IcebergPredictionsResponse(BaseModel):
    icebergs: list[IcebergPrediction]
    model: str
    horizon_hours: int
    classification: str
    demo: bool
    trained_used: bool = Field(
        default=False, description="True when at least one prediction ran on a trained model"
    )
    warning: str | None = None


class TrajectoryPoint(BaseModel):
    timestamp: datetime | None = None
    step: str = Field(description="observation | prediction")
    latitude: float
    longitude: float
    horizon_hours: int | None = None


class IcebergTrajectoryResponse(BaseModel):
    iceberg_id: str
    observations: list[TrajectoryPoint]
    predictions: list[TrajectoryPoint]
    count_observations: int
    count_predictions: int
    classification: str
    demo: bool
    prediction_model: str = Field(
        default="persistence", description="model actually used for the predictions"
    )
    warning: str | None = None


class IcebergDistanceResponse(BaseModel):
    iceberg_a: str
    iceberg_b: str
    distance_km: float
    distance_nm: float
    classification: str
    demo: bool
    warning: str | None = None


# --------------------------------------------------------------------------
# Routes / navigation
# --------------------------------------------------------------------------
class RouteOptimizationPreference(str, Enum):
    recommended = "recommended"
    shortest = "shortest"
    safest = "safest"
    fuel_efficient = "fuel_efficient"


class RouteOptimizationRequest(BaseModel):
    start_latitude: float = Field(ge=-90.0, le=0.0)
    start_longitude: float = Field(ge=-180.0, le=180.0)
    destination_latitude: float = Field(ge=-90.0, le=0.0)
    destination_longitude: float = Field(ge=-180.0, le=180.0)
    vessel_id: str = "polar_explorer"
    preference: RouteOptimizationPreference = RouteOptimizationPreference.recommended
    forecast_time: datetime | None = None


class RouteResult(BaseModel):
    waypoints: list[dict[str, Any]] = Field(
        description="Ordered waypoints [{lat, lon, risk_score, step, ...}]"
    )
    coordinates: list[list[float]] = Field(
        description="[[lat, lon], ...] polyline for mapping"
    )
    distance_km: float
    distance_nm: float
    travel_time_hours: float
    fuel_tons: float | None = None
    risk_score: float | None = None
    risk_level: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RoutesOptimizeResponse(BaseModel):
    route_id: str
    start_latitude: float
    start_longitude: float
    destination_latitude: float
    destination_longitude: float
    # Original requested coordinates kept as metadata (a land station keeps its
    # own coordinates; only the routed endpoint is snapped to an ocean cell).
    requested_start_latitude: float | None = None
    requested_start_longitude: float | None = None
    requested_destination_latitude: float | None = None
    requested_destination_longitude: float | None = None
    destination_name: str | None = None
    ocean_approach_distance_km: float | None = None
    vessel_id: str
    preference: RouteOptimizationPreference
    recommended: RouteResult
    alternatives: list[RouteResult] = Field(default_factory=list)
    demo: bool
    disclaimer: str | None = None


class RouteDetailResponse(RoutesOptimizeResponse):
    created_at: datetime
    waypoints_total: int


# --------------------------------------------------------------------------
# Vessels
# --------------------------------------------------------------------------
class VesselInfo(BaseModel):
    vessel_id: str
    name: str
    vessel_type: str = ""
    ice_class: str | None = None
    cruise_speed_knots: float | None = None
    max_speed_knots: float | None = None
    fuel_rate_lph: float | None = None
    fuel_capacity_tons: float | None = None
    safety_distance_nm: float | None = None


class VesselsResponse(BaseModel):
    vessels: list[VesselInfo]
    count: int


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
class ModelStatusEntry(BaseModel):
    model_name: str = Field(description="Canonical pipeline id (sea_ice / iceberg)")
    display_name: str = Field(description="Visible dashboard label (Sea-Ice / Iceberg)")
    deployed_model: str = Field(description="Production algorithm (e.g. persistence)")
    status: str = Field(
        description="One of: trained_on_real_data, evaluated_on_real_data, "
        "model_available_source_unknown, demo_only, not_available"
    )
    training_data_source: str | None = Field(
        default=None, description="real_processed_data / demo_synthetic / None"
    )
    metrics_available: bool = False
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Real metric rows from model_metrics"
    )
    trained_artifacts: list[dict] = Field(
        default_factory=list, description="On-disk checkpoints (name + trained_on_demo)"
    )


class ModelAccuracyEntry(BaseModel):
    available: bool
    status: str | None = None
    model: str | None = None
    reason: str | None = None
    n_test_pairs: int | None = None
    target: str | None = None
    unit: str | None = None
    horizon_hours: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    vs_zero_baseline: bool | None = None
    mean_km: float | None = None
    median_km: float | None = None
    within_10km_pct: float | None = None


class AnalyticsSummaryResponse(BaseModel):
    demo_mode: bool
    real_model_available: dict[str, bool] = Field(
        description="Per-pipeline True only when non-demo metrics were found"
    )
    model_accuracy: dict[str, ModelAccuracyEntry] = Field(
        default_factory=dict,
        description="Real-data accuracy from reports (only when VERIFIED)",
    )
    model_status: dict[str, ModelStatusEntry] = Field(
        default_factory=dict,
        description="Per-pipeline derived model status",
    )
    evaluation_checks: dict[str, dict] = Field(
        default_factory=dict,
        description="Real navigation checks; unavailable checks are not scored",
    )
    accuracy_summary: dict = Field(
        default_factory=dict,
        description="Validated complete or provisional weighted accuracy summary",
    )
    real_data_available: bool = Field(
        description="True when at least one dataset is real (non-demo)"
    )
    datasets: list[DatasetInfo]
    routes_stored: int
    icebergs_tracked: int
    sea_ice_forecasts_stored: int
    metrics_available: bool
    model_metrics_count: int
    warnings: list[str] = Field(default_factory=list)


class ModelMetricOut(BaseModel):
    pipeline: str
    model: str
    metric_name: str
    metric_value: float
    split: str | None = None
    demo: bool
    trained_on: datetime | None = None
    source_file: str | None = None


class ModelMetricsResponse(BaseModel):
    metrics: list[ModelMetricOut]
    count: int
    demo_mode: bool
    warning: str | None = None


# --------------------------------------------------------------------------
# Model registry / status
# --------------------------------------------------------------------------
class TrainedModelInfo(BaseModel):
    pipeline: str
    name: str
    available: bool
    trained_on_demo: bool | None = None
    created_utc: str | None = None
    artifact: str | None = None
    error: str | None = None


class ModelRegistryResponse(BaseModel):
    models: list[TrainedModelInfo]
    count: int
    any_real_trained: bool
    config: dict = Field(default_factory=dict)
    demo_mode: bool
    warning: str | None = None


# --------------------------------------------------------------------------
# AI Assistant
# --------------------------------------------------------------------------
class AssistantMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class AssistantChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[AssistantMessage] = Field(default_factory=list, max_length=30)
    horizon_hours: int = Field(default=24, ge=2, le=168)


class AssistantContextSource(BaseModel):
    name: str
    demo: bool
    note: str | None = None
    status: str = "ok"


class AssistantChatResponse(BaseModel):
    answer: str
    sources: list[AssistantContextSource]
    llm: dict[str, object] = Field(
        description="provider/model + configured flag; NEVER contains API keys"
    )
    demo_mode: bool
    warnings: list[str] = Field(default_factory=list)
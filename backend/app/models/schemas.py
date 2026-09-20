"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JourneyMode(str, Enum):
    OUTBOUND = "outbound"
    RETURN = "return"


class VesselPositionMode(str, Enum):
    SIMULATION = "simulation"
    LIVE = "live"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class PortResponse(BaseModel):
    port_id: str
    name: str
    country: str
    latitude: float
    longitude: float
    description: str
    facilities: list[str] = []
    timezone: str


class ResearchCenterResponse(BaseModel):
    center_id: str
    name: str
    latitude: float
    longitude: float
    country: str
    description: str
    sector: str = ""
    active: bool = True


class VesselResponse(BaseModel):
    vessel_id: str
    name: str
    vessel_type: str = Field(alias="type", default="")
    ice_class: str
    max_speed_knots: float
    cruise_speed_knots: float
    fuel_rate_lph: float = 0.0
    safety_distance_nm: float = 10.0
    fuel_capacity_tons: float
    fuel_consumption_tons_per_day: dict[str, float]
    range_nautical_miles: dict[str, float]


class JourneyRequest(BaseModel):
    departure_port_id: str
    destination_id: str
    vessel_id: str = "polar_explorer"
    journey_mode: JourneyMode = JourneyMode.OUTBOUND
    position_mode: VesselPositionMode = VesselPositionMode.SIMULATION
    current_latitude: float | None = None
    current_longitude: float | None = None


class Waypoint(BaseModel):
    latitude: float
    longitude: float
    distance_nm: float = 0.0
    heading_deg: float = 0.0
    speed_knots: float = 12.0
    sea_ice_concentration: float = 0.0
    iceberg_risk: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_fuel_tons: float = 0.0
    timestamp_hours: float = 0.0


class RouteResponse(BaseModel):
    journey_id: str
    origin: str
    destination: str
    journey_mode: str
    waypoints: list[Waypoint]
    total_distance_nm: float
    total_fuel_tons: float
    estimated_duration_hours: float
    max_risk_level: RiskLevel
    route_update_count: int = 1
    current_vessel_lat: float
    current_vessel_lon: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SeaIceForecast(BaseModel):
    timestamp: datetime
    grid_lat: list[float]
    grid_lon: list[float]
    concentration: list[list[float]]
    forecast_hours: int
    model_used: str


class IcebergPosition(BaseModel):
    iceberg_id: str
    latitude: float
    longitude: float
    size_km2: float
    predicted_lat: float
    predicted_lon: float
    prediction_hours: int
    confidence: float


class NavigationRiskGrid(BaseModel):
    grid_lat: list[float]
    grid_lon: list[float]
    risk_values: list[list[float]]
    sea_ice_contribution: list[list[float]]
    iceberg_contribution: list[list[float]]
    weather_contribution: list[list[float]]


class SimulationStatus(BaseModel):
    journey_id: str
    current_time_hours: float
    current_lat: float
    current_lon: float
    progress_percent: float
    remaining_distance_nm: float
    remaining_fuel_tons: float
    is_complete: bool
    route_update_count: int


class RecalculationRequest(BaseModel):
    journey_id: str
    current_latitude: float
    current_longitude: float
    current_time_hours: float


class RouteOptimizationPreference(str, Enum):
    RECOMMENDED = "recommended"
    SHORTEST = "shortest"
    SAFEST = "safest"
    FUEL_EFFICIENT = "fuel_efficient"


class RouteOptimizationRequest(BaseModel):
    start_latitude: float
    start_longitude: float
    destination_latitude: float
    destination_longitude: float
    forecast_time: datetime | None = None
    vessel_id: str = "polar_explorer"
    optimization_preference: RouteOptimizationPreference = (
        RouteOptimizationPreference.RECOMMENDED
    )


class RouteWaypoint(BaseModel):
    index: int
    latitude: float
    longitude: float
    distance_km: float = 0.0
    distance_nm: float = 0.0
    cumulative_distance_km: float = 0.0
    heading_deg: float = 0.0
    sea_ice_concentration: float | None = None
    iceberg_distance_km: float | None = None
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    speed_knots: float = 12.0
    elapsed_hours: float = 0.0
    fuel_tons: float = 0.0


class RouteResult(BaseModel):
    preference: str
    source: str
    waypoints: list[RouteWaypoint]
    distance_km: float
    distance_nm: float
    travel_time_hours: float
    fuel_tons: float
    risk_score: float
    max_risk_level: RiskLevel
    warnings: list[str] = []
    valid: bool = True


class RouteOptimizationResponse(BaseModel):
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
    optimization_preference: str
    forecast_time: datetime | None
    recommended: RouteResult
    alternatives: list[RouteResult]
    distance_km: float
    distance_nm: float
    travel_time_hours: float
    fuel_tons: float
    risk_score: float
    risk_level: RiskLevel
    warnings: list[str] = []
    disclaimer: str
    # Decision-support explanation + hazard configuration + debug metadata.
    hazard_config: dict[str, Any] | None = None
    explanation: list[str] = []
    debug: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class DashboardData(BaseModel):
    ports: list[PortResponse]
    research_centers: list[ResearchCenterResponse]
    vessels: list[VesselResponse]
    simulation_config: dict[str, Any]

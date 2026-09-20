"""Antarctic navigation decision-support engine.

Generates safe, distance-efficient and fuel-efficient candidate routes using
predicted sea ice, iceberg positions, ocean conditions, and vessel
information.

The output is decision support only -- it is not certified navigation advice
and no route is guaranteed safe.
"""
from navigation.astar import AStarPlanner, NoPathFoundError
from navigation.fuel_estimator import FuelEstimator, FuelEstimate
from navigation.geodesy import (
    haversine_km,
    haversine_nm,
    initial_bearing_deg,
    normalize_longitude,
)
from navigation.grid import AntarcticGrid, GridCell, GridProjection
from navigation.risk_engine import RiskConfig, RiskEngine, sea_ice_risk
from navigation.route_optimizer import RouteOptimizer, RouteValidator

__version__ = "0.1.0"

__all__ = [
    "AStarPlanner",
    "NoPathFoundError",
    "AntarcticGrid",
    "GridCell",
    "GridProjection",
    "RiskConfig",
    "RiskEngine",
    "sea_ice_risk",
    "RouteOptimizer",
    "RouteValidator",
    "FuelEstimator",
    "FuelEstimate",
    "haversine_km",
    "haversine_nm",
    "initial_bearing_deg",
    "normalize_longitude",
]
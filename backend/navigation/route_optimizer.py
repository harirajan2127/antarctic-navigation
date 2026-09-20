"""Route optimization: shortest, lowest-risk, fuel-efficient, and recommended routes.

All objectives are produced by a single weighted A* cost function:
    step_cost = distance_km * (distance_weight + safety_weight * risk)
which keeps the cost in distance units so routes are directly comparable. A
recommended route is the weighted compromise; alternatives rerun the search
with other weight combinations and are validated before being returned.
"""
from __future__ import annotations

from typing import Any

from navigation import geodesy
from navigation.astar import AStarPlanner, NoPathFoundError
from navigation.fuel_estimator import FuelEstimator
from navigation.grid import AntarcticGrid
from navigation.risk_engine import RiskEngine

_PREFERENCE_WEIGHTS: dict[str, tuple[float, float]] = {
    "recommended": None,  # resolved from RiskConfig.objective_weights
    "shortest": (0.0, 1.0),
    "safest": (1.0, 0.0),
    "fuel_efficient": (0.3, 0.7),
    "alternative": (0.85, 0.15),
}

_PREFERENCE_ORDER = ("recommended", "shortest", "safest", "fuel_efficient")


class RouteValidator:
    """Validates that a route is geometrically sound and respects constraints.

    A route is *structural*ly valid when every waypoint is inside the grid and
    navigable and no segment crosses a blocked cell. Clearance and ice-limit
    violations are reported as warnings (decision support, not hard errors).
    """

    def __init__(self, grid: AntarcticGrid, risk_engine: RiskEngine) -> None:
        self.grid = grid
        self.risk_engine = risk_engine

    def validate(self, coordinates: list[tuple[float, float]]) -> tuple[bool, list[str]]:
        """Return ``(valid, warnings)`` for a list of (lat, lon) waypoints."""
        cfg = self.risk_engine.config
        warnings: list[str] = []
        valid = True

        if len(coordinates) < 2:
            return False, ["Route has fewer than two waypoints."]

        for idx, (lat, lon) in enumerate(coordinates):
            if not self.grid.contains(lat, lon):
                warnings.append(f"Waypoint {idx} ({lat:.2f}, {lon:.2f}) is outside the grid.")
                valid = False
                continue
            cell = self.grid.cell_for(lat, lon)
            if not self.grid.is_navigable(cell):
                warnings.append(f"Waypoint {idx} ({lat:.2f}, {lon:.2f}) lies on land.")
                valid = False

        self.risk_engine._ensure_computed()
        blocked = self.risk_engine.obstacle_mask()
        for idx, (lat, lon) in enumerate(coordinates):
            cell = self.grid.cell_for(lat, lon)
            if blocked[cell.lat_index, cell.lon_index]:
                warnings.append(f"Waypoint {idx} ({lat:.2f}, {lon:.2f}) is blocked.")
                valid = False

        warnings.extend(self.risk_engine.waypoint_warnings(coordinates))
        return valid, list(dict.fromkeys(warnings))


class RouteOptimizer:
    """Generates candidate routes between two points on the risk grid."""

    def __init__(
        self,
        grid: AntarcticGrid,
        risk_engine: RiskEngine,
        vessel: dict[str, Any] | None = None,
    ) -> None:
        self.grid = grid
        self.risk_engine = risk_engine
        self.vessel = vessel or {}
        self.speed_knots = float(self.vessel.get("cruise_speed_knots") or 12.0)
        self.validator = RouteValidator(grid, risk_engine)
        self.risk_engine.apply_vessel_constraints()
        self.risk_engine.compute_total()

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def _weights_for(self, preference: str) -> tuple[float, float]:
        if preference not in _PREFERENCE_WEIGHTS:
            raise ValueError(
                f"Unknown preference {preference!r}. "
                f"Choose from {sorted(_PREFERENCE_ORDER)}."
            )
        if preference == "recommended":
            obj = self.risk_engine.config.objective_weights
            safety = float(obj.get("safety", 0.6))
            distance = float(obj.get("distance", 0.4))
            total = safety + distance or 1.0
            return safety / total, distance / total
        return _PREFERENCE_WEIGHTS[preference]

    def _plan(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        safety_weight: float,
        distance_weight: float,
    ) -> list[tuple[float, float]]:
        obstacles = self.risk_engine.obstacle_mask()
        planner = AStarPlanner(
            lats=self.grid.lats,
            lons=self.grid.lons,
            cost=self.risk_engine.total_risk,
            obstacles=obstacles,
            safety_weight=safety_weight,
            distance_weight=distance_weight,
            max_cost=self.risk_engine.config.max_risk_ratio,
        )
        cells = planner.plan(start_lat, start_lon, goal_lat, goal_lon)
        return planner.coordinates_from_cells(cells)

    # ------------------------------------------------------------------ #
    # Route construction
    # ------------------------------------------------------------------ #
    def _build_route(
        self,
        coordinates: list[tuple[float, float]],
        preference: str,
        source: str,
    ) -> dict[str, Any]:
        risk_engine = self.risk_engine
        speed_knots = self.speed_knots
        rate_tons_per_day, _rate_source = FuelEstimator.rate_tons_per_day(
            self.vessel, speed_knots
        )

        waypoints: list[dict[str, Any]] = []
        prev: tuple[float, float] | None = None
        cumulative_km = 0.0
        cumulative_hours = 0.0
        cumulative_fuel = 0.0

        risk_weighted_total = 0.0
        risk_weight_total = 0.0

        for idx, (lat, lon) in enumerate(coordinates):
            sample = risk_engine.sample_at(lat, lon)
            level = risk_engine.risk_level(sample["risk_score"])

            waypoint: dict[str, Any] = {
                "index": idx,
                "latitude": round(float(lat), 4),
                "longitude": round(float(lon), 4),
                "distance_km": 0.0,
                "distance_nm": 0.0,
                "cumulative_distance_km": round(cumulative_km, 3),
                "heading_deg": 0.0,
                "sea_ice_concentration": (
                    round(sample["sea_ice_concentration"], 4)
                    if sample["sea_ice_concentration"] == sample["sea_ice_concentration"]
                    else None
                ),
                "iceberg_distance_km": (
                    round(sample["iceberg_distance_km"], 2)
                    if sample["iceberg_distance_km"] != float("inf")
                    else None
                ),
                "risk_score": round(float(sample["risk_score"]), 4),
                "risk_level": level,
                "speed_knots": speed_knots,
                "elapsed_hours": round(cumulative_hours, 3),
                "fuel_tons": round(cumulative_fuel, 3),
            }

            if prev is not None:
                prev_lat, prev_lon = prev
                seg_km = geodesy.haversine_km(prev_lat, prev_lon, lat, lon)
                seg_nm = geodesy.haversine_nm(prev_lat, prev_lon, lat, lon)
                seg_hours = seg_nm / speed_knots
                seg_fuel = rate_tons_per_day * seg_hours / 24.0

                waypoint["distance_km"] = round(seg_km, 3)
                waypoint["distance_nm"] = round(seg_nm, 3)
                waypoint["heading_deg"] = round(
                    geodesy.initial_bearing_deg(prev_lat, prev_lon, lat, lon), 1
                )

                cumulative_km += seg_km
                cumulative_hours += seg_hours
                cumulative_fuel += seg_fuel
                waypoint["cumulative_distance_km"] = round(cumulative_km, 3)
                waypoint["elapsed_hours"] = round(cumulative_hours, 3)
                waypoint["fuel_tons"] = round(cumulative_fuel, 3)

                segment_risk = max(
                    sample["risk_score"],
                    risk_engine.sample_at(prev_lat, prev_lon)["risk_score"],
                )
                risk_weighted_total += seg_nm * segment_risk
                risk_weight_total += seg_nm

            waypoints.append(waypoint)
            prev = (lat, lon)

        total_distance_km = cumulative_km
        total_distance_nm = geodesy.KM_TO_NM * cumulative_km
        fuel_estimate = FuelEstimator.fuel_for_distance(
            total_distance_km, speed_knots, self.vessel
        )
        risk_score = (
            risk_weighted_total / risk_weight_total
            if risk_weight_total > 0
            else float(risk_engine.sample_at(coordinates[-1][0], coordinates[-1][1])["risk_score"])
        )

        warnings: list[str] = []
        warnings.extend(risk_engine.waypoint_warnings(coordinates))
        valid, validation_warnings = self.validator.validate(coordinates)
        warnings.extend(validation_warnings)
        warnings = list(dict.fromkeys(warnings))

        return {
            "preference": preference,
            "source": source,
            "coordinates": [(round(la, 4), round(lo, 4)) for la, lo in coordinates],
            "waypoints": waypoints,
            "distance_km": round(total_distance_km, 2),
            "distance_nm": round(total_distance_nm, 2),
            "travel_time_hours": round(fuel_estimate.travel_time_hours, 2),
            "fuel_tons": round(fuel_estimate.fuel_tons, 2),
            "risk_score": round(float(risk_score), 4),
            "max_risk_level": max(
                (wp["risk_level"] for wp in waypoints),
                key=lambda v: ("low", "moderate", "high", "extreme").index(v),
            ),
            "warnings": warnings,
            "valid": valid,
        }

    def _source_label(self, safety_weight: float, distance_weight: float) -> str:
        return (
            f"A* weighted cost (safety={safety_weight:.2f}, "
            f"distance={distance_weight:.2f})"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        preference: str = "recommended",
    ) -> dict[str, Any]:
        """Generate a route using the weight combination for ``preference``."""
        safety, distance = self._weights_for(preference)
        coordinates = self._plan(start_lat, start_lon, goal_lat, goal_lon, safety, distance)
        return self._build_route(coordinates, preference, self._source_label(safety, distance))

    def optimize_preference_list(self) -> list[str]:
        """The available preference keys (excluding internal variants)."""
        return list(_PREFERENCE_ORDER)

    def alternatives(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        excluded: str | None = None,
    ) -> list[dict[str, Any] | None]:
        """Generate the other preference routes as alternatives.

        A ``None`` entry indicates that no path exists for that preference
        (e.g. the safeness-optimal route is fully blocked); the caller is
        expected to surface this as a warning.
        """
        results: list[dict[str, Any] | None] = []
        for preference in _PREFERENCE_ORDER:
            if preference == excluded:
                continue
            try:
                results.append(
                    self.optimize(start_lat, start_lon, goal_lat, goal_lon, preference)
                )
            except NoPathFoundError:
                results.append(None)
        return results
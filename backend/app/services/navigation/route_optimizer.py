"""Route optimization service combining A* pathfinding with vessel/fuel models."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.navigation.astar import AStarPlanner, GridCell
from app.services.navigation.risk_grid import NavigationRiskGrid
from app.utils.geo import haversine_distance_nm, initial_bearing, intermediate_point


class RouteOptimizer:
    """Builds safe, fuel-efficient routes between two points on the risk grid.

    Produces:
        - primary safe route
        - alternative (slightly longer, lower-risk) route
    """

    def __init__(
        self,
        risk_grid: NavigationRiskGrid,
        cruise_speed_knots: float = 12.0,
        fuel_consumption_tons_per_day: float = 28.0,
    ) -> None:
        self.risk_grid = risk_grid
        self.cruise_speed_knots = cruise_speed_knots
        self.fuel_consumption_tons_per_day = fuel_consumption_tons_per_day

        self.planner = AStarPlanner(
            lats=self.risk_grid.lats,
            lons=self.risk_grid.lons,
            risk=self.risk_grid.total_risk,
            safety_weight=0.6,
            distance_weight=0.4,
            max_risk=0.95,
        )
        self._route_cache: dict[tuple[float, float, float, float], list[tuple[float, float]]] = {}

    def optimize(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> dict[str, Any]:
        """Return a route dictionary with geometry and summary metrics."""
        key = (round(start_lat, 3), round(start_lon, 3), round(goal_lat, 3), round(goal_lon, 3))
        if key in self._route_cache:
            cells = [self._cell_for(c) for c in self._route_cache[key]]  # rebuild from coords
            coords = self._route_cache[key]
        else:
            cells = self.planner.plan(start_lat, start_lon, goal_lat, goal_lon)
            coords = self.planner.coordinates_from_cells(cells)
            self._route_cache[key] = coords

        waypoints = self._build_waypoints(coords)
        total_distance = sum(wp["distance_nm"] for wp in waypoints)
        duration_h = total_distance / self.cruise_speed_knots
        fuel_tons = self._estimate_fuel(duration_h, total_distance)

        return {
            "coords": coords,
            "waypoints": waypoints,
            "total_distance_nm": round(total_distance, 2),
            "estimated_duration_hours": round(duration_h, 2),
            "estimated_fuel_tons": round(fuel_tons, 2),
            "source": "A* multi-objective route optimization",
        }

    def optimize_alternative(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> dict[str, Any]:
        """Return an alternative route that prioritizes safety over distance."""
        prev_safety_weight = self.planner.safety_weight
        prev_distance_weight = self.planner.distance_weight
        self.planner.safety_weight = 0.85
        self.planner.distance_weight = 0.15
        try:
            result = self.optimize(start_lat, start_lon, goal_lat, goal_lon)
            result["source"] = "A* alternative route (safety-prioritized)"
            return result
        finally:
            self.planner.safety_weight = prev_safety_weight
            self.planner.distance_weight = prev_distance_weight

    def _cell_for(self, coord: tuple[float, float]) -> GridCell:
        lat, lon = coord
        i = int(np.argmin(np.abs(self.risk_grid.lats - lat)))
        j = int(np.argmin(np.abs(self.risk_grid.lons - lon)))
        return GridCell(i, j)

    def _build_waypoints(self, coords: list[tuple[float, float]]) -> list[dict[str, Any]]:
        """Convert route coordinates into rich waypoint records."""
        waypoints = []
        prev = None
        cumulative = 0.0
        for idx, (lat, lon) in enumerate(coords):
            rec = {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "distance_nm": 0.0,
                "heading_deg": 0.0,
                "speed_knots": round(self.cruise_speed_knots, 1),
                "risk_level": self._risk_bucket(self.risk_grid.risk_at(lat, lon)),
                "cumulative_distance_nm": round(cumulative, 2),
                "index": idx,
            }
            if prev is not None:
                plon, plat = prev
                rec["distance_nm"] = round(haversine_distance_nm(plat, plon, lat, lon), 2)
                cumulative += rec["distance_nm"]
                rec["heading_deg"] = round(initial_bearing(plat, plon, lat, lon), 1)
                rec["cumulative_distance_nm"] = round(cumulative, 2)
            waypoints.append(rec)
            prev = (lon, lat)
        return waypoints

    @staticmethod
    def _risk_bucket(value: float) -> str:
        if value >= 0.8:
            return "extreme"
        if value >= 0.6:
            return "high"
        if value >= 0.3:
            return "moderate"
        return "low"

    def _estimate_fuel(self, duration_hours: float, distance_nm: float) -> float:
        """Estimate fuel consumption with port/idle adjustment factor."""
        # Baseline consumption scaled by distance inefficiency of the route.
        straight_distance = distance_nm  # route distance already reflects detours
        return self.fuel_consumption_tons_per_day * duration_hours / 24.0

    def interpolate_position(
        self, coords: list[tuple[float, float]], fraction: float
    ) -> tuple[float, float]:
        """Get an interpolated vessel position along the route."""
        total = sum(
            haversine_distance_nm(
                coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
            )
            for i in range(len(coords) - 1)
        )
        if total <= 0:
            return coords[0]
        target = total * fraction
        acc = 0.0
        for i in range(len(coords) - 1):
            seg_len = haversine_distance_nm(
                coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
            )
            if acc + seg_len >= target:
                frac = (target - acc) / seg_len if seg_len else 0.0
                p = intermediate_point(
                    coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], frac
                )
                return p.latitude, p.longitude
            acc += seg_len
        return coords[-1]
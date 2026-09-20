"""Journey simulation engine with two-hour rolling route recalculation.

The journey lifecycle (start / advance / recalculate / status) is owned by the
legacy ``/api/v1/navigation`` endpoints, but the actual planning now runs on the
*same* modern engine as ``POST /api/v1/routes/optimize``: a full risk grid built
from configured sea-ice forecasts, real iceberg predictions, real weather
severity, vessel constraints, hazard no-go zones and the active land mask.

Research stations lie on land; they are approached at their nearest navigable
ocean cell (``resolve_endpoint``), exactly as the route API does, so journey
geometry and the route API cannot diverge.
"""
from __future__ import annotations

import uuid
import math
from datetime import datetime, timezone
from typing import Any

from app.config import load_navigation_config, load_ports, load_research_centers, load_simulation_config
from navigation.approach import resolve_endpoint
from navigation.route_optimizer import RouteOptimizer
from services.route_engine import build_route_engine

from config import settings


_DIRECTION_BEARINGS = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}


def _parse_direction(direction: str) -> float:
    value = direction.strip().upper().replace("°", "")
    if value in _DIRECTION_BEARINGS:
        return _DIRECTION_BEARINGS[value]
    try:
        bearing = float(value) % 360.0
    except ValueError as exc:
        raise ValueError("Direction must be N, NE, E, SE, S, SW, W, NW, or 0-359 degrees") from exc
    return bearing


def _point_in_direction(lat: float, lon: float, bearing: float, distance_km: float = 25.0) -> tuple[float, float]:
    earth_radius_km = 6371.0088
    angular_distance = distance_km / earth_radius_km
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing_rad = math.radians(bearing)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0


class JourneyManager:
    """Manages vessel journeys, including two-hourly route updates.

    The vessel position is advanced along the current route; the position is
    never reset to the original port between updates.
    """

    def __init__(self) -> None:
        sim = load_simulation_config()
        self.update_interval_hours = sim.get("time_step_hours", 2)
        self.max_hours = sim.get("max_simulation_hours", 720)
        self.cruise_speed_knots = sim["vessel"].get("default_speed_knots", 12.0)
        self.fuel_consumption_tons_per_day = 28.0

    def _planner(
        self, vessel: dict[str, Any] | None = None
    ) -> tuple[Any, Any, RouteOptimizer]:
        config = load_navigation_config()
        vessel = vessel or {"cruise_speed_knots": self.cruise_speed_knots}
        grid, engine, _notes, _cls = build_route_engine(
            vessel, datetime.now(timezone.utc), config
        )
        optimizer = RouteOptimizer(grid, engine, vessel=vessel)
        return grid, engine, optimizer

    def create_journey(
        self,
        departure_port: dict[str, Any],
        destination: dict[str, Any],
        journey_mode: str = "outbound",
        vessel: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Initialize a journey and compute the first route."""
        port = departure_port
        center = destination
        origin = center if journey_mode == "return" else port
        goal = port if journey_mode == "return" else center

        journey_id = str(uuid.uuid4())
        grid, engine, optimizer = self._planner(vessel)

        max_approach = float(settings.RESEARCH_STATION_APPROACH_DISTANCE_KM or 100.0)
        centers = load_research_centers()
        ports = load_ports()

        start_ep = resolve_endpoint(
            grid, origin["latitude"], origin["longitude"], centers, ports, max_approach
        )
        goal_ep = resolve_endpoint(
            grid, goal["latitude"], goal["longitude"], centers, ports, max_approach
        )

        # Outbound the vessel leaves from the exact port; return journeys leave
        # from the research-station ocean approach point (the station is land).
        if journey_mode == "return":
            start_lat, start_lon = start_ep.lat, start_ep.lon
        else:
            start_lat, start_lon = float(origin["latitude"]), float(origin["longitude"])
        goal_lat, goal_lon = goal_ep.lat, goal_ep.lon

        route = optimizer.optimize(start_lat, start_lon, goal_lat, goal_lon)

        journey = {
            "journey_id": journey_id,
            "mode": journey_mode,
            "origin": origin["name"],
            "destination": goal["name"],
            "departure_port": port["name"],
            "destination_center": center["name"],
            "current_lat": start_lat,
            "current_lon": start_lon,
            "current_time_hours": 0.0,
            "route_update_count": 1,
            "vessel": vessel or {},
            "route": self._legacy_route(route),
            "goal_lat": goal_lat,
            "goal_lon": goal_lon,
            "completed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return journey

    def advance(self, journey: dict[str, Any]) -> dict[str, Any]:
        """Advance the vessel along the route by the configured time step.

        After advancing, the route is recalculated from the new position.
        This is the two-hour rolling navigation core.
        """
        if journey["completed"]:
            return journey

        # Advance vessel position along the current route by update_interval_hours.
        hours_elapsed = self.update_interval_hours
        distance_traveled_nm = self.cruise_speed_knots * hours_elapsed
        route_distance = journey["route"]["total_distance_nm"]

        if route_distance <= 0:
            fraction = 1.0
        else:
            fraction = min(1.0, distance_traveled_nm / route_distance)

        new_lat, new_lon = self._interpolate(journey["route"]["coords"], fraction)
        journey["current_lat"] = new_lat
        journey["current_lon"] = new_lon
        journey["current_time_hours"] = round(journey["current_time_hours"] + hours_elapsed, 2)

        remaining = self._remaining_distance(new_lat, new_lon, journey)
        if remaining < self.cruise_speed_knots * self.update_interval_hours or journey["current_time_hours"] >= self.max_hours:
            journey["completed"] = True
            journey["current_lat"] = journey["route"]["coords"][-1][0]
            journey["current_lon"] = journey["route"]["coords"][-1][1]
        else:
            # Two-hour rolling recalculation from the current vessel position.
            self._recalculate(journey)

        return journey

    def recalculate(
        self,
        journey: dict[str, Any],
        current_lat: float | None = None,
        current_lon: float | None = None,
        direction: str | None = None,
    ) -> dict[str, Any]:
        """Force a route recalculation from an arbitrary position (live mode)."""
        if current_lat is not None:
            journey["current_lat"] = current_lat
        if current_lon is not None:
            journey["current_lon"] = current_lon
        if direction is not None:
            journey["direction"] = direction
        self._recalculate(journey, direction=direction)
        journey["route_update_count"] += 1
        return journey

    def _recalculate(self, journey: dict[str, Any], direction: str | None = None) -> None:
        """Rebuild risk grid, forecast data, and route from current position."""
        grid, _engine, optimizer = self._planner(journey.get("vessel") or {})
        requested_lat = float(journey["current_lat"])
        requested_lon = float(journey["current_lon"])

        # A vessel position that lands on a configured station/port lies on the
        # land mask; resolve it to that place's ocean approach point exactly as
        # the route API does.
        max_approach = float(settings.RESEARCH_STATION_APPROACH_DISTANCE_KM or 100.0)
        centers = load_research_centers()
        ports = load_ports()
        try:
            pos_ep = resolve_endpoint(
                grid,
                journey["current_lat"],
                journey["current_lon"],
                centers,
                ports,
                max_approach,
            )
        except ValueError:
            pos_ep = None

        route_start_lat = pos_ep.lat if pos_ep is not None else requested_lat
        route_start_lon = pos_ep.lon if pos_ep is not None else requested_lon
        optimized_route = optimizer.optimize(
            route_start_lat,
            route_start_lon,
            journey["goal_lat"],
            journey["goal_lon"],
        )

        # The planner works on grid cells, but the dashboard must show the
        # exact GPS fix supplied by the operator as the first waypoint.
        route_coordinates = optimized_route["coordinates"]
        route_coordinates = [(requested_lat, requested_lon), *route_coordinates]
        effective_direction = direction or journey.get("direction")
        if effective_direction:
            bearing = _parse_direction(effective_direction)
            directional_point = _point_in_direction(requested_lat, requested_lon, bearing)
            route_coordinates.insert(1, directional_point)

        journey["route"] = self._legacy_route(
            optimizer._build_route(
                route_coordinates,
                preference="recommended",
                source=optimized_route["source"],
            )
        )
        journey["route_update_count"] += 1

    @staticmethod
    def _legacy_route(route: dict[str, Any]) -> dict[str, Any]:
        """Adapt a modern engine route dict to the legacy journey contract."""
        return {
            "coords": route["coordinates"],
            "waypoints": route["waypoints"],
            "total_distance_nm": route["distance_nm"],
            "estimated_fuel_tons": route["fuel_tons"],
            "estimated_duration_hours": route["travel_time_hours"],
            "risk_score": route["risk_score"],
            "max_risk_level": route["max_risk_level"],
            "warnings": route["warnings"],
        }

    @staticmethod
    def _interpolate(coords: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
        from app.utils.geo import haversine_distance_nm, intermediate_point

        if len(coords) < 2:
            return coords[0]
        total = sum(
            haversine_distance_nm(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
            for i in range(len(coords) - 1)
        )
        if total <= 0:
            return coords[0]
        target = total * fraction
        acc = 0.0
        for i in range(len(coords) - 1):
            seg = haversine_distance_nm(
                coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
            )
            if acc + seg >= target:
                frac = (target - acc) / seg if seg else 0.0
                p = intermediate_point(
                    coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], frac
                )
                return float(p.latitude), float(p.longitude)
            acc += seg
        return coords[-1]

    @staticmethod
    def _remaining_distance(lat: float, lon: float, journey: dict[str, Any]) -> float:
        """Remaining great-circle distance to the journey goal in nautical miles."""
        from app.utils.geo import haversine_distance_nm

        return haversine_distance_nm(lat, lon, journey["goal_lat"], journey["goal_lon"])
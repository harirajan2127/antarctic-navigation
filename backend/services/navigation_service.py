"""Route-optimisation and route-persistence service.

Wraps the ``navigation`` engine behind a stable interface that also stores
completed route results in the ``routes`` table.

Engine assembly is shared with the journey simulator and the API route
wrapper via ``services.route_engine.build_route_engine``:
    AntarcticGrid.from_config -> RiskEngine(RiskConfig.from_dict)
    sea-ice forecast + iceberg predictions + real weather severity
    + vessel constraints + hazard no-go zones + land mask
    -> optimizer.optimize() / .alternatives()
Every route is rebuilt from the latest data snapshot on each call.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.config import load_navigation_config, load_ports, load_research_centers, load_vessels

from config import DEMO_WARNING, effective_demo, settings
from database.database import SessionLocal
from database.models import RouteRecord
from navigation.astar import NoPathFoundError
from navigation.approach import resolve_endpoint
from navigation.grid import AntarcticGrid
from navigation.route_optimizer import RouteOptimizer
from services.route_engine import build_debug, build_route_engine

logger = logging.getLogger("dss.services.navigation")


def _utcnow() -> datetime:
    from datetime import timezone

    return datetime.now(timezone.utc).replace(tzinfo=None)


def _utc(ts: datetime | None) -> datetime:
    from datetime import timezone

    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _resolve_vessel(vessel_id: str) -> dict[str, Any]:
    vessels = {v["vessel_id"]: v for v in load_vessels()}
    if vessel_id not in vessels:
        raise LookupError(f"Vessel '{vessel_id}' not found.")
    return vessels[vessel_id]


def _clamp_to_region(
    grid: AntarcticGrid, lat: float, lon: float
) -> tuple[float, float, str | None]:
    if grid.contains(lat, lon):
        return lat, lon, None
    lat_c = min(max(lat, grid.lat_min), grid.lat_max)
    lon_c = min(max(lon, grid.lon_min), grid.lon_max)
    warning = (
        f"Point ({lat:.2f}, {lon:.2f}) is outside the Antarctic grid; "
        f"snapped to ({lat_c:.2f}, {lon_c:.2f}). Decision support uses the "
        "snapped position."
    )
    return lat_c, lon_c, warning


def _waypoint_dicts(route: dict[str, Any], label: str) -> dict:
    waypoints = [
        {
            "lat": wp["latitude"],
            "lon": wp["longitude"],
            "risk_score": wp.get("risk_score"),
            "risk_level": wp.get("risk_level"),
            "step": idx,
            "label": label,
        }
        for idx, wp in enumerate(route["waypoints"])
    ]
    coordinates = [[round(la, 4), round(lo, 4)] for la, lo in route["coordinates"]]
    return {
        "waypoints": waypoints,
        "coordinates": coordinates,
        "distance_km": route["distance_km"],
        "distance_nm": route["distance_nm"],
        "travel_time_hours": route["travel_time_hours"],
        "fuel_tons": route["fuel_tons"],
        "risk_score": route["risk_score"],
        "risk_level": route.get("max_risk_level"),
        "warnings": route.get("warnings", []),
    }


class NavigationService:
    """Stateless service — engine state is created per optimisation call."""

    def optimize(
        self,
        start_lat: float,
        start_lon: float,
        dest_lat: float,
        dest_lon: float,
        vessel_id: str = "polar_explorer",
        preference: str = "recommended",
        forecast_time: datetime | None = None,
    ) -> tuple[dict, str]:
        vessel = _resolve_vessel(vessel_id)
        config = load_navigation_config()
        ts = _utc(forecast_time)

        grid, engine, notes, classification = build_route_engine(vessel, ts, config)
        warnings: list[str] = list(notes)

        max_approach = float(settings.RESEARCH_STATION_APPROACH_DISTANCE_KM or 100.0)
        research_centers = load_research_centers()
        ports = load_ports()

        start_lat, start_lon, start_warn = _clamp_to_region(grid, start_lat, start_lon)
        dest_lat, dest_lon, dest_warn = _clamp_to_region(grid, dest_lat, dest_lon)
        warnings.extend(w for w in (start_warn, dest_warn) if w)

        start_ep = resolve_endpoint(
            grid, start_lat, start_lon, research_centers, ports, max_approach
        )
        goal_ep = resolve_endpoint(
            grid, dest_lat, dest_lon, research_centers, ports, max_approach
        )
        if start_ep.note:
            warnings.append(start_ep.note)
        if goal_ep.note:
            warnings.append(goal_ep.note)
        req_start_lat, req_start_lon = start_lat, start_lon
        req_dest_lat, req_dest_lon = dest_lat, dest_lon
        start_lat, start_lon = start_ep.lat, start_ep.lon
        dest_lat, dest_lon = goal_ep.lat, goal_ep.lon

        optimizer = RouteOptimizer(grid, engine, vessel=vessel)
        try:
            recommended = optimizer.optimize(
                start_lat, start_lon, dest_lat, dest_lon, preference=preference
            )
        except NoPathFoundError as exc:
            raise ValueError(str(exc)) from exc

        alt_list: list[dict] = []
        seen_routes = {tuple(recommended["coordinates"])}
        for alternative in optimizer.alternatives(
            start_lat, start_lon, dest_lat, dest_lon, excluded=preference
        ):
            if alternative is None:
                warnings.append(
                    "An alternative route preference could not be calculated "
                    "(no navigable path under that preference)."
                )
                continue
            alt = _waypoint_dicts(alternative, label=alternative["preference"])
            signature = tuple(tuple(point) for point in alt["coordinates"])
            if signature in seen_routes:
                continue
            seen_routes.add(signature)
            alt_list.append(alt)
            if len(alt_list) >= 2:
                break

        recommended_block = _waypoint_dicts(recommended, label="recommended")
        recommended_block["warnings"] = list(
            dict.fromkeys(warnings + recommended_block["warnings"])
        )

        route_id = str(uuid.uuid4())
        demo = effective_demo(classification)
        created_at = _utcnow()

        row = RouteRecord(
            route_id=route_id,
            created_at=created_at,
            start_latitude=round(start_lat, 4),
            start_longitude=round(start_lon, 4),
            destination_latitude=round(dest_lat, 4),
            destination_longitude=round(dest_lon, 4),
            vessel_id=vessel_id,
            preference=preference,
            distance_km=recommended_block["distance_km"],
            distance_nm=recommended_block["distance_nm"],
            travel_time_hours=recommended_block["travel_time_hours"],
            fuel_tons=recommended_block["fuel_tons"],
            risk_score=recommended_block["risk_score"],
            risk_level=recommended_block["risk_level"],
            warnings=recommended_block["warnings"],
            coordinates=recommended_block["coordinates"],
            waypoints=recommended_block["waypoints"],
            alternatives=alt_list,
            demo=demo,
        )
        with SessionLocal() as db:
            db.add(row)
            db.commit()

        result = {
            "route_id": route_id,
            "created_at": created_at,
            "start_latitude": round(start_lat, 4),
            "start_longitude": round(start_lon, 4),
            "destination_latitude": round(dest_lat, 4),
            "destination_longitude": round(dest_lon, 4),
            "requested_start_latitude": round(req_start_lat, 4),
            "requested_start_longitude": round(req_start_lon, 4),
            "requested_destination_latitude": round(req_dest_lat, 4),
            "requested_destination_longitude": round(req_dest_lon, 4),
            "destination_name": goal_ep.place_name,
            "ocean_approach_distance_km": (
                round(goal_ep.approach_distance_km, 1)
                if goal_ep.on_land
                else None
            ),
            "vessel_id": vessel_id,
            "preference": preference,
            "recommended": recommended_block,
            "alternatives": alt_list,
            "demo": False,
            "disclaimer": None,
            "hazard_config": engine.hazard_config(),
            "explanation": engine.explanation(),
            "debug": build_debug(grid, engine, config),
        }
        return result, classification

    def get_route(self, route_id: str) -> dict | None:
        with SessionLocal() as db:
            row = db.scalar(
                select(RouteRecord).where(RouteRecord.route_id == route_id)
            )
            if row is None:
                return None
            return {
                "route_id": row.route_id,
                "created_at": row.created_at,
                "start_latitude": row.start_latitude,
                "start_longitude": row.start_longitude,
                "destination_latitude": row.destination_latitude,
                "destination_longitude": row.destination_longitude,
                "vessel_id": row.vessel_id,
                "preference": row.preference,
                "distance_km": row.distance_km,
                "distance_nm": row.distance_nm,
                "travel_time_hours": row.travel_time_hours,
                "fuel_tons": row.fuel_tons,
                "risk_score": row.risk_score,
                "risk_level": row.risk_level,
                "warnings": row.warnings,
                "coordinates": row.coordinates,
                "waypoints": row.waypoints,
                "alternatives": row.alternatives,
                "demo": row.demo,
                "disclaimer": DEMO_WARNING if row.demo else None,
                "waypoints_total": len(row.waypoints or []),
            }

    def statistics(self) -> dict:
        from sqlalchemy import func

        with SessionLocal() as db:
            rows = db.scalars(select(RouteRecord)).all()
        if not rows:
            return {"total": 0, "by_preference": {}, "demo_count": 0}
        by_preference: dict[str, int] = {}
        demo_count = 0
        for r in rows:
            by_preference[r.preference] = by_preference.get(r.preference, 0) + 1
            if r.demo:
                demo_count += 1
        return {"total": len(rows), "by_preference": by_preference, "demo_count": demo_count}


navigation_service = NavigationService()
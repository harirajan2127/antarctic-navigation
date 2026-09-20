"""Tool functions the assistant can call against the DSS backend services.

Each tool returns a dict with:
* ``name``      - stable tool id
* ``status``    - "ok" or "error"
* ``demo``      - whether the underlying data is synthetic demo data
* ``classification`` - dataset classification string when available
* ``note``      - short honest note about sources/limits (optional)
* ``data``      - the JSON-serialisable payload used as LLM context

Tools reuse the same services as the REST routers, so the chatbot and the API
always agree — including which datasets are demo and which are real.
"""
from __future__ import annotations

import re
from typing import Any

from services import analytics_service, dataset_service
from services.iceberg_service import distance_between_icebergs, iceberg_service
from services.navigation_service import navigation_service
from services.sea_ice_service import sea_ice_service

ICEBERG_ID_RE = re.compile(r"([A-Z]{2,5})-?[A-Z]?\d{3,4}", re.IGNORECASE)


def _safe(service_call):
    """Wrap a service call so one failing tool never breaks the whole answer."""
    try:
        return service_call()
    except Exception as exc:  # noqa: BLE001 - we surface errors to the chatbot
        return {"status": "error", "error": str(exc)}


def _sources_note(data: dict) -> str:
    note = data.get("warning")
    if isinstance(note, str) and note.strip():
        return note.strip()
    if data.get("demo"):
        return "synthetic demo data — no real-world information"
    return "processed pipeline data"


def _core(data: dict, name: str, payload: dict) -> dict:
    return {
        "name": name,
        "status": "ok",
        "demo": bool(data.get("demo")),
        "classification": data.get("classification"),
        "note": _sources_note(data),
        "data": payload,
    }


# --------------------------------------------------------------------------
# Sea ice
# --------------------------------------------------------------------------
def sea_ice_forecast(horizon_hours: int = 24, model: str = "persistence") -> dict:
    data = _safe(lambda: sea_ice_service.forecast(horizon_hours=horizon_hours, model=model))
    if data.get("status") == "error":
        return {"name": "sea_ice_forecast", "status": "error", "note": str(data.get("error"))}
    payload = {
        "forecast_time": data.get("forecast_time"),
        "valid_time": data.get("valid_time"),
        "horizon_hours": data.get("horizon_hours"),
        "model": data.get("model"),
        "mean_concentration": data.get("mean_concentration"),
        "max_concentration": data.get("max_concentration"),
        "coverage_pct": data.get("coverage_pct"),
        "model_used_real": data.get("model_used_real"),
        "skill_note": data.get("skill_note"),
    }
    return _core(data, "sea_ice_forecast", payload)


def sea_ice_current() -> dict:
    data = _safe(lambda: sea_ice_service.current())
    if data.get("status") == "error":
        return {"name": "sea_ice_current", "status": "error", "note": str(data.get("error"))}
    conc = data.get("concentration") or []
    flat = [v for row in conc for v in row if v is not None]
    payload = {
        "timestamp": data.get("timestamp"),
        "grid_size": f"{len(conc)}x{len(conc[0]) if conc else 0}",
        "mean_concentration": round(sum(flat) / len(flat), 6) if flat else 0.0,
        "max_concentration": max(flat) if flat else 0.0,
    }
    return _core(data, "sea_ice_current", payload)


# --------------------------------------------------------------------------
# Icebergs
# --------------------------------------------------------------------------
def iceberg_list() -> dict:
    data = _safe(lambda: iceberg_service.list_icebergs())
    if data.get("status") == "error":
        return {"name": "iceberg_list", "status": "error", "note": str(data.get("error"))}
    payload = {
        "count": data.get("count"),
        "icebergs": [
            {
                "iceberg_id": i.get("iceberg_id"),
                "latitude": i.get("latitude"),
                "longitude": i.get("longitude"),
            }
            for i in (data.get("icebergs") or [])
        ],
    }
    return _core(data, "iceberg_list", payload)


def iceberg_trajectory(iceberg_id: str, horizon_hours: int = 24) -> dict:
    def _call():
        traj = iceberg_service.trajectory(iceberg_id)
        if traj is None:
            raise ValueError(f"Iceberg {iceberg_id} was not found")
        return iceberg_service.predict([iceberg_id], horizon_hours=horizon_hours), traj

    result = _safe(_call)
    if isinstance(result, dict) and result.get("status") == "error":
        return {"name": "iceberg_trajectory", "status": "error", "note": str(result.get("error"))}
    pred, traj = result
    obs = traj.get("observations") or []
    predictions = traj.get("predictions") or []
    predicted = (pred.get("icebergs") or [{}])[0]
    payload = {
        "iceberg_id": iceberg_id,
        "observed_positions": [
            {"time": o.get("timestamp"), "latitude": o.get("latitude"), "longitude": o.get("longitude")}
            for o in obs[-20:]
        ],
        "predicted_positions": [
            {
                "horizon_hours": p.get("horizon_hours", horizon_hours),
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
            }
            for p in predictions
        ],
        "predicted_latitude": predicted.get("predicted_lat"),
        "predicted_longitude": predicted.get("predicted_lon"),
        "prediction_model": pred.get("model"),
        "confidence": predicted.get("confidence"),
    }
    return _core(traj, "iceberg_trajectory", payload)


def iceberg_distance(iceberg_a: str, iceberg_b: str) -> dict:
    def _call():
        d = iceberg_service.distance(iceberg_a, iceberg_b)
        if d is None:
            raise ValueError(
                f"One of {iceberg_a} / {iceberg_b} is not a tracked iceberg"
            )
        return d

    data = _safe(_call)
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "iceberg_distance", "status": "error", "note": str(data.get("error"))}
    payload = {
        "iceberg_a": data.get("iceberg_a"),
        "iceberg_b": data.get("iceberg_b"),
        "distance_km": data.get("distance_km"),
        "distance_nm": data.get("distance_nm"),
    }
    return _core(data, "iceberg_distance", payload)


def extracted_iceberg_ids(question: str) -> list[str]:
    """Return iceberg IDs mentioned in a question (up to 2)."""
    found = []
    for match in ICEBERG_ID_RE.finditer(question):
        token = match.group(0).upper()
        if token not in found and token not in ("DEMO", "LSTM", "RF"):
            found.append(token)
        if len(found) >= 2:
            break
    return found


# --------------------------------------------------------------------------
# Routes / fuel
# --------------------------------------------------------------------------
def route_details(
    start_lat: float = -65.0,
    start_lon: float = 140.0,
    dest_lat: float = -77.8469,
    dest_lon: float = 166.6687,
    vessel_id: str = "polar_explorer",
    preference: str = "recommended",
) -> dict:
    def _call() -> dict:
        result, _classification = navigation_service.optimize(
            start_lat=start_lat,
            start_lon=start_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            vessel_id=vessel_id,
            preference=preference,
        )
        return result

    data = _safe(_call)
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "route_details", "status": "error", "note": str(data.get("error"))}
    recommended = data.get("recommended") or {}
    payload = {
        "route_id": data.get("route_id"),
        "start": (data.get("start_latitude"), data.get("start_longitude")),
        "destination": (data.get("destination_latitude"), data.get("destination_longitude")),
        "vessel_id": data.get("vessel_id"),
        "preference": data.get("preference"),
        "recommended": {
            "distance_km": recommended.get("distance_km"),
            "distance_nm": recommended.get("distance_nm"),
            "travel_time_hours": recommended.get("travel_time_hours"),
            "fuel_tons": recommended.get("fuel_tons"),
            "risk_level": recommended.get("risk_level"),
            "risk_score": recommended.get("risk_score"),
            "warnings": recommended.get("warnings"),
        },
        "alternatives": [
            {
                "label": f"Alternative {idx + 1}",
                "distance_km": a.get("distance_km"),
                "travel_time_hours": a.get("travel_time_hours"),
                "fuel_tons": a.get("fuel_tons"),
                "risk_level": a.get("risk_level"),
            }
            for idx, a in enumerate(data.get("alternatives") or [])
        ],
    }
    return _core(data, "route_details", payload)


def fuel_estimate(
    start_lat: float = -65.0,
    start_lon: float = 140.0,
    dest_lat: float = -77.8469,
    dest_lon: float = 166.6687,
    vessel_id: str = "polar_explorer",
    preference: str = "recommended",
) -> dict:
    """Derive fuel estimates for the recommended route and alternatives."""
    data = _safe(
        lambda: navigation_service.optimize(
            start_lat=start_lat,
            start_lon=start_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            vessel_id=vessel_id,
            preference=preference,
        )[0]
    )
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "fuel_estimate", "status": "error", "note": str(data.get("error"))}
    recommended = data.get("recommended") or {}
    payload = {
        "route_id": data.get("route_id"),
        "vessel_id": data.get("vessel_id"),
        "recommended_fuel_tons": recommended.get("fuel_tons"),
        "recommended_preference": data.get("preference"),
        "alternatives": [
            {
                "label": f"Alternative {idx + 1}",
                "fuel_tons": a.get("fuel_tons"),
                "preference": a.get("preference"),
            }
            for idx, a in enumerate(data.get("alternatives") or [])
        ],
    }
    return _core(data, "fuel_estimate", payload)


# --------------------------------------------------------------------------
# Model metrics / datasets
# --------------------------------------------------------------------------
def model_metrics(pipeline: str | None = None) -> dict:
    def _call():
        out = analytics_service.model_metrics()
        if pipeline:
            out["metrics"] = [m for m in out.get("metrics", []) if m.get("pipeline") == pipeline]
            out["count"] = len(out["metrics"])
        return out

    data = _safe(_call)
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "model_metrics", "status": "error", "note": str(data.get("error"))}
    payload = {
        "count": data.get("count"),
        "metrics": [
            {
                "pipeline": m.get("pipeline"),
                "model": m.get("model"),
                "metric_name": m.get("metric_name"),
                "metric_value": m.get("metric_value"),
                "demo": m.get("demo"),
            }
            for m in (data.get("metrics") or [])
        ],
    }
    return _core(data, "model_metrics", payload)


def datasets_status() -> dict:
    data = _safe(dataset_service.from_database)
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "datasets_status", "status": "error", "note": str(data.get("error"))}
    real = [d for d in data if not d.get("demo")]
    payload = {
        "real_data_available": len(real) > 0,
        "datasets": [
            {"dataset": d.get("dataset"), "classification": d.get("classification"), "demo": d.get("demo")}
            for d in data
        ],
    }
    return {
        "name": "datasets_status",
        "status": "ok",
        "demo": not payload["real_data_available"],
        "classification": None,
        "note": "synthetic demo data" if not payload["real_data_available"] else "real/pipeline data available",
        "data": payload,
    }


def closest_iceberg() -> dict:
    """Pick the tracked iceberg nearest a reference vessel position (haversine)."""

    def _call():
        items = (iceberg_service.list_icebergs() or {}).get("icebergs") or []
        if not items:
            raise ValueError("No tracked icebergs available")
        lat, lon = -66.5, 140.0
        best, best_km = None, float("inf")
        for it in items:
            km = distance_between_icebergs(
                it["latitude"], it["longitude"], lat, lon
            ).get("distance_km", float("inf"))
            if km < best_km:
                best, best_km = it, km
        return {"items": items, "best": best, "best_km": best_km}

    data = _safe(_call)
    if isinstance(data, dict) and data.get("status") == "error":
        return {"name": "closest_iceberg", "status": "error", "note": str(data.get("error"))}
    best = data["best"]
    payload = {
        "reference_position_lat": -66.5,
        "reference_position_lon": 140.0,
        "closest_iceberg": best.get("iceberg_id"),
        "closest_distance_km": round(data["best_km"], 2),
        "how": "haversine distance from a reference vessel position (-66.5, 140.0)",
    }
    return {
        "name": "closest_iceberg",
        "status": "ok",
        "demo": bool(best.get("demo")),
        "classification": None,
        "note": "computed in-process from latest reported positions",
        "data": payload,
    }


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
TOOLS: dict[str, Any] = {
    "sea_ice_forecast": sea_ice_forecast,
    "sea_ice_current": sea_ice_current,
    "iceberg_list": iceberg_list,
    "iceberg_trajectory": iceberg_trajectory,
    "iceberg_distance": iceberg_distance,
    "route_details": route_details,
    "fuel_estimate": fuel_estimate,
    "model_metrics": model_metrics,
    "datasets_status": datasets_status,
    "closest_iceberg": closest_iceberg,
}


def run_tool(name: str, **kwargs) -> dict:
    """Invoke a tool by name; missing kwargs use the tool's defaults."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"name": name, "status": "error", "note": f"unknown tool: {name}"}
    return fn(**kwargs)
"""Real-data validation checks for navigation evaluation reports."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from config import settings

LOGGER = logging.getLogger("dss.evaluation_checks")
EARTH_RADIUS_KM = 6371.0088
STATUS_VERIFIED = "VERIFIED"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_UNAVAILABLE = "UNAVAILABLE"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def score_distance(error_km: float, tolerance_km: float) -> float:
    """Score 100 within tolerance, linearly to zero at twice tolerance."""
    if not math.isfinite(error_km) or tolerance_km <= 0:
        return 0.0
    return round(max(0.0, min(100.0, (2 * tolerance_km - error_km) / tolerance_km * 100)), 4)


def evaluate_two_hour_predictions(
    csv_path: str | Path,
    predictor: Callable[[dict[str, Any], float], tuple[float, float]] | None = None,
    *,
    target_interval_hours: float = 2.0,
    interval_tolerance_hours: float = 0.25,
    error_tolerance_km: float = 20.0,
) -> dict[str, Any]:
    """Evaluate real 2-hour prediction pairs without inventing ground truth.

    Only consecutive observations whose actual timestamp gap is within the
    configured interval tolerance are scored. A resampled/interpolated point
    may be used for display, but it is never counted as ground truth here.
    """
    path = Path(csv_path)
    if not path.exists():
        return {"status": STATUS_UNAVAILABLE, "message": "2-hour ground-truth data unavailable", "missing_or_invalid_records": 0, "target_percent": settings.TWO_HOUR_EVALUATION_TARGET_PERCENT}

    df = pd.read_csv(path)
    required = {"iceberg_id", "timestamp", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        return {"status": STATUS_UNAVAILABLE, "message": f"Missing required fields: {', '.join(sorted(missing))}", "target_percent": settings.TWO_HOUR_EVALUATION_TARGET_PERCENT}

    before = len(df)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    duplicate_count = int(df.duplicated(["iceberg_id", "timestamp"], keep="first").sum())
    invalid_count = int((df["timestamp"].isna() | df["latitude"].isna() | df["longitude"].isna()).sum())
    df = df.dropna(subset=list(required)).drop_duplicates(["iceberg_id", "timestamp"])
    df = df.sort_values(["iceberg_id", "timestamp"])

    intervals: list[float] = []
    pairs: list[dict[str, Any]] = []
    for iceberg_id, group in df.groupby("iceberg_id", sort=False):
        group = group.reset_index(drop=True)
        for index in range(1, len(group)):
            previous = group.iloc[index - 1]
            actual = group.iloc[index]
            gap = (actual["timestamp"] - previous["timestamp"]).total_seconds() / 3600
            if gap <= 0 or not math.isfinite(gap):
                invalid_count += 1
                continue
            intervals.append(gap)
            if abs(gap - target_interval_hours) <= interval_tolerance_hours:
                prediction = (float(previous["latitude"]), float(previous["longitude"]))
                if predictor is not None:
                    prediction = predictor(previous.to_dict(), gap)
                error_km = haversine_km(prediction[0], prediction[1], float(actual["latitude"]), float(actual["longitude"]))
                pairs.append({
                    "iceberg_id": str(iceberg_id),
                    "timestamp": str(previous["timestamp"]),
                    "latitude": float(previous["latitude"]),
                    "longitude": float(previous["longitude"]),
                    "predicted_latitude": float(prediction[0]),
                    "predicted_longitude": float(prediction[1]),
                    "actual_timestamp": str(actual["timestamp"]),
                    "prediction_horizon_hours": gap,
                    "distance_error_km": error_km,
                    "distance_error_nm": error_km / 1.852,
                    "score": score_distance(error_km, error_tolerance_km),
                })

    actual_interval = float(np.median(intervals)) if intervals else None
    tracks = int(df["iceberg_id"].nunique())
    LOGGER.info("2-hour evaluation: interval=%s hours tracks=%d pairs=%d valid=%d invalid=%d", actual_interval, tracks, len(pairs), len(pairs), invalid_count + duplicate_count)
    base = {
        "target_percent": settings.TWO_HOUR_EVALUATION_TARGET_PERCENT,
        "dataset_time_interval_hours": actual_interval,
        "number_of_iceberg_tracks": tracks,
        "number_of_2h_prediction_pairs": len(pairs),
        "number_of_valid_2h_evaluations": len(pairs),
        "missing_or_invalid_records": invalid_count + duplicate_count,
        "duplicate_records": duplicate_count,
        "records_before_cleaning": before,
        "prediction_horizon_hours": target_interval_hours,
        "error_tolerance_km": error_tolerance_km,
        "predictions": pairs,
    }
    if not pairs:
        base.update({"status": STATUS_INSUFFICIENT, "message": "Insufficient 2-hour ground-truth data", "accuracy_percent": None, "average_distance_error_km": None, "average_distance_error_nm": None})
        LOGGER.warning("Insufficient 2-hour ground-truth data; median available interval=%s", actual_interval)
        return base

    errors = np.asarray([row["distance_error_km"] for row in pairs], dtype=float)
    scores = np.asarray([row["score"] for row in pairs], dtype=float)
    base.update({
        "status": STATUS_VERIFIED,
        "message": None,
        "accuracy_percent": round(float(scores.mean()), 4),
        "average_distance_error_km": round(float(errors.mean()), 4),
        "average_distance_error_nm": round(float(errors.mean() / 1.852), 4),
    })
    LOGGER.info("2-hour results: average_error_km=%.3f score=%.3f", errors.mean(), scores.mean())
    return base


def evaluate_land_mask_predictions(
    predictions: Iterable[dict[str, Any]],
    grid: Any,
    mask: np.ndarray | None,
    *,
    coastline_tolerance_km: float = 5.0,
) -> dict[str, Any]:
    """Classify predicted geographic points against the loaded land grid."""
    if mask is None:
        return {"status": STATUS_UNAVAILABLE, "message": "Land-mask data unavailable", "accuracy_percent": None, "target_percent": settings.LAND_MASK_EVALUATION_TARGET_PERCENT}

    land_indices = np.argwhere(np.asarray(mask, dtype=bool))
    ocean = land = invalid = 0
    records = []
    for prediction in predictions:
        lat = prediction.get("predicted_latitude", prediction.get("predicted_lat"))
        lon = prediction.get("predicted_longitude", prediction.get("predicted_lon"))
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)) or not grid.contains(lat, lon):
            invalid += 1
            continue
        cell = grid.cell_for(lat, lon)
        on_land = bool(mask[cell.lat_index, cell.lon_index])
        if not on_land and len(land_indices):
            distances = [haversine_km(lat, lon, float(grid.lats[i]), float(grid.lons[j])) for i, j in land_indices]
            on_land = min(distances) <= coastline_tolerance_km
        record = {"latitude": lat, "longitude": lon, "on_land": on_land, "valid_ocean": not on_land}
        records.append(record)
        if on_land:
            land += 1
        else:
            ocean += 1

    total = ocean + land + invalid
    accuracy = round(100 * ocean / total, 4) if total else None
    result = {
        "target_percent": settings.LAND_MASK_EVALUATION_TARGET_PERCENT,
        "status": STATUS_VERIFIED if total else STATUS_UNAVAILABLE,
        "message": None if total else "No valid predicted coordinates",
        "total_predicted_points": total,
        "ocean_points": ocean,
        "land_points": land,
        "invalid_coordinates": invalid,
        "coastline_tolerance_km": coastline_tolerance_km,
        "accuracy_percent": accuracy,
        "predictions": records,
    }
    LOGGER.info("land mask results: total=%d ocean=%d land=%d invalid=%d score=%s", total, ocean, land, invalid, accuracy)
    return result

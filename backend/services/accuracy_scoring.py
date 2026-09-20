"""Validated weighted accuracy aggregation for the evaluation summary."""
from __future__ import annotations

import math
from typing import Any, Mapping

WEIGHTS = {
    "sea_ice": 0.35,
    "iceberg": 0.30,
    "route": 0.20,
    "recalc": 0.10,
    "land_mask": 0.05,
}
LABELS = {
    "sea_ice": "Sea-Ice Model (R2)",
    "iceberg": "Iceberg Model (km error)",
    "route": "Route Engine",
    "recalc": "2-Hour Recalculation",
    "land_mask": "Land Mask",
}


def _numeric_accuracy(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 100.0:
        return None
    return number


def _validate_weights(weights: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in WEIGHTS:
        try:
            value = float(weights[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid weight for {key}") from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Weight for {key} must be between 0 and 1")
        normalized[key] = value
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Component weights must sum to 1.0, got {sum(normalized.values()):.12f}")
    return normalized


def calculate_weighted_accuracy(
    components: Mapping[str, Any],
    *,
    weights: Mapping[str, Any] = WEIGHTS,
    component_status: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Calculate final or provisional weighted accuracy without fake values."""
    valid_weights = _validate_weights(weights)
    breakdown: list[dict[str, Any]] = []
    available_weighted_sum = 0.0
    available_weight = 0.0
    missing: list[str] = []

    for key, weight in valid_weights.items():
        raw = components.get(key)
        accuracy = _numeric_accuracy(raw)
        if accuracy is None:
            status = (component_status or {}).get(key, "N/A")
            contribution = None
            missing.append(key)
        else:
            status = "Valid"
            contribution = accuracy * weight
            available_weighted_sum += contribution
            available_weight += weight
        breakdown.append({
            "component": key,
            "label": LABELS[key],
            "accuracy": accuracy,
            "weight": weight,
            "weighted_contribution": contribution,
            "status": status,
        })

    available_weight = round(available_weight, 12)
    available_weighted_sum = round(available_weighted_sum, 12)
    coverage = available_weight * 100.0
    provisional = available_weighted_sum / available_weight if available_weight else None
    complete = not missing
    return {
        "score": provisional,
        "provisional": not complete and provisional is not None,
        "available_weighted_sum": available_weighted_sum,
        "available_weight": available_weight,
        "available_weight_coverage_percent": coverage,
        "missing_components": missing,
        "status": "Valid" if complete else ("N/A" if not available_weight else "Provisional"),
        "breakdown": breakdown,
    }

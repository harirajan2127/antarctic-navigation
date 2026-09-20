#!/usr/bin/env python3
"""Compute a single total accuracy figure (% out of 100) for the DSS from the
real-data evaluation reports produced by scripts/evaluate_real_models.py.

The total is a weighted combination of the five validated components. Weights
reflect operational importance for Antarctic navigation safety.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from services.accuracy_scoring import WEIGHTS, calculate_weighted_accuracy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT = PROJECT_ROOT / "reports" / "real_model_accuracy_results.json"


def load() -> dict:
    if not REPORT.exists():
        print(f"Missing report: {REPORT}")
        sys.exit(1)
    with open(REPORT, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    results = load()
    data = results.get("data", {})

    # --- Component accuracies (each already 0..100) ------------------------ #
    si = data.get("sea_ice", {})
    si_r2 = si.get("metrics", {}).get("r2")
    si_acc = float(si_r2) * 100.0 if si_r2 is not None else None

    ib = data.get("iceberg", {})
    ib_mean_km = ib.get("haversine_mean_km")
    # 0 km error -> 100%, 100 km error -> 0%
    ib_acc = (1.0 - float(ib_mean_km) / 100.0) * 100.0 if ib_mean_km is not None else None

    route_acc = 100.0 if data.get("route", {}).get("status") == "VERIFIED" else None
    recalc_data = data.get("two_hour", {})
    recalc_acc = recalc_data.get("accuracy_percent") if recalc_data.get("status") == "VERIFIED" else None
    mask_data = data.get("land_mask", {})
    mask_acc = mask_data.get("accuracy_percent") if mask_data.get("status") == "VERIFIED" else None

    components = {
        "sea_ice": si_acc,
        "iceberg": ib_acc,
        "route": route_acc,
        "recalc": recalc_acc,
        "land_mask": mask_acc,
    }

    source_status = {
        "recalc": "Insufficient data" if recalc_data.get("status") == "INSUFFICIENT_DATA" else "Error" if recalc_data.get("status") == "FAILED" else "N/A",
        "land_mask": "Insufficient data" if mask_data.get("status") == "INSUFFICIENT_DATA" else "Error" if mask_data.get("status") == "FAILED" else "N/A",
    }
    calculation = calculate_weighted_accuracy(components, weights=WEIGHTS, component_status=source_status)
    total = calculation["score"]

    print("=" * 70)
    print("  ANTARCTIC DSS - TOTAL MODEL ACCURACY (out of 100%)")
    print("=" * 70)
    print(f"  Source: reports/real_model_accuracy_results.json\n")
    order = [("Sea-Ice Model (R2)", "sea_ice"),
             ("Iceberg Model (km error)", "iceberg"),
             ("Route Engine", "route"),
             ("2-Hour Recalculation", "recalc"),
             ("Land Mask", "land_mask")]
    print("  Component                          Accuracy   Weight   Contribution   Status")
    print("  " + "-" * 78)
    for label, key in order:
        row = next(item for item in calculation["breakdown"] if item["component"] == key)
        value = f"{row['accuracy']:.2f}%" if row["accuracy"] is not None else "N/A"
        contribution = f"{row['weighted_contribution']:.2f}" if row["weighted_contribution"] is not None else "N/A"
        print(f"  {label:<30}  {value:>7}    {row['weight']*100:>5.1f}%       {contribution:>7}      {row['status']}")
    print("  " + "-" * 58)
    if calculation["provisional"]:
        print(f"  TOTAL WEIGHTED ACCURACY: PROVISIONAL {total:.2f}%")
    elif total is None:
        print("  TOTAL WEIGHTED ACCURACY:              N/A")
    else:
        print(f"  TOTAL WEIGHTED ACCURACY:            {total:>6.2f}%")
    print(f"  AVAILABLE WEIGHT COVERAGE:           {calculation['available_weight_coverage_percent']:.1f}%")
    if calculation["missing_components"]:
        missing_labels = [next(item["label"] for item in calculation["breakdown"] if item["component"] == key) for key in calculation["missing_components"]]
        print(f"  MISSING COMPONENT:                    {', '.join(missing_labels)}")

    if total is None:
        print("\n  Overall Grade: Not Available")
    elif calculation["provisional"]:
        print("\n  OVERALL GRADE: PROVISIONAL")
    else:
        grade = "A (Production-Ready)" if total >= 85 else "B (Research-Grade)" if total >= 70 \
            else "C (Functional)" if total >= 50 else "D (Experimental)"
        print(f"\n  Overall Grade: {grade}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
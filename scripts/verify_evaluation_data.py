#!/usr/bin/env python3
"""Verify the real evaluation datasets used by the accuracy evaluation.

Reads ``Real data/processed`` and reports row counts, ranges, missing-value
counts, duplicate rows, distinct keys and — most importantly — whether future
ground-truth positions exist for icebergs and whether the sea-ice part has
multiple observation timesteps (required to measure any next-day skill).

Usage::

    python scripts/verify_evaluation_data.py --check

``--check`` exits with a non-zero code when a required ground-truth condition
is not met (used by CI / pipelines).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluate_real_models import (
    PROJECT_ROOT,
    REAL_DATA,
    REPORTS_DIR,
    STATUS_NOT_AVAILABLE,
    STATUS_PASSED,
    STATUS_FAILED,
    save_json,
    verify_evaluation_data,
)

FAIL_LIST = [
    "Missing real iceberg positions with future ground-truth observations.",
    "Missing a multi-timestep real sea-ice CSV (at least 4 distinct timestamps).",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real evaluation data.")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if required ground-truth data is missing.")
    args = parser.parse_args()

    print("=" * 70)
    print("  ANTARCTIC DSS - REAL EVALUATION DATA VERIFICATION")
    print("=" * 70)
    print(f"  Root: {PROJECT_ROOT}")

    out = verify_evaluation_data()
    datasets = out.get("datasets", {})

    print("\n  Datasets:")
    problems: list[str] = []
    for name, d in datasets.items():
        if d.get("status") == STATUS_NOT_AVAILABLE:
            print(f"    - {name}: NOT AVAILABLE ({d.get('file', 'no file')})")
            continue
        print(f"    - {name}: {d.get('file')}")
        print(f"        rows={d.get('rows')} cols={len(d.get('columns', []))} "
              f"missing={d.get('missing')} duplicates={d.get('duplicate_rows')}")
        for key in ("latitude_range", "longitude_range", "timestamp_range",
                    "distinct_timestamps", "distinct_icebergs",
                    "ground_truth_future", "concentration_range"):
            if key in d:
                print(f"        {key}={d.get(key)}")
        if name == "iceberg" and d.get("ground_truth_future") is not True:
            problems.append(FAIL_LIST[0])
        if name == "sea_ice" and (d.get("distinct_timestamps") or 0) < 4:
            problems.append(FAIL_LIST[1])

    # Persist machine-readable verification output next to the reports.
    report = {
        "timestamp": str(np.datetime64("now", "s")),
        "data_mode": "real",
        "status": STATUS_PASSED if not problems else STATUS_FAILED,
        "problems": problems,
        "datasets": datasets,
    }
    save_json(REPORTS_DIR / "evaluation_data_report.json", report)

    print(f"\n  Verification status: {report['status']}")
    if problems:
        for p in problems:
            print(f"    - PROBLEM: {p}")

    if args.check:
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
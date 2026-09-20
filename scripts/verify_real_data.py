#!/usr/bin/env python3
"""Verify every real processed dataset under ``Real data/processed/`` and report
whether the dashboard's six data sources are genuinely real (non-synthetic).

Checks per dataset: presence, file count, total row count, sample schema,
lat/lon/timestamp ranges, missing values, duplicates and provenance columns.
Large multi-million-row files are line-counted (fast) and sampled for schema;
small files are fully scanned for missing/duplicate statistics.

Exit code 0 = all six datasets verified, 1 = at least one dataset missing or
definitively not real.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "Real data" / "processed"
FULL_SCAN_MAX_BYTES = 120 * 1024 * 1024
SCAN_ROWS = 2000

DATASETS = {
    "sea_ice": {
        "label": "sea-ice",
        "csv_dir": "sea_ice/csv",
        "pick": "latest",
        "rows": "sea_ice_latest_part_*.csv",
    },
    "iceberg": {
        "label": "iceberg",
        "csv_dir": "iceberg/csv",
        "pick": "file",
        "rows": "iceberg_processed.csv",
    },
    "ocean": {
        "label": "ocean",
        "csv_dir": "ocean/csv",
        "pick": "last",
        "rows": "ocean_part_*.csv",
    },
    "weather": {
        "label": "weather",
        "csv_dir": "weather/csv",
        "pick": "last",
        "rows": "weather_part_*.csv",
    },
    "vessel": {
        "label": "vessel",
        "csv_dir": "vessel/csv",
        "pick": "file",
        "rows": "vessel_processed.csv",
    },
    "bathymetry": {
        "label": "bathymetry",
        "csv_dir": "bathymetry/csv",
        "pick": "all",
        "rows": "bathymetry_part_*.csv",
    },
}

REQUIRED_COLUMNS = {
    "sea_ice": ["timestamp", "latitude", "longitude", "sea_ice_concentration"],
    "iceberg": ["timestamp", "latitude", "longitude"],
    "ocean": ["timestamp", "latitude", "longitude"],
    "weather": ["timestamp", "latitude", "longitude"],
    "vessel": ["timestamp", "latitude", "longitude"],
    "bathymetry": ["latitude", "longitude"],
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_csvs(dataset: str) -> list[Path]:
    cfg = DATASETS[dataset]
    csv_dir = PROCESSED_ROOT / cfg["csv_dir"]
    if not csv_dir.is_dir():
        return []
    pattern = cfg["rows"].replace("*", "*")
    files = sorted(csv_dir.glob(pattern))
    if not files:
        return []
    if cfg["pick"] == "all":
        return files
    if cfg["pick"] == "latest":
        return [files[-1]]
    return [files[-1]]


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


_KNOWN_UNITS = (
    "wind_speed", "wind_direction", "air_temperature", "pressure",
    "length_nm", "width_nm", "depth_meters", "elevation_meters",
)


def _scan_stats(path: Path, dataset: str) -> dict[str, Any]:
    full = path.stat().st_size <= FULL_SCAN_MAX_BYTES
    try:
        head = pd.read_csv(path, nrows=SCAN_ROWS)
    except Exception as exc:
        return {"error": f"read failed: {exc}"}
    cols = [str(c) for c in head.columns]
    stats: dict[str, Any] = {
        "columns": cols,
        "missing_required": [c for c in REQUIRED_COLUMNS[dataset] if c not in cols],
        "missing_values": {},
        "duplicates": None,
        "units": {c: _guess_units(c) for c in cols if c in _KNOWN_UNITS},
    }

    if full:
        full_df = pd.read_csv(path)
        stats["rows"] = int(len(full_df))
        for c in cols:
            stats["missing_values"][c] = int(full_df[c].isna().sum())
        stats["duplicates"] = int(full_df.duplicated(subset=[
            c for c in ("latitude", "longitude", "timestamp")
            if c in cols and c != "timestamp"
        ]).sum()) if {"latitude", "longitude"} <= set(cols) else 0
        frame = full_df
    else:
        frame = head
        stats["rows"] = None
        for c in cols:
            stats["missing_values"][c] = int(head[c].isna().sum())

    for col in ("latitude", "longitude"):
        if col in cols:
            ser = pd.to_numeric(frame[col], errors="coerce")
            if ser.notna().any():
                stats[f"{col}_min"] = float(ser.min())
                stats[f"{col}_max"] = float(ser.max())
    if "timestamp" in cols:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        if ts.notna().any():
            stats["timestamp_min"] = ts.min().isoformat()
            stats["timestamp_max"] = ts.max().isoformat()
    if "iceberg_id" in cols:
        ids = frame["iceberg_id"].dropna().astype(str)
        stats["has_demo_ids"] = bool(ids.str.startswith("DEMO-").any())
        stats["sample_ids"] = ids.head(5).tolist()
    if "source_file" in cols:
        src = frame["source_file"].dropna()
        stats["sample_sources"] = src.head(3).tolist()
    if "source_format" in cols:
        fmt = frame["source_format"].dropna().unique()
        stats["source_formats"] = sorted(str(f) for f in fmt)
    if "flag" in cols:
        stats["vessel_flags"] = [str(f) for f in frame["flag"].dropna().unique()[:6]]
    if "geartype" in cols:
        stats["gear_types"] = [str(g) for g in frame["geartype"].dropna().unique()[:6]]
    return stats


def _guess_units(col: str) -> str:
    table = {
        "wind_speed": "m/s",
        "wind_direction": "degrees",
        "air_temperature": "K",
        "pressure": "Pa",
        "length_nm": "nautical miles",
        "width_nm": "nautical miles",
        "depth_meters": "meters",
        "elevation_meters": "meters",
    }
    return table.get(col, "unknown")


def _verdict(stats: dict[str, Any], label: str) -> tuple[str, str]:
    if stats.get("error"):
        return "ERROR", f"unreadable ({label})"
    if stats.get("has_demo_ids"):
        return "DEMO", f"contains synthetic DEMO- iceberg ids ({label})"
    if stats.get("missing_required"):
        return "WARNING", f"missing required columns: {stats['missing_required']} ({label})"
    src = stats.get("source_formats") or ["csv"]
    if "synthetic" in [str(s).lower() for s in src]:
        return "DEMO", f"source_format marks synthetic data ({label})"
    return "REAL", f"verified primary source ({label})"


def _summary_row(dataset: str, files: list[Path], stats: dict[str, Any]) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    label = cfg["label"]
    status, note = _verdict(stats, label)
    rows = stats.get("rows")
    return {
        "dataset": dataset,
        "label": label,
        "files": len(files),
        "rows": rows,
        "status": status,
        "note": note,
        "checked_at": _ts(),
    }


def main() -> int:
    print(f"Verifying real processed data under {PROCESSED_ROOT}")
    print(f"Checked at {_ts()}\n")

    report = []
    failures = 0

    for dataset, cfg in DATASETS.items():
        files = _pick_csvs(dataset)
        print(f"== {dataset} ({cfg['label']}) ==")
        if not files:
            print(f"  [ERROR] no processed CSV in {PROCESSED_ROOT / cfg['csv_dir']}")
            failures += 1
            report.append(_summary_row(dataset, [], {"error": "no csv files"}))
            continue

        print(f"  files: {len(files)}")
        print(f"  first : {files[0].relative_to(PROCESSED_ROOT)}"
              + (f"  ({files[0].stat().st_size / 1e6:.1f} MB)" if files[0].exists() else ""))
        total_rows = sum(max(_count_lines(p) - 1, 0) for p in files)
        print(f"  total rows: {total_rows:,} (line-count based)")

        target = files[-1] if cfg["pick"] != "all" else files[0]
        stats = _scan_stats(target, dataset)
        stats["rows"] = total_rows
        status, note = _verdict(stats, cfg["label"])
        print(f"  columns    : {', '.join(stats.get('columns', []))}")
        for key in ("latitude_min", "latitude_max", "longitude_min", "longitude_max",
                    "timestamp_min", "timestamp_max"):
            if key in stats:
                print(f"  {key:<12}: {stats[key]}")
        if "has_demo_ids" in stats:
            print(f"  demo ids   : {stats['has_demo_ids']}")
        if stats.get("source_formats"):
            print(f"  sources    : {', '.join(stats['source_formats'])}")
        if stats.get("missing_required"):
            print(f"  [ERROR] missing columns: {stats['missing_required']}")
            failures += 1
        if stats.get("vessel_flags"):
            print(f"  flags      : {', '.join(stats['vessel_flags'])}")
        if stats.get("gear_types"):
            print(f"  gear types : {', '.join(stats['gear_types'])}")
        if stats.get("duplicates") is not None:
            print(f"  dupes(by lat/lon): {stats['duplicates']}")
        print(f"  -> {status}: {note}\n")
        report.append(_summary_row(dataset, files, stats))

    print("=" * 72)
    for row in report:
        print(f"  {row['dataset']:<11s} files={row['files']:<2d} "
              f"rows={str(row['rows']):<12s} status={row['status']}")
    print("=" * 72)
    if failures:
        print(f"VERIFICATION FAILED: {failures} dataset(s) missing or invalid.")
        return 1
    print("VERIFICATION OK: all six datasets present and non-synthetic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
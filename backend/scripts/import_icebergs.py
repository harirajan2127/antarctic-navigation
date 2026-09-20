#!/usr/bin/env python3
"""Import and validate iceberg tracking data.

Supports CSV, NetCDF, and GeoJSON sources. Normalizes to:
    iceberg_id, timestamp, latitude, longitude, length_nm, width_nm

Usage:
    python scripts/import_icebergs.py path/to/icebergs.csv [--tag NAME]
    python scripts/import_icebergs.py path/to/tracks.nc
    python scripts/import_icebergs.py path/to/features.geojson
    python scripts/import_icebergs.py --demo            # generate labeled demo data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.cli import run_main
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.icebergs.import_data import generate_demo, import_file

log = get_logger(__name__)


def main(args: argparse.Namespace) -> int:
    settings = get_pipeline_settings()

    if args.demo:
        csv_path, nc_path = generate_demo(settings.DATA_PROCESSED_ROOT)
        print(f"\nGenerated demo icebergs:\n  {csv_path}\n  {nc_path}")
        return 0

    if not args.source:
        print("Provide a source file path, or use --demo.")
        return 2

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"Source file not found: {source}")
        return 2

    result = import_file(source, settings.DATA_PROCESSED_ROOT, tag=args.tag)
    print(f"\nIceberg import: {result.status.upper()} ({source.name})")
    print(f"  records        : {result.meta.get('records', '-')}")
    print(f"  unique icebergs: {result.meta.get('unique_icebergs', '-')}")
    print(f"  time window    : {result.meta.get('time_start', '-')} -> {result.meta.get('time_end', '-')}")
    for c in result.checks:
        print(f"  [{c.severity}] {c.message}")
    return 0 if result.status != "invalid" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import iceberg tracking data (CSV/NetCDF/GeoJSON).")
    parser.add_argument("source", nargs="?", type=str, help="Path to the source file")
    parser.add_argument("--tag", type=str, default=None, help="Name for the processed output files")
    parser.add_argument("--demo", action="store_true", help="Generate labeled synthetic demo icebergs")
    args = parser.parse_args()
    sys.exit(run_main(lambda: main(args)))
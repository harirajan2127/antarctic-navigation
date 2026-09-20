#!/usr/bin/env python3
"""Run the full Antarctic dataset preprocessing pipeline.

Consumes the pipeline-processed datasets under ``backend/datasets/processed/``
and writes clean ML-ready outputs + all reports under ``backend/data/``.

Usage:
    python data/run_preprocessing.py                       # defaults to pipeline outputs
    python data/run_preprocessing.py --sea-ice demo/sea_ice.nc --ocean ...
    python data/run_preprocessing.py --tolerance-hours 12 --daily
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger

from data.processing.sea_ice_processor import SeaIceProcessor
from data.processing.ocean_processor import OceanProcessor
from data.processing.weather_processor import WeatherProcessor
from data.processing.iceberg_processor import IcebergProcessor
from data.features.feature_engineering import FeatureEngineering

log = get_logger(__name__)

DATA_ROOT = Path(__file__).resolve().parent


def _default_paths() -> dict[str, Path]:
    settings = get_pipeline_settings()
    iceberg_csv = settings.DATA_PROCESSED_ROOT / "icebergs.csv"
    if not iceberg_csv.exists():
        iceberg_csv = settings.DATA_PROCESSED_ROOT / "icebergs_demo.csv"
        log.info(
            "No real iceberg import found (icebergs.csv) — falling back to %s",
            iceberg_csv.name,
        )
    return {
        "sea_ice": settings.DATA_PROCESSED_ROOT / "sea_ice.nc",
        "ocean": settings.DATA_PROCESSED_ROOT / "ocean_surface.nc",
        "weather": settings.DATA_PROCESSED_ROOT / "weather_surface.nc",
        "icebergs": iceberg_csv,
    }


def main(args: argparse.Namespace) -> int:
    inputs = _default_paths()
    if args.raw_root:
        inputs = {k: Path(args.raw_root) / v.name for k, v in inputs.items()}
    if args.sea_ice:
        inputs["sea_ice"] = Path(args.sea_ice)
    if args.ocean:
        inputs["ocean"] = Path(args.ocean)
    if args.weather:
        inputs["weather"] = Path(args.weather)
    if args.icebergs:
        inputs["icebergs"] = Path(args.icebergs)

    out = DATA_ROOT / "processed"
    reports = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Antarctic pre-processing pipeline")
    print("=" * 70)
    print(f"Input  : {[str(v) for v in inputs.values()]}")

    # 1・Sea ice
    if inputs["sea_ice"].exists():
        SeaIceProcessor(
            inputs["sea_ice"],
            out / "sea_ice" / "sea_ice_clean.nc",
            reports,
            resample_daily=not args.keep_available,
        ).process()
        print("  [1/5] sea-ice            -> ", out / "sea_ice" / "sea_ice_clean.nc")
    else:
        print("  [1/5] SKIP  sea-ice (missing file)")

    # 2・Ocean
    if inputs["ocean"].exists():
        OceanProcessor(
            inputs["ocean"],
            out / "ocean" / "ocean_clean.nc",
            reports,
            ref_grid_path=out / "sea_ice" / "sea_ice_clean.nc"
            if (out / "sea_ice" / "sea_ice_clean.nc").exists() else None,
            resample_daily=not args.keep_available,
        ).process()
        print("  [2/5] ocean             -> ", out / "ocean" / "ocean_clean.nc")
    else:
        print("  [2/5] SKIP  ocean (missing file)")

    # 3・Weather
    if inputs["weather"].exists():
        WeatherProcessor(
            inputs["weather"],
            out / "weather" / "weather_clean.nc",
            reports,
            resample_daily=not args.keep_available,
        ).process()
        print("  [3/5] weather           -> ", out / "weather" / "weather_clean.nc")
    else:
        print("  [3/5] SKIP  weather (missing file)")

    # 4・Icebergs
    if inputs["icebergs"].exists():
        IcebergProcessor(
            inputs["icebergs"],
            out / "icebergs" / "icebergs_clean.csv",
            out / "icebergs" / "icebergs_clean.nc",
            reports,
        ).process()
        print("  [4/5] icebergs          -> ", out / "icebergs" / "icebergs_clean.csv")
    else:
        print("  [4/5] SKIP  icebergs (missing file)")

    # 5・Feature engineering
    need = ["sea_ice", "ocean", "weather"]
    clean_paths = {
        d: out / d / {"sea_ice": "sea_ice_clean.nc", "ocean": "ocean_clean.nc", "weather": "weather_clean.nc"}[d]
        for d in need
    }
    icebergs_clean = out / "icebergs" / "icebergs_clean.csv"
    if icebergs_clean.exists() and any(p.exists() for p in clean_paths.values()):
        FeatureEngineering(
            icebergs_clean,
            clean_paths["sea_ice"],
            clean_paths["ocean"],
            clean_paths["weather"],
            out / "features",
            reports,
            tolerance_hours=args.tolerance_hours,
        ).build()
        print("  [5/5] features          -> ", out / "features" / "feature_table.csv")
    else:
        print("  [5/5] SKIP  feature engineering (missing inputs)")

    print("-" * 70)
    print(f"Reports : {reports}")
    print(f"Outputs : {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Antarctic datasets into ML-ready features.")
    parser.add_argument("--sea-ice", type=str, help="Input sea-ice NetCDF")
    parser.add_argument("--ocean", type=str, help="Input ocean NetCDF")
    parser.add_argument("--weather", type=str, help="Input weather NetCDF")
    parser.add_argument("--icebergs", type=str, help="Input iceberg CSV/NetCDF/GeoJSON")
    parser.add_argument("--raw-root", type=str, help="Directory containing all four pipeline-processed inputs")
    parser.add_argument("--tolerance-hours", type=float, default=24.0, help="Max lag hours before env features are marked stale (default 24)")
    parser.add_argument("--keep-available", action="store_true", help="Do NOT resample to daily; keep available time steps")
    args = parser.parse_args()
    sys.exit(main(args))
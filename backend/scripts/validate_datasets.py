#!/usr/bin/env python3
"""Validate all datasets and generate the dataset report (JSON + Markdown).

Checks per dataset:
  - presence and completeness of files
  - coordinate ranges (Antarctic bounds)
  - date ranges and cadence
  - units attributes
  - missing values
  - data freshness (age of newest observation, true cadence)

Usage:
    python scripts/validate_datasets.py [--out NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.cli import run_main
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.report import DatasetReport
from data_pipeline.common.validation import summarize_result
from data_pipeline.ocean.acquire import validate as validate_ocean
from data_pipeline.processor.accessor import EnvironmentalAccessor
from data_pipeline.sea_ice.acquire import validate as validate_sea_ice
from data_pipeline.weather.acquire import validate as validate_weather
from data_pipeline.icebergs.import_data import validate_processed as validate_icebergs

log = get_logger(__name__)


def main(args: argparse.Namespace) -> int:
    settings = get_pipeline_settings()
    report = DatasetReport()
    accessor = EnvironmentalAccessor(settings.DATA_PROCESSED_ROOT)

    validators = {
        "sea_ice": validate_sea_ice,
        "ocean": validate_ocean,
        "weather": validate_weather,
        "icebergs": validate_icebergs,
    }

    for name, validator in validators.items():
        result = validator(settings.DATA_PROCESSED_ROOT)
        report.add_result(result)
        print(summarize_result(result))
        for c in result.checks:
            print(f"    [{c.severity}] {c.message}")

    # Freshness for the two-hour rolling update support.
    for entry in accessor.freshness_report():
        report.add_freshness(entry.pop("dataset"), entry)

    json_path, md_path = report.write(name=args.out)
    print(f"\nReport written:\n  JSON      : {json_path}\n  Markdown  : {md_path}")
    return 0 if report.overall_status() in ("valid", "incomplete") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate datasets and write the dataset report.")
    parser.add_argument("--out", type=str, default="dataset_report", help="Report base name")
    args = parser.parse_args()
    sys.exit(run_main(lambda: main(args)))
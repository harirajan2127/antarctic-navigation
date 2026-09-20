#!/usr/bin/env python3
"""Create labeled synthetic demo datasets for development and testing.

Usage:
    python scripts/create_demo_data.py [--days 30]

Every generated file carries ``classification: synthetic_demo`` and prints a
clear notice. Demo data is never labeled as real.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.cli import run_main
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.demo.create import create_all_demo_data

log = get_logger(__name__)


def main(args: argparse.Namespace) -> int:
    outputs = create_all_demo_data(days=args.days)
    print("\nDemo datasets generated (ALL SYNTHETIC, NOT REAL DATA):\n")
    for dataset, paths in outputs.items():
        for p in paths:
            print(f"  {dataset}: {p}")
    print(
        "\nDISCLAIMER: These files simulate observations for development only. "
        "They must never be used for real navigation decisions."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic demo datasets.")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to generate")
    args = parser.parse_args()
    sys.exit(run_main(lambda: main(args)))
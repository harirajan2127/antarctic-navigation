#!/usr/bin/env python3
"""Download Antarctic oceanographic data from Copernicus Marine.

Variables: uo (current U), vo (current V), thetao (temperature), so (salinity).

Usage:
    python scripts/download_ocean.py [--mode auto|real|demo] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Requires COPERNICUS_MOTU_USER and COPERNICUS_MOTU_PASS for live data;
falls back to labeled synthetic demo data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.cli import add_common_download_args, run_main, setup_verbose_logging
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.ocean.acquire import ensure_ocean

log = get_logger(__name__)


def main(args: argparse.Namespace) -> int:
    setup_verbose_logging(args.verbose)
    out = ensure_ocean(start=args.start, end=args.end, mode=args.mode)
    print(f"\nOcean dataset ready: {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Copernicus Marine oceanographic data.")
    add_common_download_args(parser)
    args = parser.parse_args()
    sys.exit(run_main(lambda: main(args)))
#!/usr/bin/env python3
"""Download Antarctic sea-ice concentration data.

Usage:
    python scripts/download_sea_ice.py [--mode auto|real|demo] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Sources:
  - Copernicus Marine SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS (needs COPERNICUS_MOTU_USER/PASS).
  - NSIDC Sea Ice Index (needs NSIDC_API_TOKEN and NSIDC_SEA_ICE_URL_PATTERN).
  - Synthetic labeled demo data otherwise.

Requires a ``.env`` file (copied from ``.env.example``) or environment variables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.cli import add_common_download_args, run_main, setup_verbose_logging
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.sea_ice.acquire import ensure_sea_ice

log = get_logger(__name__)


def main(args: argparse.Namespace) -> int:
    setup_verbose_logging(args.verbose)
    out = ensure_sea_ice(start=args.start, end=args.end, mode=args.mode)
    print(f"\nSea-ice dataset ready: {out}")
    print(f"  classification : {_classification(out)}")
    return 0


def _classification(path: Path) -> str:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return ds.attrs.get("classification", "unknown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Antarctic sea-ice concentration data.")
    add_common_download_args(parser)
    args = parser.parse_args()
    sys.exit(run_main(lambda: main(args)))
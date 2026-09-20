"""Shared CLI helpers for pipeline scripts."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Callable

from data_pipeline.common.config import creds_summary, get_pipeline_settings


def add_common_download_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the download scripts."""
    parser.add_argument(
        "--mode",
        choices=["auto", "real", "demo"],
        default="auto",
        help=(
            "auto: use live provider when credentials are configured, else demo; "
            "real: require live provider; demo: force synthetic labeled data"
        ),
    )
    parser.add_argument("--start", type=date_iso, default=None, help="YYYY-MM-DD start date")
    parser.add_argument("--end", type=date_iso, default=None, help="YYYY-MM-DD end date")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")


def date_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def setup_verbose_logging(verbose: bool) -> None:
    if verbose:
        logging.getLogger("data_pipeline").setLevel(logging.DEBUG)


def print_credentials() -> None:
    summary = creds_summary()
    print("Credential configuration:")
    for k, v in summary.items():
        print(f"  {k}: {'configured' if v else 'not configured'}")


def run_main(main: Callable[[], int]) -> int:
    get_pipeline_settings().ensure_dirs()
    try:
        return main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}")
        return 1
"""Demo-data generation orchestrator.

Creates clearly-labeled synthetic datasets for development and testing when
real provider data is not available. Every artifact is tagged
``classification: synthetic_demo``; nothing is ever labeled as real.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.ocean.acquire import _generate_demo_ocean
from data_pipeline.sea_ice.acquire import _generate_demo_sea_ice
from data_pipeline.weather.acquire import _generate_demo_weather
from data_pipeline.icebergs.import_data import generate_demo as generate_demo_icebergs

log = get_logger(__name__)


def create_all_demo_data(
    processed_root: Path | None = None,
    *,
    days: int = 30,
) -> dict[str, list[Path]]:
    """Generate the complete set of labeled demo datasets.

    Returns a dict of dataset name -> list of output paths.
    """
    settings = get_pipeline_settings()
    processed_root = Path(processed_root or settings.DATA_PROCESSED_ROOT)
    processed_root.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    outputs: dict[str, list[Path]] = {
        "sea_ice": [_generate_demo_sea_ice(start, end, processed_root / "sea_ice.nc")],
        "ocean": [_generate_demo_ocean(start, end, processed_root / "ocean_surface.nc")],
        "weather": [_generate_demo_weather(start, end, processed_root / "weather_surface.nc")],
        "icebergs": list(generate_demo_icebergs(processed_root, n_icebergs=40, days=days)),
    }

    log.info("Demo data generation complete under %s", processed_root)
    return outputs
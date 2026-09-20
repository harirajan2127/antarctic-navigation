"""ERA5 weather dataset acquisition and validation.

Real source: Copernicus Climate Data Store (CDS) ERA5 single levels,
variables u10 (10m east wind), v10 (10m north wind), t2m (2m temperature),
msl (mean sea level pressure). Requires a CDS API key. Falls back to
labeled synthetic demo data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from data_pipeline.common import netcdf_utils
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.downloader import DownloadError
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import WEATHER_SPEC
from data_pipeline.common.validation import (
    ValidationResult,
    numeric_range_check,
    validate_missing_values,
)

log = get_logger(__name__)

WEATHER_VARIABLES = ("u10", "v10", "t2m", "msl")


def _era5(start: datetime, end: datetime, out_file: Path) -> Path:
    """Request ERA5 from the CDS using ``cdsapi``."""
    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise DownloadError(
            "cdsapi is required for ERA5 downloads. Install with: pip install cdsapi"
        ) from exc

    settings = get_pipeline_settings()
    if ":" not in settings.CDS_API_KEY:
        raise DownloadError(
            "CDS_API_KEY must be formatted as '<uid>:<api-key>' (see .env.example)."
        )

    client = cdsapi.Client(url=settings.CDS_API_URL, key=settings.CDS_API_KEY)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(WEATHER_VARIABLES),
        "year": sorted({d.year for d in _daterange_excl_end(start, end)}),
        "month": sorted({d.month for d in _daterange_excl_end(start, end)}),
        "day": sorted({d.day for d in _daterange_excl_end(start, end)}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [
            settings.ANTARCTIC_LAT_MAX,
            settings.ANTARCTIC_LON_MIN,
            settings.ANTARCTIC_LAT_MIN,
            settings.ANTARCTIC_LON_MAX,
        ],
    }
    log.info("Requesting ERA5 via CDS (area %s)...", request["area"])
    # The CDS API writes the file; time out behaviour is internal to cdsapi
    # with the configured retries in cdsapiClient.
    client.retrieve("reanalysis-era5-single-levels", request, str(out_file))
    return out_file


def ensure_weather(
    start: datetime | None = None,
    end: datetime | None = None,
    mode: str = "auto",
) -> Path:
    """Return a processed ERA5-dataset NetCDF file (real or labeled demo)."""
    settings = get_pipeline_settings()
    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=7))

    processed = settings.DATA_PROCESSED_ROOT / "weather_surface.nc"
    if processed.exists() and settings.PIPELINE_SKIP_EXISTING:
        log.info("Weather processed file already exists: %s", processed)
        return processed

    if mode == "demo" or (mode == "auto" and not settings.has_cds_credentials):
        log.info("Weather: no live CDS source configured -> generating labeled demo data.")
        return _generate_demo_weather(start, end, processed)

    if not settings.has_cds_credentials:
        raise DownloadError("No CDS/ERA5 credentials configured; use --mode demo.")

    raw = settings.DATA_RAW_ROOT / "era5.nc"
    try:
        _era5(start, end, raw)
        return _normalize_era5(raw, processed)
    except Exception as exc:
        log.error("ERA5 download failed: %s", exc)
        if mode == "real":
            raise
        return _generate_demo_weather(start, end, processed)


def _normalize_era5(raw: Path, processed: Path) -> Path:
    ds = xr.open_dataset(raw)
    ds = ds.rename(
        {name: {"latitude": "latitude", "longitude": "longitude"}.get(name, name) for name in ds.variables}
    ).sortby("time")
    ds.attrs.update(
        netcdf_utils.stamp_attrs(
            {
                "classification": "real_observation",
                "source_name": "ecmwf-era5",
                "product": "reanalysis-era5-single-levels",
            }
        )
    )
    write_netcdf(ds, processed)
    return processed


def _generate_demo_weather(start: datetime, end: datetime, out_file: Path) -> Path:
    if out_file.exists():
        return out_file

    lats = np.arange(-75.0, -50.0, 0.5)
    lons = np.arange(-179.5, 180.0, 0.5)
    # 6-hourly demo cadence (clearly not hourly; never claimed as such).
    times = np.arange(
        np.datetime64(start.date()),
        np.datetime64(end.date()) + np.timedelta64(1, "D"),
        np.timedelta64(6, "h"),
    )
    glat, glon = np.meshgrid(lats, lons, indexing="ij")

    rng = np.random.RandomState(777)
    u10 = 6.0 * rng.standard_normal((len(times), len(lats), len(lons))).astype("float32")
    v10 = 6.0 * rng.standard_normal((len(times), len(lats), len(lons))).astype("float32")
    t2m = (
        np.clip(273.15 - 0.6 * (glat + 50.0), 230.0, 285.0)[None, :, :]
        + 2.0 * rng.standard_normal((len(times), len(lats), len(lons))).astype("float32")
    )
    msl = (
        99000.0
        + 600.0 * rng.standard_normal((len(times), len(lats), len(lons))).astype("float32")
    )

    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u10),
            "v10": (("time", "latitude", "longitude"), v10),
            "t2m": (("time", "latitude", "longitude"), t2m),
            "msl": (("time", "latitude", "longitude"), msl),
        },
        coords={
            "time": times.astype("datetime64[ns]"),
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
        },
        attrs={
            **netcdf_utils.stamp_attrs(netcdf_utils.DEMO_GLOBAL_ATTRS),
            "title": WEATHER_SPEC.title,
            "cadence_hours": str(6),
        },
    )
    ds["u10"].attrs.update(units="m s-1", long_name="10m east wind (synthetic demo)")
    ds["v10"].attrs.update(units="m s-1", long_name="10m north wind (synthetic demo)")
    ds["t2m"].attrs.update(units="K", long_name="2m temperature (synthetic demo)")
    ds["msl"].attrs.update(units="Pa", long_name="Mean sea-level pressure (synthetic demo)")
    return write_netcdf(ds, out_file)


def validate(processed_root: Path) -> ValidationResult:
    """Validate the processed weather dataset."""
    settings = get_pipeline_settings()
    path = Path(processed_root) / "weather_surface.nc"
    result = ValidationResult(dataset="weather", file=path.name, status="missing")
    if not path.exists():
        result.add("error", "FILE_MISSING", "Processed weather NetCDF not found")
        return result

    ds = xr.open_dataset(path)
    result.meta = {
        "classification": ds.attrs.get("classification", "unknown"),
        "source": ds.attrs.get("source_name", "unknown"),
        "cadence_hours": ds.attrs.get("cadence_hours"),
    }
    result.status = "valid"
    if ds.attrs.get("classification") == "synthetic_demo":
        result.add("warning", "DEMO_DATA", "Synthetic demo data; not real measurements.")

    for coord in ("time", "latitude", "longitude"):
        if coord not in ds.coords:
            result.status = "invalid"
            result.add("error", "COORD_MISSING", f"Missing coordinate {coord}")

    checks = (
        ("u10", -80.0, 80.0),
        ("v10", -80.0, 80.0),
        ("t2m", 150.0, 330.0),
        ("msl", 90000.0, 105000.0),
    )
    for var, lo, hi in checks:
        if var not in ds:
            result.status = "invalid"
            result.add("error", "VARIABLE_MISSING", f"Missing variable {var}")
            continue
        issues = [
            numeric_range_check(var, ds[var].values, lo, hi),
            validate_missing_values(var, ds[var].values),
        ]
        for iss in issues:
            if iss and iss.severity == "error":
                result.status = "invalid"
            if iss:
                result.add(iss.severity, iss.code, iss.message, iss.detail)

    ds.close()
    return result


def _daterange_excl_end(start: datetime, end: datetime):
    """Yield each day in [start, end) for building CDS requests."""
    current = start.date()
    while current < end.date():
        yield current
        current += timedelta(days=1)
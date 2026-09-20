"""Sea-ice dataset acquisition and validation.

Real sources (when credentials are configured):
- Copernicus Marine SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS via MOTU.
- NSIDC Sea Ice Index (nsidc0051) via HTTPS with a bearer token.

Both are abstracted behind :func:`ensure_sea_ice` which falls back to a
clearly-labeled synthetic product when real sources are unavailable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

import xarray as xr

from data_pipeline.common import netcdf_utils
from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.downloader import DownloadError, ExistingFileSkipped, RobustDownloader
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import SEA_ICE_SPEC
from data_pipeline.common.validation import ValidationResult, numeric_range_check, validate_missing_values

log = get_logger(__name__)


def _copernicus_sea_ice(
    start: datetime,
    end: datetime,
    out_file: Path,
) -> Path:
    """Download sea-ice concentration from Copernicus Marine via MOTU."""
    try:
        import motuclient  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DownloadError(
            "motuclient is required for Copernicus downloads. "
            "Install with: pip install motuclient"
        ) from exc

    from motuclient import motu_api

    settings = get_pipeline_settings()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-u", settings.COPERNICUS_MOTU_USER,
        "-p", settings.COPERNICUS_MOTU_PASS,
        "-m", settings.COPERNICUS_MOTU_URL,
        "-d", settings.COPERNICUS_SEAICE_DATASET_ID,
        "-x", str(settings.ANTARCTIC_LON_MIN),
        "-X", str(settings.ANTARCTIC_LON_MAX),
        "-y", str(settings.ANTARCTIC_LAT_MIN),
        "-Y", str(settings.ANTARCTIC_LAT_MAX),
        "-t", start.strftime("%Y-%m-%d %H:00:00"),
        "-T", end.strftime("%Y-%m-%d %H:00:00"),
        "-v", "sea_ice_concentration",
        "-o", str(out_file.parent),
        "-f", out_file.name,
    ]
    log.info("Requesting Copernicus Marine sea-ice product via MOTU...")
    motu_api.execute(args)
    return out_file


def _nsidc_sea_ice(
    start: datetime,
    end: datetime,
    out_file: Path,
) -> Path:
    """Download NSIDC sea-ice concentration for the requested date range.

    NSIDC serves daily gridded sea-ice products (e.g. Sea Ice Index G02202,
    GSFC NRT) over HTTPS with a bearer token. The exact remote file name
    varies by product, so the operator supplies a URL pattern with a
    ``{date}`` placeholder via the ``NSIDC_SEA_ICE_URL_PATTERN`` variable:

      https://daacdata.apps.nsidc.org/pub/DATASETS/<product>/south/nt_{date}_f17_v02r00.nc

    The downloaded file is normalized to the standard schema (time,
    latitude, longitude, sea_ice_concentration) by ``_normalize_nsidc``.
    """
    import os

    import requests

    settings = get_pipeline_settings()
    pattern = settings.NSIDC_SEA_ICE_URL_PATTERN or os.environ.get("NSIDC_SEA_ICE_URL_PATTERN", "")
    if not pattern:
        raise DownloadError(
            "NSIDC download requires NSIDC_SEA_ICE_URL_PATTERN with a {date} "
            "placeholder (product-specific remote path). Alternatively configure "
            "Copernicus Marine credentials, or use --mode demo."
        )
    if "{date}" not in pattern:
        raise DownloadError("NSIDC_SEA_ICE_URL_PATTERN must contain {date}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    downloader = RobustDownloader()
    current = start
    raws: list[Path] = []
    while current <= end:
        url = pattern.format(date=current.strftime("%Y%m%d"))
        raw_day = settings.DATA_RAW_ROOT / f"sea_ice_nsidc_{current.strftime('%Y%m%d')}.nc"
        try:
            downloader.download(
                url,
                raw_day,
                extra_headers={"Authorization": f"Bearer {settings.NSIDC_API_TOKEN}"},
                progress_label=f"nsidc {current:%Y-%m-%d}",
            )
            raws.append(raw_day)
        except ExistingFileSkipped:
            raws.append(raw_day)
        except DownloadError as exc:
            log.error("NSIDC day %s failed: %s", current.date(), exc)
        current += timedelta(days=1)

    if not raws:
        raise DownloadError("No NSIDC sea-ice files could be downloaded for the date range.")

    _normalize_nsidc(sorted(raws), out_file)
    return out_file


def _normalize_nsidc(raw_files: list[Path], processed: Path) -> Path:
    """Concatenate daily NSIDC files into one processed dataset."""
    ds_parts = []
    for path in raw_files:
        day = xr.open_dataset(path)
        day = day.rename(
            {
                name: {"lat": "latitude", "lon": "longitude"}.get(name, name)
                for name in day.variables
            }
        )
        if "sic" in day:
            day = day.rename({"sic": "sea_ice_concentration"})
        ds_parts.append(day)

    ds = xr.concat(ds_parts, dim="time")
    ds.attrs.update(
        netcdf_utils.stamp_attrs(
            {
                "classification": "real_observation",
                "source_name": "nsidc",
                "product": "NSIDC daily sea-ice (operator-configured product)",
            }
        )
    )
    ds["sea_ice_concentration"] = ds["sea_ice_concentration"].clip(0.0, 1.0)
    ds["sea_ice_concentration"].attrs["units"] = "fraction"
    write_netcdf(ds, processed)
    for p in ds_parts:
        try:
            p.close()
        except Exception:
            pass
    return processed


def _existing_classification(path: Path) -> str:
    try:
        with xr.open_dataset(path) as ds:
            return str(ds.attrs.get("classification", "unknown"))
    except Exception:
        return "unknown"


def ensure_sea_ice(
    start: datetime | None = None,
    end: datetime | None = None,
    mode: str = "auto",
    force: bool = False,
) -> Path:
    """Return a processed sea-ice NetCDF file, real or labeled demo.

    ``mode``: "auto" (real if credentials exist, else demo), "real", "demo".
    ``force``: re-acquire even if a valid processed file exists. By default
    an existing real file is reused, but an existing synthetic file is never
    silently kept when a live source is available or ``--mode real`` is used.
    """
    settings = get_pipeline_settings()
    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=30))

    processed = settings.DATA_PROCESSED_ROOT / "sea_ice.nc"
    use_real = settings.has_copernicus_credentials or settings.has_nsidc_credentials

    if processed.exists() and settings.PIPELINE_SKIP_EXISTING and not force:
        classification = _existing_classification(processed)
        wants_real = mode == "real" or (mode == "auto" and use_real)
        if not (wants_real and classification == "synthetic_demo"):
            log.info("Sea-ice processed file already exists: %s", processed)
            return processed
        log.warning(
            "Existing sea-ice file is synthetic demo but a live source is "
            "available; replacing it with real data (mode=%s).",
            mode,
        )

    if mode == "demo" or (mode == "auto" and not use_real):
        log.info("Sea-ice: no live source configured -> generating labeled demo data.")
        demo = _generate_demo_sea_ice(start, end, processed)
        return demo

    if not use_real:
        raise DownloadError("No sea-ice credentials configured; use --mode demo.")

    if settings.has_copernicus_credentials:
        raw = settings.DATA_RAW_ROOT / "sea_ice_copernicus.nc"
        try:
            _copernicus_sea_ice(start, end, raw)
            _normalize_copernicus_sea_ice(raw, processed)
            return processed
        except Exception as exc:
            log.error("Copernicus sea-ice download failed: %s", exc)
            if mode == "real":
                raise
    # NSIDC path is documented as a stub requiring the provider manifest.
    raise DownloadError(
        "NSIDC download requires the provider's file manifest; configure "
        "Copernicus Marine credentials for a live source, or use --mode demo."
    )


def _normalize_copernicus_sea_ice(raw: Path, processed: Path) -> Path:
    ds = xr.open_dataset(raw)
    ds = ds.rename({"latitude": "latitude", "longitude": "longitude"})
    if "sic" in ds:
        ds = ds.rename({"sic": "sea_ice_concentration"})
    ds.attrs.update(
        netcdf_utils.stamp_attrs(
            {
                "classification": "real_observation",
                "source_name": "copernicus-marine",
                "product": get_pipeline_settings().COPERNICUS_SEAICE_DATASET_ID,
            }
        )
    )
    ds["sea_ice_concentration"].attrs["units"] = "fraction"
    write_netcdf(ds, processed)
    return processed


def _generate_demo_sea_ice(start: datetime, end: datetime, out_file: Path) -> Path:
    """Create labeled synthetic sea-ice fields (daily, coarse grid)."""
    if out_file.exists():
        log.info(
            "Sea-ice output already exists (%s); reusing it (classification "
            "is read from the file itself).",
            out_file.name,
        )
        return out_file

    settings = get_pipeline_settings()
    lats = np.arange(-89.0, -50.0, 1.0)
    lons = np.arange(-179.0, 180.0, 1.0)
    times = np.arange(
        np.datetime64(start.date()),
        np.datetime64(end.date()) + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )

    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    conc = np.zeros((len(times), len(lats), len(lons)), dtype="float32")
    for ti, t in enumerate(times):
        day_seed = int(t.astype("datetime64[D]").astype("int64")) % 1000
        rng = np.random.RandomState(day_seed)
        # A smooth ice margin plus daily noise.
        margin = np.full_like(glat, -64.0) + 0.3 * np.cos(np.deg2rad(glon + day_seed))
        ice = glat < margin
        conc[ti] = np.where(ice, 0.9 + 0.1 * rng.random_sample(glat.shape), 0.0)
        conc[ti] = np.clip(np.where(glat < -85.0, 1.0, conc[ti]), 0.0, 1.0)

    ds = xr.Dataset(
        {
            "sea_ice_concentration": (("time", "latitude", "longitude"), conc),
        },
        coords={
            "time": times.astype("datetime64[ns]"),
            "latitude": (("latitude",), lats),
            "longitude": (("longitude",), lons),
        },
        attrs={
            **netcdf_utils.stamp_attrs(netcdf_utils.DEMO_GLOBAL_ATTRS),
            "title": SEA_ICE_SPEC.title,
            "cadence_hours": str(SEA_ICE_SPEC.cadence_hours_default),
        },
    )
    ds["sea_ice_concentration"].attrs.update(
        units="fraction", long_name="Sea-ice concentration (synthetic demo)"
    )
    return write_netcdf(ds, out_file)


def validate(processed_root: Path) -> ValidationResult:
    """Validate the processed sea-ice dataset."""
    settings = get_pipeline_settings()
    path = Path(processed_root) / "sea_ice.nc"
    result = ValidationResult(dataset="sea_ice", file=path.name, status="missing")
    if not path.exists():
        result.add("error", "FILE_MISSING", "Processed sea-ice NetCDF not found")
        return result

    ds = xr.open_dataset(path)
    result.meta = {
        "classification": ds.attrs.get("classification", "unknown"),
        "source": ds.attrs.get("source_name", "unknown"),
        "cadence_hours": ds.attrs.get("cadence_hours"),
    }
    result.status = "valid"

    if ds.attrs.get("classification") == "synthetic_demo":
        result.add(
            "warning",
            "DEMO_DATA",
            "File contains synthetic demo data and must not be used operationally.",
        )

    for coord in ("time", "latitude", "longitude"):
        if coord not in ds.coords and coord not in ds.variables:
            result.status = "invalid"
            result.add("error", "COORD_MISSING", f"Missing coordinate {coord}")
            continue

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    if lat.size and lon.size:
        if lat.min() < -90 or lat.max() > -50:
            result.status = "invalid"
            result.add("error", "REGION", "Latitudes outside Antarctic region [-90,-50]")
        if lon.min() < -180 or lon.max() > 180:
            result.status = "invalid"
            result.add("error", "REGION", "Longitudes outside [-180,180]")

    if "sea_ice_concentration" in ds:
        v = ds["sea_ice_concentration"]
        issues = [
            numeric_range_check("sea_ice_concentration", v.values, 0.0, 1.0),
            validate_missing_values("sea_ice_concentration", v.values),
        ]
        for iss in issues:
            if iss and iss.severity == "error":
                result.status = "invalid"
            if iss:
                result.add(iss.severity, iss.code, iss.message, iss.detail)
        if v.attrs.get("units") != "fraction":
            result.status = "invalid"
            result.add("error", "UNITS", f"Expected units 'fraction', got {v.attrs.get('units')}")

    time_vals = ds["time"].values
    if time_vals.size:
        # Validate time ordering and cadence.
        diffs = np.diff(time_vals.astype("datetime64[h]").astype("int64"))
        if (diffs <= 0).any():
            result.status = "invalid"
            result.add("error", "TIME_ORDER", "Time coordinate is not strictly increasing")
        elif result.meta.get("cadence_hours"):
            result.meta["actual_cadence_hours"] = float(np.median(diffs[diffs > 0]))

    ds.close()
    return result
"""Iceberg tracking import pipeline.

Reads iceberg observations from CSV, NetCDF, or GeoJSON and normalizes them
to the standard schema:

    iceberg_id, timestamp, latitude, longitude, length_nm, width_nm

Also validates required fields, coordinate ranges, and time ordering, then
writes the normalized table to the processed store (CSV + NetCDF).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.common.config import get_pipeline_settings
from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.common.schemas import ICEBERG_SPEC
from data_pipeline.common.validation import ValidationResult

log = get_logger(__name__)

REQUIRED_COLUMNS = ["iceberg_id", "timestamp", "latitude", "longitude", "length_nm", "width_nm"]
COLUMN_ALIASES = {
    "id": "iceberg_id",
    "iceberg": "iceberg_id",
    "time": "timestamp",
    "date": "timestamp",
    "lat": "latitude",
    "lat_dd": "latitude",
    "lon": "longitude",
    "lon_dd": "longitude",
    "length": "length_nm",
    "width": "width_nm",
}


class IcebergImportError(Exception):
    """Raised when an iceberg source file cannot be imported."""


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".nc", ".nc4", ".cdf"):
        return "netcdf"
    if suffix == ".geojson":
        return "geojson"
    if suffix == ".json":
        # Peek: FeatureCollection => geojson.
        with open(path, "r", encoding="utf-8") as fh:
            head = fh.read(4096)
        if '"type"' in head and ('"FeatureCollection"' in head or '"Feature"' in head):
            return "geojson"
        return "json"
    raise IcebergImportError(f"Unsupported iceberg file format: {path.suffix}")


def normalize_to_pandas(source: Path | pd.DataFrame) -> pd.DataFrame:
    """Read any supported source and return a normalized DataFrame.

    Native ``*_km`` columns are converted to nautical miles; daily-alt
    aliases (``id``/``time``/``lat``/``lon``) are normalized to the standard
    schema.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        fmt = detect_format(Path(source))
        if fmt == "csv":
            df = pd.read_csv(source)
        elif fmt == "netcdf":
            df = _netcdf_to_dataframe(Path(source))
        elif fmt == "geojson":
            df = _geojson_to_dataframe(Path(source))
        else:
            raise IcebergImportError("Unsupported format: JSON without FeatureCollection type")

    # Unit conversion *before* renaming so aliased km columns become nm.
    km_to_nm = 0.539957
    if "length_km" in df.columns:
        df["length_nm"] = pd.to_numeric(df["length_km"], errors="coerce") * km_to_nm
        df = df.drop(columns=["length_km"])
    if "width_km" in df.columns:
        df["width_nm"] = pd.to_numeric(df["width_km"], errors="coerce") * km_to_nm
        df = df.drop(columns=["width_km"])

    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IcebergImportError(f"Source missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _netcdf_to_dataframe(path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    times = ds["time"].values.astype("datetime64[s]").astype(object)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ids = ds.get("iceberg_id", ds.get("id", np.arange(len(lats)))).values
    length = ds["length_nm"].values if "length_nm" in ds else np.full_like(lats, np.nan)
    width = ds["width_nm"].values if "width_nm" in ds else np.full_like(lats, np.nan)
    df = pd.DataFrame(
        {
            "iceberg_id": np.broadcast_to(ids, np.asarray(lats).shape),
            "timestamp": np.broadcast_to(times, np.asarray(lats).shape),
            "latitude": lats,
            "longitude": lons,
            "length_nm": length,
            "width_nm": width,
        }
    )
    ds.close()
    return df


def _geojson_to_dataframe(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        geojson = json.load(fh)

    rows: list[dict[str, Any]] = []
    features = geojson.get("features", [])
    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        if not coords or coords[0] is None:
            continue
        rows.append(
            {
                "iceberg_id": props.get("iceberg_id", props.get("id", "unknown")),
                "timestamp": props.get(
                    "timestamp",
                    props.get("time", props.get("observation_time", datetime.now(timezone.utc).isoformat())),
                ),
                "latitude": coords[1],
                "longitude": coords[0],
                "length_nm": props.get("length_nm", _km_to_nm(props.get("length_km", np.nan))),
                "width_nm": props.get("width_nm", _km_to_nm(props.get("width_km", np.nan))),
            }
        )
    if not rows:
        raise IcebergImportError(f"GeoJSON contains no features with coordinates: {path}")
    return pd.DataFrame(rows)


def _km_to_nm(value: Any) -> float:
    try:
        return float(value) * 0.539957
    except (TypeError, ValueError):
        return float("nan")


def save_dataframe(df: pd.DataFrame, processed_root: Path, tag: str | None = None) -> tuple[Path, Path]:
    """Write the normalized DataFrame to processed store (CSV + NetCDF)."""
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    base = tag or "icebergs"
    csv_path = processed_root / f"{base}.csv"
    nc_path = processed_root / f"{base}.nc"

    df.to_csv(csv_path, index=False)

    ds = xr.Dataset(
        {
            "iceberg_id": ("record", np.asarray(df["iceberg_id"].astype(str), dtype=object)),
            "latitude": ("record", df["latitude"].to_numpy(dtype=float)),
            "longitude": ("record", df["longitude"].to_numpy(dtype=float)),
            "length_nm": ("record", df["length_nm"].to_numpy(dtype=float)),
            "width_nm": ("record", df["width_nm"].to_numpy(dtype=float)),
        },
        coords={
            "time": ("record", df["timestamp"].to_numpy(dtype="datetime64[ns]")),
        },
        attrs={
            "title": ICEBERG_SPEC.title,
            "classification": "synthetic_demo" if base == "icebergs_demo" else df.attrs.get("classification", "imported"),
            "source_name": df.attrs.get("source_name", "import"),
            "cadence_hours": str(ICEBERG_SPEC.cadence_hours_default),
        },
    )
    write_netcdf(ds, nc_path)
    return csv_path, nc_path


def import_file(source: Path, processed_root: Path, tag: str | None = None) -> ValidationResult:
    """Import, validate, and persist an iceberg source file."""
    source = Path(source)
    result = ValidationResult(dataset="icebergs", file=source.name, status="valid")

    try:
        df = normalize_to_pandas(source)
    except (IcebergImportError, csv.Error, OSError, ValueError) as exc:
        result.status = "invalid"
        result.add("error", "IMPORT_FAILED", str(exc))
        return result

    if df.empty:
        result.status = "invalid"
        result.add("error", "EMPTY", "No valid records after normalization")
        return result

    result.meta = {
        "records": int(len(df)),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "unique_icebergs": int(df["iceberg_id"].nunique()),
        "source": source.name,
    }

    # Validate coordinates.
    bad_lat = df[(df["latitude"] < -90) | (df["latitude"] > -50)]
    if not bad_lat.empty:
        result.add(
            "error", "LAT_OUT_OF_RANGE",
            f"{len(bad_lat)} records outside Antarctic latitude band [-90,-50]",
            df.loc[bad_lat.index, ["iceberg_id", "latitude"]].head(10).to_dict("records"),
        )
        result.status = "invalid"
    bad_lon = df[(df["longitude"] < -180) | (df["longitude"] > 180)]
    if not bad_lon.empty:
        result.add("error", "LON_OUT_OF_RANGE", f"{len(bad_lon)} records with longitude outside [-180,180]")
        result.status = "invalid"

    # Validate lengths/widths and missing values.
    if df["length_nm"].isna().any() or (df["length_nm"] < 0).any():
        result.add("warning", "LENGTH_INVALID", "Some length_nm values are missing or negative")
    if df["width_nm"].isna().any() or (df["width_nm"] < 0).any():
        result.add("warning", "WIDTH_INVALID", "Some width_nm values are missing or negative")

    # Validate timestamp ordering.
    if not df["timestamp"].is_monotonic_increasing:
        result.add("warning", "TIME_ORDER", "Records are not sorted in time (they will be sorted)")

    csv_path, nc_path = save_dataframe(df, processed_root, tag=tag)
    result.add("info", "IMPORTED", f"Imported to {csv_path.name} and {nc_path.name}")
    return result


def generate_demo(processed_root: Path, n_icebergs: int = 40, days: int = 14) -> tuple[Path, Path]:
    """Generate clearly-labeled synthetic iceberg tracks."""
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(4242)
    now = datetime.now(timezone.utc)
    records = []
    dt = pd.Timedelta("6h")
    for i in range(n_icebergs):
        lat0 = rng.uniform(-76.0, -58.0)
        lon0 = rng.uniform(-180.0, 180.0)
        speed = rng.uniform(0.05, 0.5)  # deg per 6h
        heading = rng.uniform(0, 360)
        length_nm = rng.uniform(0.1, 5.0)
        width_nm = length_nm * rng.uniform(0.4, 0.9)
        for k in range(int(days * 4)):
            ts = now - pd.Timedelta(days=days) + k * dt
            lat = lat0 + speed * np.cos(np.radians(heading)) * k * 1.1
            lon = lon0 + speed * np.sin(np.radians(heading)) * k
            records.append(
                {
                    "iceberg_id": f"DEMO-B{i:03d}",
                    "timestamp": ts,
                    "latitude": round(float(_clip_lat(lat)), 4),
                    "longitude": round(float(_wrap_lon(lon)), 4),
                    "length_nm": round(float(length_nm), 3),
                    "width_nm": round(float(width_nm), 3),
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["classification"] = "synthetic_demo"
    df.attrs["source_name"] = "synthetic-demo-generator"
    csv_path, nc_path = save_dataframe(df, processed_root, tag="icebergs_demo")
    log.info("Demo icebergs -> %s (%d records)", csv_path, len(df))
    return csv_path, nc_path


def _clip_lat(lat: float) -> float:
    return max(-89.0, min(-50.0, lat))


def _wrap_lon(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def validate_processed(processed_root: Path) -> ValidationResult:
    """Validate the processed iceberg store (prefers the real import)."""
    settings = get_pipeline_settings()
    root = Path(processed_root)
    path = root / "icebergs.csv"
    if not path.exists():
        path = root / "icebergs_demo.csv"
    result = ValidationResult(dataset="icebergs", file=path.name, status="missing")
    if not path.exists():
        result.add("error", "FILE_MISSING", "No processed iceberg CSV found")
        return result

    df = pd.read_csv(path)
    result.meta = {"records": int(len(df)), "unique_icebergs": int(df["iceberg_id"].nunique())}
    result.status = "valid"

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.status = "invalid"
        result.add("error", "SCHEMA_MISSING", f"Missing columns: {missing_cols}")
        return result

    if df.empty:
        result.status = "invalid"
        result.add("error", "EMPTY", "Processed iceberg file is empty")
        return result

    if (df["latitude"] < -90).any() or (df["latitude"] > 90).any():
        result.status = "invalid"
        result.add("error", "LAT_OUT_OF_RANGE", "Latitude outside [-90,90]")
    if (df["longitude"] < -180).any() or (df["longitude"] > 180).any():
        result.status = "invalid"
        result.add("error", "LON_OUT_OF_RANGE", "Longitude outside [-180,180]")

    try:
        ts = pd.to_datetime(df["timestamp"])
    except Exception as exc:
        result.status = "invalid"
        result.add("error", "TIME_PARSE", f"Timestamp parsing failed: {exc}")
        return result

    if ts.isna().any():
        result.add("warning", "TIME_NA", f"{int(ts.isna().sum())} unparseable timestamps")

    from data_pipeline.common.validation import validate_missing_values

    issue = validate_missing_values("length_nm", df["length_nm"].to_numpy(dtype=float))
    if issue:
        result.add(issue.severity, issue.code, issue.message, issue.detail)

    return result
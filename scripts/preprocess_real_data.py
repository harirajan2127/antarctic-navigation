#!/usr/bin/env python3
"""Convert every dataset in ``Real data/raw/<dataset>/`` to a standardized CSV.

The pipeline is fully READ-ONLY w.r.t. the raw folder: no original file is
ever modified, renamed or overwritten.  It:

  1. discovers every supported file under ``Real data/raw``;
  2. inspects each file (format, variables, coordinates, time, CRS, units,
     NoData value, resolution) and writes an inventory report;
  3. converts each dataset to one (or several partitioned) standardized CSVs
     under ``Real data/processed/<dataset>/csv/``;
  4. applies coordinates/time standardisation, region + ocean/land filtering
     and quality control; every removal is counted and explained;
  5. writes a metadata JSON per dataset under
     ``Real data/processed/<dataset>/metadata/``;
  6. writes validation/quality logs and markdown reports under ``reports/``.

Usage:
    python scripts/preprocess_real_data.py --dataset all
    python scripts/preprocess_real_data.py --dataset ocean
    python scripts/preprocess_real_data.py --dataset sea_ice --sources amsr2
    python scripts/preprocess_real_data.py --dataset weather --weather-times daily
    python scripts/preprocess_real_data.py --dataset all --dry-run
"""
from __future__ import annotations

import argparse
import csv as _csv
import io
import json
import math
import os
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "Real data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "Real data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_FILE = PROJECT_ROOT / "config" / "data_config.yaml"

UTC = timezone.utc

# Selected Antarctic coastal band used by the existing application pipeline
# for the *primary* platform datasets (sea-ice, iceberg per data_config.yaml).
SELECTED_LAT_MIN = -75.0
SELECTED_LAT_MAX = -55.0
# Broader Antarctic band used by the app constants for other fields.
APP_LAT_MIN = -90.0
APP_LAT_MAX = -50.0

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".json": "json",
    ".geojson": "geojson",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".nc4": "netcdf",
    ".gpkg": "geopackage",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".asc": "ascii_grid",
    ".grib": "grib",
    ".idx": "idx",
    ".zip": "zip",
    ".hdf": "hdf",
}

NODATA_THRESHOLD_COEFF = 0.5  # nodata/missing rows above this fraction warn per file

DATASETS = ("sea_ice", "iceberg", "ocean", "weather", "vessel", "bathymetry")

LOG_RECORDS: list[dict[str, str]] = []


def log(level: str, msg: str) -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {level.upper():7s} {msg}", flush=True)
    LOG_RECORDS.append({"ts": ts, "level": level.upper(), "message": msg})


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Discovery / inventory
# ---------------------------------------------------------------------------

def _iter_dataset_files(dataset: str, extension: str | None = None) -> list[Path]:
    root = RAW_ROOT / dataset
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == ".gitkeep":
            continue
        if extension is not None and p.suffix.lower() != extension:
            continue
        out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def discover() -> dict[str, list[dict[str, Any]]]:
    inventory: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        files = _iter_dataset_files(dataset)
        rows = []
        for p in files:
            fmt = SUPPORTED_FORMATS.get(p.suffix.lower(), "unsupported")
            rows.append(
                {
                    "dataset": dataset,
                    "path": str(p.relative_to(RAW_ROOT)),
                    "size_bytes": p.stat().st_size,
                    "format": fmt,
                    "extension": p.suffix.lower(),
                }
            )
        inventory[dataset] = rows
    return inventory


def _netcdf_schema(path: Path) -> dict[str, Any]:
    try:
        import xarray as xr

        with xr.open_dataset(path, decode_times=True) as ds:
            vars_ = {}
            for name, v in ds.variables.items():
                vars_[str(name)] = {
                    "dims": [str(d) for d in v.dims],
                    "shape": list(v.shape),
                    "units": v.attrs.get("units"),
                    "fill": str(getattr(v, "_FillValue", None)),
                    "standard_name": v.attrs.get("standard_name"),
                }
            attrs = {k: str(v)[:120] for k, v in ds.attrs.items() if k in ("title", "institution", "classification", "Conventions")}
            crs = str(ds.attrs.get("crs", ""))[:200]
            return {
                "variables": vars_,
                "coords": [str(c) for c in ds.coords],
                "dims": {str(k): int(v) for k, v in ds.sizes.items()},
                "attrs": attrs,
                "crs": crs,
            }
    except Exception as exc:
        return {"error": str(exc)}


def _raster_schema(path: Path) -> dict[str, Any]:
    try:
        import rasterio

        with rasterio.open(path) as r:
            return {
                "crs": str(r.crs),
                "transform": str(r.transform),
                "shape": [int(r.height), int(r.width)],
                "bands": int(r.count),
                "nodata": str(r.nodata),
                "dtype": r.dtypes[0],
                "res": [float(abs(t)) for t in (r.transform.a, r.transform.e)],
                "bounds": [float(v) for v in r.bounds],
            }
    except Exception as exc:
        return {"error": str(exc)}


def _csv_schema(path: Path, nrows: int = 50) -> dict[str, Any]:
    try:
        df = pd.read_csv(path, nrows=nrows)
        return {
            "columns": [str(c) for c in df.columns],
            "n_columns": int(df.shape[1]),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "sample_rows": int(df.shape[0]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _zip_schema(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            first = names[0] if names else None
            head = ""
            if first:
                with z.open(first) as f:
                    head = f.read(400).decode("utf-8", errors="replace")
            return {"entries": len(names), "first_entry": first, "head": head[:300]}
    except Exception as exc:
        return {"error": str(exc)}


def inspect_files(inventory: dict[str, list[dict[str, Any]]], quick: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Best-effort per-file schema inspection (bounded: cheap op per file)."""
    max_per_ds = 60 if quick else None
    result: dict[str, list[dict[str, Any]]] = {}
    for dataset, files in inventory.items():
        subset = files[:max_per_ds] if max_per_ds else files
        out = []
        for fi in files:
            p = RAW_ROOT / fi["path"]
            fmt = fi["format"]
            if fmt in ("netcdf",):
                schema = _netcdf_schema(p)
            elif fmt in ("geotiff", "tiff", "ascii_grid"):
                schema = _raster_schema(p)
            elif fmt == "csv":
                schema = _csv_schema(p)
            elif fmt == "zip":
                schema = _zip_schema(p)
            elif fmt == "grib":
                schema = _grib_schema(p)
            elif fmt == "idx":
                schema = {
                    "size_bytes": p.stat().st_size,
                    "binary": not _looks_text(p),
                    "note": "cfgrib cache/index file (reads nothing here)",
                }
            elif fmt == "unsupported":
                schema = {"note": "unsupported format"}
            else:
                schema = {"note": f"{fmt} reader not instantiated for inventory"}
            out.append({**fi, "schema": schema})
        result[dataset] = out
    return result


def _looks_text(p: Path, n: int = 512) -> bool:
    try:
        with open(p, "rb") as f:
            data = f.read(n)
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    except OSError:
        return False


def _grib_schema(path: Path) -> dict[str, Any]:
    try:
        import eccodes

        with open(path, "rb") as f:
            try:
                msg = eccodes.codes_grib_new_from_file(f)
            except Exception:
                return {"error": "no message"}
            if msg is None:
                return {"error": "empty"}
            keys = {}
            for k in ("cfVarName", "paramId", "dataDate", "dataTime", "gridType",
                      "Nx", "Ny", "numberOfPoints", "typeOfLevel", "edition"):
                try:
                    keys[k] = eccodes.codes_get(msg, k)
                except Exception:
                    keys[k] = None
            eccodes.codes_release(msg)
            return keys
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Coordinate / time standardisation + QC
# ---------------------------------------------------------------------------

def _parse_timestamp(series: pd.Series, tz: str = "UTC") -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts


def _iso(ts_series: pd.Series) -> pd.Series:
    return ts_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def qc_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum()
    return {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {
            str(c): int(v) for c, v in missing.items() if int(v) > 0
        },
    }


def _lat_lon_ok(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat_ok = lat.between(-90, 90)
    lon_ok = lon.between(-180, 180)
    return lat_ok.to_numpy(), lon_ok.to_numpy()


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    lat_ok, lon_ok = _lat_lon_ok(df)
    bad = ~(lat_ok & lon_ok)
    return df[~bad].copy()


# ---------------------------------------------------------------------------
# Bathymetry-derived ocean / land mask (region: 20E-120E only, documented)
# ---------------------------------------------------------------------------

class OceanMask:
    """Land/ocean mask from ETOPO bathymetry when longitude in [20,120]."""

    def __init__(self, path: Path | None = None):
        self.lat: np.ndarray | None = None
        self.lon: np.ndarray | None = None
        self.ocean2d: np.ndarray | None = None
        self._loaded = False
        if path is None:
            path = _iter_dataset_files("bathymetry")[0] if _iter_dataset_files("bathymetry") else None
        if path is not None and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        try:
            import xarray as xr

            with xr.open_dataset(path) as ds:
                z = ds["z"].values
                lat = ds["latitude"].values.astype(float)
                lon = ds["longitude"].values.astype(float)
            if z.ndim != 2:
                return
            if len(lat) != z.shape[0] or len(lon) != z.shape[1]:
                z = z.T
                if len(lat) != z.shape[0]:
                    return
            finite = np.isfinite(z) & (z >= -20000)
            bad = ~finite
            self.ocean2d = (z <= 0.0) | bad  # missing -> unknown counts as ocean (best effort)
            self.lat, self.lon = lat, lon
            self._loaded = True
        except Exception as exc:
            log("warning", f"OceanMask could not load bathymetry {path}: {exc}")

    @property
    def available(self) -> bool:
        return self._loaded and self.ocean2d is not None

    def is_ocean(self, lat: float, lon: float) -> bool | None:
        """None = cannot evaluate (outside mask coverage); else bool."""
        if not self.available:
            return None
        if not (20.0 <= lon <= 120.0):
            return None
        if not (self.lat is not None and self.lon is not None):
            return None
        iy = int(np.argmin(np.abs(self.lat - lat)))
        ix = int(np.argmin(np.abs(self.lon - lon)))
        return bool(self.ocean2d[iy, ix])


# ---------------------------------------------------------------------------
# Chunked / partitioned CSV writer
# ---------------------------------------------------------------------------

class PartWriter:
    """Streams row groups to CSV part files with buffered flushes."""

    def __init__(self, out_dir: Path, stem: str, columns: list[str],
                 max_rows_per_part: int = 5_000_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.columns = columns
        self.max_rows = max(max_rows_per_part, 100_000)
        self.default = {c: np.nan for c in columns}
        self.parts: list[dict[str, Any]] = []
        self._current_rows = 0
        self._current_path: Path | None = None
        self._handle: Any = None
        self._buffer: list[list[Any]] = []

    def _open_part(self, epoch: int) -> None:
        if self._handle is not None:
            self._flush()
            self._handle.close()
            self.parts.append(
                {
                    "path": str(self._current_path.relative_to(self.out_dir)),
                    "rows": int(self._current_rows),
                }
            )
        self._current_path = self.out_dir / f"{self.stem}_part_{epoch:03d}.csv"
        self._handle = open(self._current_path, "w", newline="", encoding="utf-8")
        self._csv = _csv.writer(self._handle)
        self._csv.writerow(self.columns)
        self._current_rows = 0

    def _flush(self) -> None:
        if self._handle is None:
            return
        if self._buffer:
            self._csv.writerows(self._buffer)
            self._buffer.clear()

    def add(self, row: dict[str, Any], epoch: int) -> None:
        if self._handle is None or self._current_rows >= self.max_rows:
            if self._handle is not None:
                self._flush()
            self._open_part(epoch)
        self._buffer.append([row.get(c, np.nan) for c in self.columns])
        self._current_rows += 1
        if len(self._buffer) >= 200_000:
            self._flush()

    def close(self) -> list[dict[str, Any]]:
        if self._handle is not None:
            self._flush()
            self._handle.close()
            self._handle = None
            self.parts.append(
                {
                    "path": str(self._current_path.relative_to(self.out_dir)),
                    "rows": int(self._current_rows),
                }
            )
        # Merge any re-opened parts (same path written in non-sequential order).
        merged: dict[str, int] = {}
        for part in self.parts:
            merged[part["path"]] = merged.get(part["path"], 0) + part["rows"]
        self.parts = [{"path": k, "rows": v} for k, v in merged.items()]
        return self.parts


# ---------------------------------------------------------------------------
# Sea-ice conversion
# ---------------------------------------------------------------------------

AMSR2_GT = (-3950000.0, 3125.0, 0.0, 4350000.0, 0.0, -3125.0)   # EPSG:3976
AMSR2_SHAPE = (2656, 2528)
AMSR2_CRS = "EPSG:3976"
DMI_PROJ4 = "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
DMI_GT = (-3946875.0, 1000.0, 0.0, 4346125.0, 0.0, -1000.0)
DMI_SHAPE = (8294, 7894)


def _polar_grid(gt: tuple, shape: tuple[int, int]):
    x0, dx, _, y0, _, dy = gt
    ny, nx = shape
    cols = x0 + np.arange(nx) * dx
    rows = y0 + np.arange(ny) * dy
    return cols, rows


def _source_transformer(crs: str | None, proj4: str | None):
    """Transformer with always_xy=True: input (lon, lat) -> source metres."""
    from pyproj import Transformer

    source = proj4 if proj4 else (crs if crs else AMSR2_CRS)
    return Transformer.from_crs("EPSG:4326", source, always_xy=True)


def build_target_grid(step: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.round(np.arange(SELECTED_LAT_MIN, SELECTED_LAT_MAX + 1e-6, step), 6)
    lon = np.round(np.arange(-180.0, 180.0 - 1e-6, step), 6)
    return lat, lon


def _nearest_xy_index(xx: float, yy: float, cols: np.ndarray, rows: np.ndarray) -> tuple[int, int] | None:
    ix = int(round((xx - cols[0]) / (cols[1] - cols[0])))
    iy = int(round((yy - rows[0]) / (rows[1] - rows[0])))
    if 0 <= ix < len(cols) and 0 <= iy < len(rows):
        return iy, ix
    return None


def _polar_mapping(lat: np.ndarray, lon: np.ndarray, gt: tuple, shape: tuple[int, int],
                   crs: str | None = None, proj4: str | None = None) -> np.ndarray:
    """For target (lat_band x lon) grid cells -> flat (iy, ix) source index or -1."""
    cols, rows = _polar_grid(gt, shape)
    tr = _source_transformer(crs, proj4)
    lon2, lat2 = np.meshgrid(lon, lat)
    xx, yy = tr.transform(lon2.ravel(), lat2.ravel())
    idx = np.full(xx.size, -1, dtype=np.int64)
    for i in range(xx.size):
        if not (np.isfinite(xx[i]) and np.isfinite(yy[i])):
            continue
        r = _nearest_xy_index(float(xx[i]), float(yy[i]), cols, rows)
        if r is not None:
            idx[i] = r[0] * len(cols) + r[1]
    return idx


def _gather_flat(z2: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat = z2.ravel()
    out = np.full(idx.size, np.nan, dtype=np.float32)
    m = idx >= 0
    out[m] = flat[idx[m]]
    return out


def _find_resample_via_pandas(coords: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Fallback regrid: nearest via pandas merge_asof-alike using argmin over unique centers."""
    return values


def convert_sea_ice(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = PROCESSED_ROOT / "sea_ice" / "csv"
    meta_dir = PROCESSED_ROOT / "sea_ice" / "metadata"
    step = args.grid_step
    lat, lon = build_target_grid(step)
    n_target = lat.size * lon.size
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    lat_flat = lat_g.ravel().astype(np.float32)
    lon_flat = lon_g.ravel().astype(np.float32)

    sources = [s.strip().lower() for s in str(args.sources).split(",") if s.strip()]
    if not sources:
        sources = ["amsr2", "dmi"]

    # Build per-source index maps lazily (source grids are fixed).
    maps: dict[str, np.ndarray] = {}
    mask_land: np.ndarray | None = None  # cells that are land -> dropped

    writer = PartWriter(out_dir, "sea_ice",
                        ["timestamp", "latitude", "longitude", "sea_ice_concentration",
                         "sea_ice_concentration_unit", "quality_flag", "nodata_flag",
                         "source_file", "source_format"],
                        max_rows_per_part=args.max_rows_per_part)
    latest_writer = PartWriter(out_dir, "sea_ice_latest",
                               ["timestamp", "latitude", "longitude", "sea_ice_concentration",
                                "sea_ice_concentration_unit", "quality_flag", "nodata_flag",
                                "source_file", "source_format"],
                               max_rows_per_part=args.max_rows_per_part)

    files_seen = 0
    rows_written = 0
    rows_land = 0
    rows_nodata = 0
    per_source_counts: dict[str, int] = {s: 0 for s in sources}
    skipped: list[str] = []

    amsr2_files = sorted(
        [p for p in _iter_dataset_files("sea_ice", ".nc")
         if "asi-AMSR2-s3125" in p.name], key=lambda p: p.name)
    dmi_files = sorted(
        [p for p in _iter_dataset_files("sea_ice", ".nc")
         if p.name.startswith("dmi_asip_seaice_mosaic")], key=lambda p: p.name)

    if "amsr2" in sources and not amsr2_files:
        skipped.append("amsr2 (no files)")
    if "dmi" in sources and not dmi_files:
        skipped.append("dmi (no files)")

    # ---- Land mask: cells that AMSR2 marks NaN on every sampled day ----
    try:
        land_idx = _build_land_mask(amsr2_files, lat, lon, AMSR2_GT, AMSR2_SHAPE, AMSR2_CRS, None,
                                    max_files=min(30, len(amsr2_files)) if args.quick else len(amsr2_files))
        mask_land = land_idx
    except Exception as exc:
        log("warning", f"land-mask build skipped: {exc}")

    for src in sources:
        src_files = amsr2_files if src == "amsr2" else dmi_files
        if not src_files:
            log("warning", f"sea_ice source '{src}' has no NetCDF files")
            continue
        gt = AMSR2_GT if src == "amsr2" else DMI_GT
        shape = AMSR2_SHAPE if src == "amsr2" else DMI_SHAPE
        crs = AMSR2_CRS if src == "amsr2" else None
        proj4 = None if src == "amsr2" else DMI_PROJ4
        key = src
        if key not in maps:
            maps[key] = _polar_mapping(lat, lon, gt, shape, crs, proj4)

        src_idx = maps[key]
        valid_src = src_idx >= 0
        if mask_land is not None:
            keep = valid_src & (~mask_land)
        else:
            keep = valid_src
        if keep.sum() == 0:
            log("warning", f"sea_ice source '{src}': no target cells map onto the grid")
            continue

        timestamps: list[tuple] = []
        for p in src_files:
            try:
                ts = pd.to_datetime(_file_timestamp(src, p), errors="coerce")
            except Exception:
                ts = pd.NaT
            if pd.isna(ts):
                log("warning", f"sea_ice {src}: could not parse timestamp from {p.name}")
                continue
            timestamps.append((ts, p))

        timestamps.sort(key=lambda t: t[0])
        if args.limit_days_per_year:
            keep_dates = _limit_dates(src_files, args.limit_days_per_year)
            timestamps = [(t, p) for (t, p) in timestamps if _in_keep(p, keep_dates)]

        rows_land_src = int((valid_src & (mask_land if mask_land is not None else np.zeros(lat_flat.size, dtype=bool))).sum())
        rows_land += rows_land_src
        last_ts = timestamps[-1][0] if timestamps else None

        log("info", f"sea_ice source '{src}': converting {len(timestamps)} daily files")
        try:
            import xarray as xr
        except ImportError:
            log("error", "xarray required to convert sea-ice NetCDF")
            break

        for ts, p in timestamps:
            try:
                with xr.open_dataset(p) as ds:
                    z = ds["z"] if src == "amsr2" else ds["ice_concentration"]
                    vals = z.values[idx_slice(z, src, ts)]
            except Exception as exc:
                log("warning", f"sea_ice {p.name}: read failed: {exc}")
                continue
            vals = np.asarray(vals, dtype=np.float32).ravel()
            conc = _gather_flat(vals, src_idx)
            conc = conc[keep]
            lats_k = lat_flat[keep]
            lons_k = lon_flat[keep]
            if np.nanmax(conc) > 1.0:
                conc = conc / 100.0
            nodata = np.isnan(conc)
            files_seen += 1
            epoch = int(ts.strftime("%Y%m"))
            row_template = {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sea_ice_concentration_unit": "fraction (0..1)",
                "source_file": str(p.relative_to(RAW_ROOT)),
                "source_format": "netcdf",
            }
            for i in range(conc.size):
                if nodata[i]:
                    rows_nodata += 1
                row = {
                    **row_template,
                    "latitude": float(lats_k[i]),
                    "longitude": float(lons_k[i]),
                    "sea_ice_concentration": None if nodata[i] else round(float(conc[i]), 6),
                    "quality_flag": ("" if not nodata[i] else "nodata"),
                    "nodata_flag": 1 if nodata[i] else 0,
                }
                writer.add(row, epoch)
                rows_written += 1
                per_source_counts[src] += 1
                if isinstance(last_ts, pd.Timestamp) and ts == last_ts:
                    latest_writer.add(row, epoch)

    parts = writer.close()
    latest_parts = latest_writer.close()
    log("info", f"sea_ice done: {rows_written} rows, {len(parts)} parts, "
                f"{rows_land} land cells dropped, {rows_nodata} nodata rows kept")

    meta = {
        "dataset": "sea_ice",
        "sources": sources,
        "n_files_seen": files_seen,
        "rows_written": int(rows_written),
        "rows_land_removed": int(rows_land),
        "rows_nodata": int(rows_nodata),
        "parts": parts,
        "latest_snapshot_parts": latest_parts,
        "grid_step_deg": float(step),
        "lat_band": [SELECTED_LAT_MIN, SELECTED_LAT_MAX],
        "target_cells": int(n_target),
        "skipped": skipped,
        "per_source_counts": per_source_counts,
    }
    _write_metadata(meta, meta_dir, "sea_ice_processed_metadata")
    return meta


def idx_slice(z, src: str, ts):
    """Return the 2-D field for the (single) time step of a daily product."""
    if z.ndim == 2:
        return slice(None), slice(None)
    if z.ndim == 3:
        return 0, slice(None), slice(None)
    return slice(None), slice(None)


def _file_timestamp(src: str, p: Path):
    if src == "amsr2":
        # asi-AMSR2-s3125-YYYYMMDD-v5.4.nc
        import re
        m = re.search(r"(\d{8})", p.name)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d")
        return None
    # dmi_asip_seaice_mosaic_ant_YYYYMMDD.nc
    import re
    m = re.search(r"(\d{8})", p.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return None


def _build_land_mask(files: list[Path], lat: np.ndarray, lon: np.ndarray,
                     gt: tuple, shape: tuple[int, int], crs: str | None,
                     proj4: str | None, max_files: int = 1_000_000) -> np.ndarray:
    """Land = cell that is NaN in every sampled AMSR2 file. Return bool mask."""
    mapping = _polar_mapping(lat, lon, gt, shape, crs, proj4)
    land = np.zeros(mapping.size, dtype=bool)  # start as ocean-ish
    valid = mapping >= 0
    # Only consider cells that map onto the grid; mark land if never finite.
    ever = np.zeros(mapping.size, dtype=bool)
    samples = 0
    import xarray as xr
    for p in files[:max_files]:
        try:
            with xr.open_dataset(p) as ds:
                v = np.asarray(ds["z"].values, dtype=np.float32).ravel()
            refined = _gather_flat(v, mapping)
            ever |= np.isfinite(refined)
            samples += 1
            if samples >= max_files:
                break
        except Exception:
            continue
    if samples == 0:
        return np.zeros(mapping.size, dtype=bool)
    land[valid] = ~ever[valid]
    land[~valid] = True  # cells outside source grid are not sea-ice cells
    return land


def _limit_dates(files: list[Path], n: int) -> set[str]:
    import re
    dates = []
    for p in files:
        m = re.search(r"(\d{8})", p.name)
        if m:
            dates.append(m.group(1))
    dates = sorted(dates)
    if not dates:
        return set()
    picked = set()
    # pick `n` days spread across the year (option easy on files)
    step = max(1, len(dates) // max(1, n))
    picked = set(dates[::step])
    return picked


def _in_keep(p: Path, keep: set[str]) -> bool:
    import re
    m = re.search(r"(\d{8})", p.name)
    return bool(m and m.group(1) in keep)


# ---------------------------------------------------------------------------
# Iceberg conversion
# ---------------------------------------------------------------------------

ICEBERG_SCHEMA = [
    "iceberg_id", "timestamp", "latitude", "longitude",
    "length_nm", "width_nm", "remarks", "source_file", "source_format",
]


def convert_iceberg(args: argparse.Namespace) -> dict[str, Any]:
    files = [p for p in _iter_dataset_files("iceberg", ".csv")]
    if args.quick and files:
        files = files[:: max(1, len(files) // 6)]
    frames = []
    ocean_mask = OceanMask()
    removed = {"region": 0, "land": 0, "invalid_coords": 0, "invalid_time": 0, "duplicates": 0}
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            log("error", f"iceberg {p.name}: read failed: {exc}")
            continue
        df = df.rename(columns={
            "Iceberg": "iceberg_id", "Iceberg ID": "iceberg_id",
            "Length (NM)": "length_nm", "Width (NM)": "width_nm",
            "Latitude": "latitude", "Longitude": "longitude",
            "Remarks": "remarks", "Last Update": "last_update",
            "Last Update (UTC)": "last_update", "Date": "last_update",
        })
        if "iceberg_id" not in df.columns:
            log("error", f"iceberg {p.name}: no iceberg_id column")
            continue
        df["source_file"] = str(p.relative_to(RAW_ROOT))
        df["source_format"] = "csv"
        frames.append(df)

    if not frames:
        raise SystemExit("No iceberg CSV files could be read")
    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    n_before = before
    # timestamps
    if "last_update" not in df.columns and "timestamp" not in df.columns:
        # fall back to the file date
        df["timestamp"] = pd.NaT
    ts = _parse_timestamp(df.get("timestamp", df["last_update"]))
    bad_t = ts.isna()
    removed["invalid_time"] = int(bad_t.sum())
    df["timestamp"] = ts
    df = df[~bad_t]

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat_ok = lat.between(-90, 90); lon_ok = lon.between(-180, 180)
    bad_c = ~(lat_ok & lon_ok)
    removed["invalid_coords"] = int(bad_c.sum())
    df = df[~bad_c]
    df["latitude"] = lat[~bad_c]
    df["longitude"] = lon[~bad_c]

    # duplicates
    dup = df.duplicated()
    removed["duplicates"] = int(dup.sum())
    df = df[~dup]

    # region filter (sea-ice/iceberg band)
    in_band = df["latitude"].between(SELECTED_LAT_MIN, SELECTED_LAT_MAX)
    removed["region"] = int((~in_band).sum())
    df = df[in_band].reset_index(drop=True)

    # on-land removal (mask coverage limited to 20E-120E)
    land_rows = np.zeros(len(df), dtype=bool)
    if ocean_mask.available:
        for i, (_, row) in enumerate(df.iterrows()):
            res = ocean_mask.is_ocean(float(row["latitude"]), float(row["longitude"]))
            if res is False:
                land_rows[i] = True
    removed["land"] = int(land_rows.sum())
    df = df[~land_rows]

    df = df[ICEBERG_SCHEMA].copy() if all(c in df.columns for c in ICEBERG_SCHEMA) else df[[c for c in ICEBERG_SCHEMA if c in df.columns]]
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in ("length_nm", "width_nm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_dir = PROCESSED_ROOT / "iceberg" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iceberg_processed.csv"
    df.to_csv(out_path, index=False)
    cols = [c for c in ICEBERG_SCHEMA if c in df.columns]
    after = int(len(df))
    meta = {
        "dataset": "iceberg",
        "n_files": int(len(files)),
        "rows_before": n_before,
        "rows_after": after,
        "removed": removed,
        "output_csv": str(out_path.relative_to(PROCESSED_ROOT)),
        "columns": cols,
        "missing": {str(c): int(df[c].isna().sum()) for c in df.columns if int(df[c].isna().sum()) > 0},
        "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "sources": files[:3],
    }
    _write_metadata(meta, PROCESSED_ROOT / "iceberg" / "metadata", "iceberg_processed_metadata")
    log("info", f"iceberg done: {after} rows (from {n_before}) -> {out_path.name}")
    return meta


# ---------------------------------------------------------------------------
# Ocean conversion
# ---------------------------------------------------------------------------

OCEAN_SCHEMA = [
    "timestamp", "latitude", "longitude", "depth", "uo", "vo", "thetao", "so",
    "siconc", "source_file", "source_format",
]


def convert_ocean(args: argparse.Namespace) -> dict[str, Any]:
    files = _iter_dataset_files("ocean", ".nc")
    out_dir = PROCESSED_ROOT / "ocean" / "csv"
    writer = PartWriter(out_dir, "ocean", OCEAN_SCHEMA, max_rows_per_part=args.max_rows_per_part)
    import xarray as xr

    rows_total = 0
    rows_land = 0
    n_files = 0
    day_stride = 15 if args.quick else 1
    for p in files:
        try:
            with xr.open_dataset(p) as ds:
                lat = ds["latitude"].values.astype(np.float32)
                lon = ds["longitude"].values.astype(np.float32)
                depth_val = float(ds["depth"].values[0]) if "depth" in ds and ds["depth"].size else 0.0
                time_vals = ds["time"].values
                n_t = len(time_vals)
                for ti in range(0, n_t, day_stride):
                    arrs = {}
                    for v in ("uo", "vo", "thetao", "so", "siconc"):
                        if v in ds.variables:
                            arrs[v] = np.asarray(ds[v].isel(time=ti).values, dtype=np.float32).squeeze()
                    t0 = pd.to_datetime(time_vals[ti])
                    epoch = int(t0.strftime("%Y%m"))
                    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
                    ocean = None
                    if "thetao" in arrs:
                        ocean = np.isfinite(arrs["thetao"])
                    elif "so" in arrs:
                        ocean = np.isfinite(arrs["so"])
                    if ocean is None:
                        ocean = np.ones(lat2.shape, dtype=bool)
                    land_cells = (~ocean).sum()
                    rows_land += int(land_cells)
                    for iy in range(lat2.shape[0]):
                        for ix in range(lat2.shape[1]):
                            if not ocean[iy, ix]:
                                continue
                            row = {
                                "timestamp": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "latitude": float(lat2[iy, ix]),
                                "longitude": float(lon2[iy, ix]),
                                "depth": depth_val,
                                "uo": _v(arrs.get("uo"), iy, ix),
                                "vo": _v(arrs.get("vo"), iy, ix),
                                "thetao": _v(arrs.get("thetao"), iy, ix),
                                "so": _v(arrs.get("so"), iy, ix),
                                "siconc": _v(arrs.get("siconc"), iy, ix),
                                "source_file": str(p.relative_to(RAW_ROOT)),
                                "source_format": "netcdf",
                            }
                            writer.add(row, epoch)
                            rows_total += 1
                n_files += 1
        except Exception as exc:
            log("error", f"ocean {p.name}: conversion failed: {exc}")
            continue
    parts = writer.close()
    meta = {
        "dataset": "ocean",
        "n_files": n_files,
        "rows_written": int(rows_total),
        "rows_land_removed": int(rows_land),
        "parts": parts,
        "depth_level": "surface (first depth level)",
        "variables": ["uo", "vo", "thetao", "so", "siconc"],
    }
    _write_metadata(meta, PROCESSED_ROOT / "ocean" / "metadata", "ocean_processed_metadata")
    log("info", f"ocean done: {rows_total} rows, {len(parts)} parts")
    return meta


def _v(arr, iy, ix):
    if arr is None:
        return np.nan
    return float(arr[iy, ix]) if np.isfinite(arr[iy, ix]) else np.nan


# ---------------------------------------------------------------------------
# Weather conversion
# ---------------------------------------------------------------------------

WEATHER_SCHEMA = [
    "timestamp", "latitude", "longitude", "wind_u", "wind_v", "wind_speed",
    "wind_direction", "air_temperature", "pressure", "source_file", "source_format",
]

VAR_IDS = {"u10": "wind_u", "v10": "wind_v", "t2m": "air_temperature", "msl": "pressure"}


def convert_weather(args: argparse.Namespace) -> dict[str, Any]:
    files = _iter_dataset_files("weather", ".grib")
    out_dir = PROCESSED_ROOT / "weather" / "csv"
    writer = PartWriter(out_dir, "weather", WEATHER_SCHEMA, max_rows_per_part=args.max_rows_per_part)
    if not files:
        log("warning", "weather: no grib files")
        return {"dataset": "weather", "rows_written": 0, "parts": []}
    try:
        import eccodes
    except ImportError:
        log("error", "weather: eccodes not available")
        return {"dataset": "weather", "rows_written": 0, "parts": [], "error": "eccodes missing"}

    rows_total = 0
    for p in files:
        daily: dict = {}  # date -> {'time': dt, 'grid': (lat, lon, nx), 'vars': {name: flat}}
        n_scanned = 0
        n_decoded = 0
        with open(p, "rb") as f:
            while True:
                try:
                    msg = eccodes.codes_grib_new_from_file(f)
                except eccodes.EndOfFileError:
                    break
                if msg is None:
                    break
                try:
                    name = str(eccodes.codes_get(msg, "cfVarName"))
                    date = int(eccodes.codes_get(msg, "dataDate"))
                    tm = int(eccodes.codes_get(msg, "dataTime"))
                    if name in VAR_IDS:
                        key_date = date
                        entry = daily.get(key_date)
                        if entry is None:
                            entry = {"dateselected_time": None, "vars": {}, "nx": None,
                                     "ny": None, "lat0": None, "lon0": None,
                                     "di": None, "dj": None, "ipos": 1, "jpos": 0}
                            daily[key_date] = entry
                        # record grid metadata once (same grid for every message)
                        if entry["nx"] is None:
                            entry["nx"] = int(eccodes.codes_get(msg, "Nx"))
                            entry["ny"] = int(eccodes.codes_get(msg, "Ny"))
                            entry["lat0"] = float(eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees"))
                            entry["lon0"] = float(eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees"))
                            entry["di"] = float(eccodes.codes_get(msg, "iDirectionIncrementInDegrees"))
                            entry["dj"] = float(eccodes.codes_get(msg, "jDirectionIncrementInDegrees"))
                            try:
                                entry["ipos"] = int(eccodes.codes_get(msg, "iScansPositively"))
                            except Exception:
                                entry["ipos"] = 1
                            try:
                                entry["jpos"] = int(eccodes.codes_get(msg, "jScansPositively"))
                            except Exception:
                                entry["jpos"] = 0
                        # choose earliest dataTime per date
                        if entry["dateselected_time"] is None or tm < entry["dateselected_time"]:
                            entry["dateselected_time"] = tm
                except Exception:
                    pass
                finally:
                    eccodes.codes_release(msg)
                n_scanned += 1
        log("info", f"weather scan {p.name[:12]}…: {n_scanned} messages, {len(daily)} dates")

        # Optional quick/verification limiting of daily cadence (spread across the year).
        all_dates = sorted(daily.keys())
        limit_days = args.limit_days_per_year
        if limit_days and len(all_dates) > limit_days:
            stride = len(all_dates) // limit_days
            keep_dates = set(all_dates[::stride])
        else:
            keep_dates = set(all_dates)

        # Decode pass: only the selected daily time, only kept dates.
        for date in sorted(keep_dates):
            entry = daily[date]
            entry["sel_time"] = entry["dateselected_time"]
            t0 = pd.to_datetime(f"{date:08d} {entry['sel_time']:04d}", format="%Y%m%d %H%M")
            entry["dt"] = t0
        with open(p, "rb") as f:
            while True:
                try:
                    msg = eccodes.codes_grib_new_from_file(f)
                except eccodes.EndOfFileError:
                    break
                if msg is None:
                    break
                released = False
                try:
                    name = str(eccodes.codes_get(msg, "cfVarName"))
                    if name not in VAR_IDS:
                        eccodes.codes_release(msg)
                        released = True
                        continue
                    date = int(eccodes.codes_get(msg, "dataDate"))
                    tm = int(eccodes.codes_get(msg, "dataTime"))
                    if date not in keep_dates:
                        eccodes.codes_release(msg)
                        released = True
                        continue
                    entry = daily.get(date)
                    if entry is None or entry["dateselected_time"] != tm or name in entry["vars"]:
                        eccodes.codes_release(msg)
                        released = True
                        continue
                    vals = np.asarray(eccodes.codes_get_values(msg), dtype=np.float32)
                    entry["vars"][name] = vals
                    n_decoded += 1
                except Exception:
                    pass
                finally:
                    if not released:
                        eccodes.codes_release(msg)

        # emit rows
        for date in sorted(keep_dates):
            entry = daily[date]
            if entry["dateselected_time"] is None:
                continue
            if "u10" not in entry["vars"] or "v10" not in entry["vars"]:
                continue
            nx = entry["nx"]; ny = entry["ny"]
            u = entry["vars"]["u10"].reshape(ny, nx)
            v = entry["vars"]["v10"].reshape(ny, nx)
            t2m = entry["vars"].get("t2m")
            msl = entry["vars"].get("msl")
            lat, lon = _build_weather_grid(entry)
            # clip to Antarctic band
            lat2 = np.broadcast_to(lat[:, None], (ny, nx))
            lon2 = np.broadcast_to(lon[None, :], (ny, nx))
            keep = (lat2 >= APP_LAT_MIN) & (lat2 <= APP_LAT_MAX)
            if not keep.any():
                continue
            iys, ixs = np.where(keep)
            res = {
                "u": u[iys, ixs], "v": v[iys, ixs],
                "lat": lat2[iys, ixs], "lon": lon2[iys, ixs],
            }
            if t2m is not None:
                res["t2m"] = t2m.reshape(ny, nx)[iys, ixs]
            if msl is not None:
                res["msl"] = msl.reshape(ny, nx)[iys, ixs]
            wind_speed = np.hypot(res["u"], res["v"])
            wind_dir = (np.degrees(np.arctan2(-res["u"], -res["v"])) + 360.0) % 360.0
            epoch = int(entry["dt"].strftime("%Y%m"))
            for i in range(len(res["lat"])):
                writer.add({
                    "timestamp": entry["dt"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": float(res["lat"][i]),
                    "longitude": float(res["lon"][i]),
                    "wind_u": _f(res["u"][i]),
                    "wind_v": _f(res["v"][i]),
                    "wind_speed": _f(wind_speed[i]),
                    "wind_direction": _f(wind_dir[i]),
                    "air_temperature": _f(res.get("t2m")[i] if res.get("t2m") is not None else np.nan),
                    "pressure": _f(res.get("msl")[i] if res.get("msl") is not None else np.nan),
                    "source_file": str(p.relative_to(RAW_ROOT)),
                    "source_format": "grib",
                }, epoch)
                rows_total += 1
        log("info", f"weather {p.name[:12]}…: decoded {n_decoded} selected messages")
    parts = writer.close()
    meta = {
        "dataset": "weather",
        "n_files": int(len(files)),
        "rows_written": int(rows_total),
        "parts": parts,
        "cadence": "daily (first available dataTime per date)",
        "variables": ["wind_u", "wind_v", "wind_speed", "wind_direction", "air_temperature", "pressure"],
        "units": {"wind_u": "m s-1", "wind_v": "m s-1", "wind_speed": "m s-1",
                  "wind_direction": "degrees clockwise from north (towards)",
                  "air_temperature": "K", "pressure": "Pa"},
    }
    _write_metadata(meta, PROCESSED_ROOT / "weather" / "metadata", "weather_processed_metadata")
    log("info", f"weather done: {rows_total} rows, {len(parts)} parts")
    return meta


def _build_weather_grid(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build the lat/lon vectors from GRIB grid metadata (probe-verified: ERA5
    Antarctic subset, first point -50.0/-180.0, j scans negatively to -90.0)."""
    di = entry["di"] * (1 if entry["ipos"] else -1)
    dj = entry["dj"] * (1 if entry["jpos"] else -1)
    lat = entry["lat0"] + np.arange(entry["ny"]) * dj
    lon = entry["lon0"] + np.arange(entry["nx"]) * di
    return lat.astype(np.float64), lon.astype(np.float64)


def _f(v):
    return None if not np.isfinite(v) else round(float(v), 5)


# ---------------------------------------------------------------------------
# Vessel conversion (Global Fishing Watch monthly fleet CSVs)
# ---------------------------------------------------------------------------

VESSEL_SCHEMA = [
    "timestamp", "latitude", "longitude", "flag", "geartype",
    "hours", "fishing_hours", "mmsi_present", "source_file", "source_format",
]


def convert_vessel(args: argparse.Namespace) -> dict[str, Any]:
    zips = _iter_dataset_files("vessel", ".zip")
    out_dir = PROCESSED_ROOT / "vessel" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    region_rows = 0
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if not name.lower().endswith(".csv"):
                        continue
                    with z.open(name) as f:
                        for chunk in pd.read_csv(f, chunksize=1_000_000):
                            if not {"cell_ll_lat", "cell_ll_lon"}.issubset(chunk.columns):
                                continue
                            chunk = chunk.rename(columns={
                                "date": "timestamp", "cell_ll_lat": "latitude",
                                "cell_ll_lon": "longitude",
                            })
                            chunk = chunk[(chunk["latitude"] >= APP_LAT_MIN) & (chunk["latitude"] <= APP_LAT_MAX)]
                            if chunk.empty:
                                continue
                            region_rows += int(len(chunk))
                            chunk["timestamp"] = _parse_timestamp(chunk["timestamp"])
                            chunk["timestamp"] = chunk["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                            chunk["source_file"] = str(zp.relative_to(RAW_ROOT))
                            chunk["source_format"] = "zip|csv"
                            keep_cols = [c for c in VESSEL_SCHEMA if c in chunk.columns]
                            frames.append(chunk[keep_cols].copy())
                            log("info", f"vessel {os.path.basename(name)}: +{len(chunk)} in-region rows")
        except Exception as exc:
            log("error", f"vessel {zp.name}: {exc}")
    if not frames:
        return {"dataset": "vessel", "rows_written": 0, "error": "no vessel track data"}
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated()
    out = out[~dup].sort_values("timestamp").reset_index(drop=True)
    out_path = out_dir / "vessel_processed.csv"
    out.to_csv(out_path, index=False)
    meta = {
        "dataset": "vessel",
        "n_files": int(len(zips)),
        "rows_written": int(len(out)),
        "rows_region": int(region_rows),
        "duplicates_removed": int(dup.sum()),
        "output_csv": str(out_path.relative_to(PROCESSED_ROOT)),
        "columns": VESSEL_SCHEMA,
        "note": "Aggregated GFW fleet presence per 0.1-degree cell-day (coordinates = cell lower-left "
                "corners); no per-MMSI track ID available in source.",
    }
    _write_metadata(meta, PROCESSED_ROOT / "vessel" / "metadata", "vessel_processed_metadata")
    log("info", f"vessel done: {len(out)} in-region rows")
    return meta


# ---------------------------------------------------------------------------
# Bathymetry conversion
# ---------------------------------------------------------------------------

BATHY_SCHEMA = [
    "latitude", "longitude", "elevation_meters", "depth_meters",
    "nodata_flag", "source_file", "source_format",
]


def convert_bathymetry(args: argparse.Namespace) -> dict[str, Any]:
    files = _iter_dataset_files("bathymetry", ".nc")
    if not files:
        return {"dataset": "bathymetry", "rows_written": 0, "error": "no netcdf"}
    out_dir = PROCESSED_ROOT / "bathymetry" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = PartWriter(out_dir, "bathymetry", BATHY_SCHEMA, max_rows_per_part=args.max_rows_per_part)
    import xarray as xr

    rows = 0
    row_stride = 10 if args.quick else 1
    col_stride = 10 if args.quick else 1
    for p in files:
        with xr.open_dataset(p) as ds:
            lat = ds["latitude"].values.astype(np.float64)
            lon = ds["longitude"].values.astype(np.float64)
            z = ds["z"].values.astype(np.float64)
            if z.shape[0] != lat.size:
                z = z.T
        lat_band = np.where((lat >= APP_LAT_MIN) & (lat <= APP_LAT_MAX))[0]
        iys = lat_band[::row_stride]
        for iy in iys:
            la = lat[iy]
            epoch = max(0, int(round(la // 10)) + 95)
            for ix in range(0, lon.size, col_stride):
                lo = lon[ix]
                v = z[iy, ix]
                nodata = not math.isfinite(v) or v <= -99999.0
                elev = None if nodata else round(float(v), 2)
                depth = (None if nodata else round(float(-v), 2)) if (not nodata and v < 0) else None
                writer.add({
                    "latitude": float(la), "longitude": float(lo),
                    "elevation_meters": elev, "depth_meters": depth,
                    "nodata_flag": 1 if nodata else 0,
                    "source_file": str(p.relative_to(RAW_ROOT)),
                    "source_format": "netcdf",
                }, epoch)
                rows += 1
    parts = writer.close()
    meta = {
        "dataset": "bathymetry",
        "n_files": int(len(files)),
        "rows_written": int(rows),
        "parts": parts,
        "units": {"elevation_meters": "metres above sea level", "depth_meters": "metres below sea level"},
        "nodata": "-99999 / NaN",
        "note": "elevation_meters = source z; depth_meters = -z where z<0",
    }
    _write_metadata(meta, PROCESSED_ROOT / "bathymetry" / "metadata", "bathymetry_processed_metadata")
    log("info", f"bathymetry done: {rows} rows, {len(parts)} parts")
    return meta


# ---------------------------------------------------------------------------
# Metadata / writers
# ---------------------------------------------------------------------------

def _write_metadata(payload: dict[str, Any], meta_dir: Path, stem: str) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload.setdefault("conversion_utc", now_utc())
    payload.setdefault("status", "ok")
    payload.setdefault("warnings", [w["message"] for w in LOG_RECORDS if w["level"] == "WARNING"])
    out = meta_dir / f"{stem}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def _write_manifest() -> Path:
    manifest = {"created_utc": now_utc(), "datasets": {}}
    for dataset in DATASETS:
        csv_dir = PROCESSED_ROOT / dataset / "csv"
        if not csv_dir.exists():
            continue
        entries = []
        for p in sorted(csv_dir.glob("*.csv")):
            entries.append({"file": p.name, "rows": _count_csv_rows(p)})
        manifest["datasets"][dataset] = {"csv_dir": str(csv_dir.relative_to(PROCESSED_ROOT)), "files": entries}
    out = PROCESSED_ROOT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def _count_csv_rows(p: Path) -> int | None:
    try:
        return int(_first_pass_csv_rows(p))
    except Exception:
        return None


def _first_pass_csv_rows(p: Path) -> int:
    n = 0
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for _ in f:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _md_report(filename: str, title: str, sections: list[tuple[str, str]]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"_Generated {now_utc()}_", ""]
    for heading, body in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    out = REPORTS_DIR / filename
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _json_report(filename: str, payload: Any) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / filename
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def write_reports(inventory: dict[str, list[dict[str, Any]]],
                  inspected: dict[str, list[dict[str, Any]]],
                  results: dict[str, dict[str, Any]],
                  start: float, dry_run: bool = False) -> dict[str, Path]:
    # Markdown inventory
    md_rows = []
    for dataset, files in inspected.items():
        md_rows.append(f"\n### {dataset} ({len(files)} files)")
        if not files:
            md_rows.append("- (empty)")
        for fi in files[:120]:
            s = fi.get("schema", {})
            extra = ""
            if "variables" in s:
                extra = ", vars=" + ", ".join(list(s["variables"])[:12])
            elif "columns" in s:
                extra = ", cols=" + ", ".join(s["columns"][:12])
            elif "entries" in s:
                extra = f", entries={s['entries']}"
            elif "Nx" in s:
                extra = f", {s.get('Nx')}x{s.get('Ny')}"
            md_rows.append(f"- `{fi['path']}` [{fi['format']}] {fi['size_bytes']}B{extra}")
        if len(files) > 120:
            md_rows.append(f"- … and {len(files) - 120} more")
    _md_report("dataset_inventory.md", "Dataset Inventory", [("Detected files", "\n".join(md_rows))])

    # Conversion + preprocessing + QC reports (JSON + MD)
    rows_conv = []
    rows_pre = []
    rows_qc = []
    for dataset, r in results.items():
        rows_conv.append(f"- **{dataset}**: rows={r.get('rows_written', r.get('rows_after', '?'))}, parts={len(r.get('parts') or [])}, files={r.get('n_files', r.get('n_files_seen', '?'))}")
        removed = r.get("removed", {})
        rows_pre.append(f"- **{dataset}**: {'; '.join(f'{k}={v}' for k, v in removed.items()) if removed else 'n/a'}")
        rows_qc.append(f"- **{dataset}**: missing={json.dumps(r.get('missing', {}))}")
    _md_report("csv_conversion_report.md", "CSV Conversion Report", [
        ("Converted datasets", "\n".join(rows_conv)),
        ("Output layout", "`Real data/processed/<dataset>/csv/` + `metadata/`"),
    ])
    _md_report("preprocessing_report.md", "Preprocessing Report", [
        ("Filters applied", "\n".join(rows_pre) or "- none"),
        ("Region bands", f"sea_ice/iceberg: {SELECTED_LAT_MIN}..{SELECTED_LAT_MAX}; other: {APP_LAT_MIN}..{APP_LAT_MAX}"),
    ])
    _md_report("quality_control_report.md", "Quality Control Report", [
        ("Missing values", "\n".join(rows_qc) or "- none"),
        ("Policy", "NoData preserved with `nodata_flag`; no zero/random filling."),
    ])

    # Metadata manifest json (list of metadata files)
    meta_files = []
    for dataset in DATASETS:
        md = PROCESSED_ROOT / dataset / "metadata"
        if md.exists():
            for p in sorted(md.glob("*.json")):
                meta_files.append(str(p.relative_to(PROCESSED_ROOT)))
    _json_report("metadata_manifest.json", {
        "created_utc": now_utc(),
        "metadata_files": meta_files,
    })

    _json_report("pipeline_report.json", {
        "created_utc": now_utc(),
        "dry_run": dry_run,
        "duration_seconds": round(time.time() - start, 2),
        "datasets": results,
        "log": LOG_RECORDS[-400:],
    })
    md = _md_report("pipeline_report.md", "Real Data Pipeline Report", [
        ("Status", "dry run (no files written)" if dry_run else "completed"),
        ("Duration", f"{time.time() - start:.1f} s"),
        ("Converted", "\n".join(rows_conv) or "- none"),
    ])
    return {
        "inventory": REPORTS_DIR / "dataset_inventory.md",
        "conversion": REPORTS_DIR / "csv_conversion_report.md",
        "preprocess": REPORTS_DIR / "preprocessing_report.md",
        "qc": REPORTS_DIR / "quality_control_report.md",
        "metadata_manifest": REPORTS_DIR / "metadata_manifest.json",
        "pipeline_json": REPORTS_DIR / "pipeline_report.json",
        "pipeline_md": md,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Real data/raw datasets to standardized CSVs.")
    p.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    p.add_argument("--grid-step", type=float, default=0.25,
                   help="target lat/lon grid step for re-projected gridded products (sea_ice)")
    p.add_argument("--sources", default="amsr2,dmi",
                   help="sea_ice sources to convert: amsr2,dmi (comma separated)")
    p.add_argument("--max-rows-per-part", type=int, default=5_000_000,
                   help="target max rows per CSV part file")
    p.add_argument("--limit-days-per-year", type=int, default=None,
                   help="optional: cap conversions to N spread-out days per year (debug)")
    p.add_argument("--quick", action="store_true",
                   help="verification mode: smaller samples per dataset")
    p.add_argument("--dry-run", action="store_true", help="inspect + report only, no CSVs")
    p.add_argument("--skip-writes", action="store_true",
                   help="alias of --dry-run kept for compatibility")
    return p.parse_args(argv)


_CONVERTERS = {
    "sea_ice": convert_sea_ice,
    "iceberg": convert_iceberg,
    "ocean": convert_ocean,
    "weather": convert_weather,
    "vessel": convert_vessel,
    "bathymetry": convert_bathymetry,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    if args.dry_run or args.skip_writes:
        args.dry_run = True
    if args.quick and args.limit_days_per_year is None:
        args.limit_days_per_year = 8

    log("info", f"Discovering datasets under {RAW_ROOT}")
    inventory = discover()
    counts = {d: len(fs) for d, fs in inventory.items()}
    log("info", f"detected: {counts}")

    inspected = inspect_files(inventory, quick=args.quick)

    if args.dry_run:
        results: dict[str, Any] = {d: {"dry_run": True, "n_files": len(fs)} for d, fs in inventory.items()}
        write_reports(inventory, inspected, results, start, dry_run=True)
        log("info", "dry-run complete (no CSVs written)")
        return 0

    results: dict[str, Any] = {}
    datasets = [args.dataset] if args.dataset != "all" else list(DATASETS)
    for dataset in datasets:
        files = inventory.get(dataset, [])
        if not files:
            log("warning", f"dataset '{dataset}' has no files; marked unavailable")
            results[dataset] = {"status": "unavailable", "rows_written": 0, "skip": "no source files"}
            continue
        log("info", f"Converting dataset '{dataset}' ({len(files)} files)")
        try:
            t0 = time.time()
            res = _CONVERTERS[dataset](args)
            res["duration_seconds"] = round(time.time() - t0, 2)
            results[dataset] = res
        except Exception as exc:
            log("error", f"dataset '{dataset}' conversion failed: {exc}")
            results[dataset] = {"status": "error", "error": str(exc), "rows_written": 0}
            raise

    manifest = _write_manifest()
    log("info", f"manifest -> {manifest}")
    written = {
        "inventory": REPORTS_DIR / "dataset_inventory.md",
        "conversion": REPORTS_DIR / "csv_conversion_report.md",
        "preprocess": REPORTS_DIR / "preprocessing_report.md",
        "qc": REPORTS_DIR / "quality_control_report.md",
        "metadata_manifest": REPORTS_DIR / "metadata_manifest.json",
        "pipeline_json": REPORTS_DIR / "pipeline_report.json",
        "pipeline_md": REPORTS_DIR / "pipeline_report.md",
    }

    write_reports(inventory, inspected, results, start)
    log("info",
        f"done in {time.time() - start:.1f}s. Reports under {REPORTS_DIR}. Outputs under {PROCESSED_ROOT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
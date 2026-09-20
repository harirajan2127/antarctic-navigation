"""Shared processing utilities.

Coordinate standardization, Antarctic clipping, report generation,
spatiotemporal grid lookup, and geodesic helpers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf

log = get_logger(__name__)

# ── Antarctic region defaults ─────────────────────────────────────

LAT_MIN = -90.0
LAT_MAX = -50.0
LON_MIN = -180.0
LON_MAX = 180.0
R_NM = 3440.065  # mean Earth radius in nautical miles
KM_TO_NM = 0.539957
NM_TO_KM = 1.852


# ── Coordinate helpers ────────────────────────────────────────────

def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename common coordinate/variable conventions to standard names."""
    rename = {}
    for orig, target in {
        "lat": "latitude",
        "lat_dd": "latitude",
        "y": "latitude",
        "lon": "longitude",
        "lon_dd": "longitude",
        "x": "longitude",
        "valid_time": "time",
        "datetime": "time",
    }.items():
        if orig in ds.dims or orig in ds.coords:
            rename[orig] = target
    if rename:
        ds = ds.rename(rename)
    if "time" in ds.coords:
        try:
            ds["time"] = pd.to_datetime(ds["time"].values)
        except Exception:
            pass
    return ds


def antarctic_clip(
    ds: xr.Dataset,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
) -> xr.Dataset:
    """Subset latitude to Antarctic bounding box."""
    if "latitude" not in ds.coords:
        return ds
    mask = (ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max)
    valid_lats = ds["latitude"].values[mask.values]
    if len(valid_lats) == 0:
        return ds
    return ds.sel(latitude=valid_lats)


def daily_resample(
    ds: xr.Dataset,
    variables: Sequence[str],
    method: str = "mean",
) -> xr.Dataset:
    """Resample to daily means along time."""
    if "time" not in ds.coords:
        return ds
    rs = ds[list(variables)].resample(time="1D")
    return rs.mean() if method == "mean" else rs.nearest()


def fill_gaps_along_time(
    ds: xr.Dataset, variables: Sequence[str], max_gap: int = 2
) -> xr.Dataset:
    """Linear interpolation along time for isolated gaps."""
    for var in variables:
        if var in ds:
            ds[var] = ds[var].interpolate_na(
                dim="time", method="linear", max_gap=max_gap
            )
    return ds


# ── Coverage / validation helpers ─────────────────────────────────

def coverage_report(ds: xr.Dataset) -> dict[str, Any]:
    """Compute spatial/temporal coverage statistics."""
    cov: dict[str, Any] = {}
    if "time" in ds.coords:
        t = ds["time"].values
        cov["temporal_start"] = str(t[0])
        cov["temporal_end"] = str(t[-1])
        cov["n_time_steps"] = int(len(t))
    if "latitude" in ds.coords:
        lat = ds["latitude"].values
        cov["lat_range"] = [float(lat.min()), float(lat.max())]
        cov["n_lat"] = len(lat)
    if "longitude" in ds.coords:
        lon = ds["longitude"].values
        cov["lon_range"] = [float(lon.min()), float(lon.max())]
        cov["n_lon"] = len(lon)
    return cov


def check_units(
    ds: xr.Dataset, variable: str, expected: str
) -> dict[str, Any] | None:
    actual = ds[variable].attrs.get("units", None)
    if actual is None:
        return {
            "severity": "warning",
            "code": "UNITS_MISSING",
            "message": f"{variable}: no units attribute",
            "variable": variable,
            "expected": expected,
            "actual": None,
            "conforms": False,
        }
    if actual != expected:
        return {
            "severity": "warning",
            "code": "UNITS_MISMATCH",
            "message": f"{variable}: expected {expected}, got {actual}",
            "variable": variable,
            "expected": expected,
            "actual": actual,
            "conforms": False,
        }
    return {
        "variable": variable,
        "expected": expected,
        "actual": actual,
        "conforms": True,
    }


def value_range_stats(
    values: np.ndarray, valid_range: tuple[float, float], variable: str
) -> dict[str, Any]:
    flat = np.asarray(values, dtype=float).ravel()
    n_total = int(flat.size)
    n_nan = int(np.isnan(flat).sum())
    n_valid = n_total - n_nan
    out_range = int(((flat < valid_range[0]) | (flat > valid_range[1])).sum())
    return {
        "variable": variable,
        "total": n_total,
        "missing": n_nan,
        "missing_pct": round(n_nan / n_total * 100, 2) if n_total else 0.0,
        "min": round(float(np.nanmin(flat)), 4) if n_valid else None,
        "max": round(float(np.nanmax(flat)), 4) if n_valid else None,
        "mean": round(float(np.nanmean(flat)), 4) if n_valid else None,
        "out_of_range": out_range,
        "valid_range": list(valid_range),
    }


# ── Geodesic helpers ──────────────────────────────────────────────

def haversine_nm(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in nautical miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_NM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def forward_bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Forward azimuth in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ── Spatiotemporal grid lookup ────────────────────────────────────

class EnvLookup:
    """Nearest-neighbor spatiotemporal lookup on a processed NetCDF grid.

    Provides a transparent, auditable way to query environmental fields at
    an arbitrary (lat, lon, time).  The caller always sees the gap between
    the query time and the nearest available observation, and can set a
    tolerance to reject stale matches.
    """

    def __init__(self, path: Path, variables: Sequence[str]):
        ds = xr.open_dataset(path)
        ds = standardize_coords(ds)
        self.lats = ds["latitude"].values
        self.lons = ds["longitude"].values
        self.times = ds["time"].values
        self.data: dict[str, np.ndarray] = {}
        for v in variables:
            if v in ds:
                self.data[v] = ds[v].values
        ds.close()

    def lookup(
        self,
        lat: float,
        lon: float,
        time: np.datetime64,
        tolerance_hours: float = 24.0,
    ) -> dict[str, Any]:
        """Query environmental variables at (lat, lon, time).

        Returns a dict with one entry per variable plus metadata:
          - obs_time: the matched timestamp (string)
          - gap_hours: absolute time difference from query
          - stale: True if gap exceeds tolerance_hours
          - stale_count: 1 if stale, else 0

        When ``stale`` is True all variable values are NaN — the caller
        must never silently use a stale field as if it were current.
        """
        lat_idx = int(np.argmin(np.abs(self.lats - lat)))
        lon_idx = int(np.argmin(np.abs(self.lons - lon)))
        time_diffs = np.abs(self.times - time)
        time_idx = int(np.argmin(time_diffs))
        gap_hours = float(time_diffs[time_idx] / np.timedelta64(1, "h"))
        stale = gap_hours > tolerance_hours
        result: dict[str, Any] = {
            "obs_time": str(self.times[time_idx]),
            "gap_hours": round(gap_hours, 2),
            "stale": stale,
            "stale_count": 1 if stale else 0,
        }
        for var, vals in self.data.items():
            if stale:
                result[var] = np.nan
            else:
                result[var] = float(vals[time_idx, lat_idx, lon_idx])
        return result


# ── Report dataclass ──────────────────────────────────────────────

@dataclass
class ProcessingReport:
    dataset: str
    status: str = "ok"
    shape: dict[str, int] | None = None
    missing: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    units: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = d["warnings"] or []
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# {self.dataset} Processing Report",
            f"**Status**: {self.status}",
            "",
        ]
        if self.shape:
            lines.append("## Dataset Shape")
            for k, v in self.shape.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        if self.missing:
            lines.append("## Missing Data")
            for var, info in self.missing.items():
                cnt = info.get("count", info.get("missing", "?"))
                pct = info.get("pct", info.get("missing_pct", 0))
                lines.append(f"- **{var}**: {cnt} NaN ({pct:.2f}%)")
            lines.append("")
        if self.units:
            lines.append("## Unit Conformity")
            for u in self.units:
                ok = "OK" if u.get("conforms") else "MISMATCH"
                lines.append(
                    f"- `{u['variable']}`: {ok} "
                    f"(expected: {u.get('expected', '?')}, actual: {u.get('actual', '?')})"
                )
            lines.append("")
        if self.coverage:
            lines.append("## Data Coverage")
            self._render_payload(lines, self.coverage, indent=0)
            lines.append("")
        if self.warnings:
            lines.append("## Notes / Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_payload(lines: list[str], payload: dict[str, Any], indent: int) -> None:
        """Render a coverage payload as nested markdown bullets."""
        pad = "  " * indent
        for k, v in payload.items():
            if isinstance(v, dict):
                lines.append(f"{pad}- `{k}`: {v.get('n_total', '')}".rstrip())
                ProcessingReport._render_payload(
                    lines,
                    {kk: vv for kk, vv in v.items() if kk not in ("n_total",)},
                    indent + 1,
                )
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"{pad}- `{k}`:")
                for item in v:
                    keys = ", ".join(f"{kk}={vv}" for kk, vv in item.items())
                    lines.append(f"{pad}  - {keys}")
            else:
                lines.append(f"{pad}- `{k}`: {v}")


def write_report(
    report: ProcessingReport, reports_dir: Path, stem: str
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    log.info("Report: %s", md_path)
    return json_path, md_path

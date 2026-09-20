"""Iceberg track preprocessing.

Reads iceberg observations from CSV, NetCDF, or GeoJSON; validates
``iceberg_id``; parses/sorts timestamps; computes per-track kinematics
(displacement, speed, direction, time difference, distance travelled);
and preserves the full historical track for every iceberg. Writes a
clean time-series CSV + NetCDF plus a processing report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import is_string_dtype

from data_pipeline.common.logging_setup import get_logger
from data_pipeline.common.netcdf_utils import write_netcdf
from data_pipeline.icebergs.import_data import (
    normalize_to_pandas,
)

from data.processing._common import (
    ProcessingReport,
    forward_bearing,
    haversine_nm,
    write_report,
)

log = get_logger(__name__)

KINEMATICS = (
    "time_diff_hours",
    "displacement_nm",
    "speed_knots",
    "bearing_deg",
    "heading_change_deg",
    "distance_traveled_nm",
    "first_seen",
    "age_hours",
    "track_displacement_nm",
)


class IcebergProcessor:
    """Build a clean, kinematics-augmented iceberg track table."""

    def __init__(
        self,
        input_path: Path,
        output_csv: Path,
        output_nc: Path,
        reports_dir: Path,
        *,
        min_velocity_hours: float = 0.05,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_csv = Path(output_csv)
        self.output_nc = Path(output_nc)
        self.reports_dir = Path(reports_dir)
        self.min_velocity_hours = min_velocity_hours

    def process(self) -> tuple[Path, Path]:
        log.info("Iceberg processing: %s", self.input_path)
        df = normalize_to_pandas(self.input_path)

        report = ProcessingReport(dataset="icebergs")
        warnings: list[str] = []

        # ── 1. Validate iceberg_id ────────────────────────────────
        n_before = len(df)
        bad_id = df["iceberg_id"].isna() | (df["iceberg_id"].astype(str).str.strip() == "")
        if bad_id.any():
            warnings.append(f"Dropped {int(bad_id.sum())} rows with empty iceberg_id")
            df = df[~bad_id].copy()
        if not is_string_dtype(df["iceberg_id"]):
            df["iceberg_id"] = df["iceberg_id"].astype(str)

        # ── 2. Parse + sort ───────────────────────────────────────
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        n_parsed = int(df["timestamp"].notna().sum())
        if n_parsed < len(df):
            warnings.append(
                f"Dropped {len(df) - n_parsed} rows with unparseable timestamps"
            )
            df = df[df["timestamp"].notna()].copy()
        df = df.sort_values(["iceberg_id", "timestamp"]).reset_index(drop=True)

        # ── 3. Coordinate sanity ──────────────────────────────────
        bad_lat = (df["latitude"] < -90) | (df["latitude"] > 90)
        bad_lon = (df["longitude"] < -180) | (df["longitude"] > 180)
        if bad_lat.any() or bad_lon.any():
            warnings.append(
                f"Dropped {(bad_lat | bad_lon).sum()} rows with out-of-range coordinates"
            )
            df = df[~(bad_lat | bad_lon)].copy()

        if df.empty:
            report.status = "failed"
            report.warnings = warnings
            write_report(report, self.reports_dir, "iceberg_processing")
            raise ValueError("Iceberg preprocessing produced an empty table")

        # ── 4. Per-track kinematics ───────────────────────────────
        parts = [
            self._kinematics(group, self.min_velocity_hours)
            for _, group in df.groupby("iceberg_id")
        ]
        df = pd.concat(parts, ignore_index=True)

        # ── 5. Shape + coverage + missing ─────────────────────────
        missing_info = {
            col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()
        }
        report.shape = {"rows": int(len(df)), "columns": int(df.shape[1]), "icebergs": int(df["iceberg_id"].nunique())}
        report.coverage = {
            "temporal_start": str(df["timestamp"].min()),
            "temporal_end": str(df["timestamp"].max()),
            "n_icebergs": int(df["iceberg_id"].nunique()),
        }
        report.missing = {k: {"count": v, "total": n_before, "pct": round(v / n_before * 100, 2) if n_before else 0.0} for k, v in missing_info.items()}
        report.warnings = warnings
        report.status = "ok" if not warnings else "completed_with_warnings"

        # ── 6. Persist ────────────────────────────────────────────
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        self._write_netcdf(df, self.output_nc)
        write_report(report, self.reports_dir, "iceberg_processing")
        log.info("Icebergs done: %s (%d rows)", self.output_csv, len(df))
        return self.output_csv, self.output_nc

    @staticmethod
    def _kinematics(group: pd.DataFrame, min_velocity_hours: float) -> pd.DataFrame:
        """Add per-row kinematics and track-level metadata for one iceberg."""
        g = group.sort_values("timestamp").reset_index(drop=True).copy()
        n = len(g)
        g["time_diff_hours"] = np.nan
        g["displacement_nm"] = np.nan
        g["speed_knots"] = np.nan
        g["bearing_deg"] = np.nan
        g["heading_change_deg"] = np.nan
        g["distance_traveled_nm"] = 0.0
        g["track_displacement_nm"] = np.nan
        g["first_seen"] = pd.NaT
        g["age_hours"] = np.nan
        if n < 2:
            g["track_displacement_nm"] = 0.0
            g["first_seen"] = g["timestamp"].iloc[0]
            g["age_hours"] = 0.0
            return g

        lat1 = g["latitude"].to_numpy()[:-1]
        lon1 = g["longitude"].to_numpy()[:-1]
        lat2 = g["latitude"].to_numpy()[1:]
        lon2 = g["longitude"].to_numpy()[1:]
        t1 = g["timestamp"].to_numpy(dtype="datetime64[ns]")[:-1]
        t2 = g["timestamp"].to_numpy(dtype="datetime64[ns]")[1:]

        time_diff_h = (t2 - t1) / np.timedelta64(1, "h")
        time_diff_h = np.maximum(time_diff_h, min_velocity_hours)
        disp = haversine_nm(lat1, lon1, lat2, lon2)
        bearing = forward_bearing(lat1, lon1, lat2, lon2)
        speed = disp / time_diff_h
        # wrap heading changes to (-180, 180]
        hc = np.diff(bearing)
        hc = (hc + 180.0) % 360.0 - 180.0
        cum = np.cumsum(disp)

        g.loc[g.index[1:], "time_diff_hours"] = np.round(time_diff_h, 4)
        g.loc[g.index[1:], "displacement_nm"] = np.round(disp, 4)
        g.loc[g.index[1:], "speed_knots"] = np.round(speed, 4)
        g.loc[g.index[1:], "bearing_deg"] = np.round(bearing, 2)
        if n >= 3:
            g.loc[g.index[2:], "heading_change_deg"] = np.round(hc, 2)
        g.loc[g.index[1:], "distance_traveled_nm"] = np.round(cum, 4)
        g.loc[g.index[1:], "track_displacement_nm"] = float(cum[-1])
        g.loc[g.index[0], "track_displacement_nm"] = float(cum[-1])
        g["first_seen"] = g["timestamp"].iloc[0]
        g["age_hours"] = np.round(
            (g["timestamp"] - g["timestamp"].iloc[0]) / np.timedelta64(1, "h"), 2
        )
        return g

    @staticmethod
    def _write_netcdf(df: pd.DataFrame, path: Path) -> Path:
        ds = xr.Dataset(
            {
                "iceberg_id": ("record", np.asarray(df["iceberg_id"].astype(str), dtype=object)),
                "latitude": ("record", df["latitude"].to_numpy(dtype=float)),
                "longitude": ("record", df["longitude"].to_numpy(dtype=float)),
                "length_nm": ("record", df["length_nm"].to_numpy(dtype=float)),
                "width_nm": ("record", df["width_nm"].to_numpy(dtype=float)),
                "displacement_nm": ("record", df["displacement_nm"].to_numpy(dtype=float)),
                "speed_knots": ("record", df["speed_knots"].to_numpy(dtype=float)),
                "bearing_deg": ("record", df["bearing_deg"].to_numpy(dtype=float)),
                "distance_traveled_nm": ("record", df["distance_traveled_nm"].to_numpy(dtype=float)),
            },
            coords={
                "time": ("record", df["timestamp"].to_numpy(dtype="datetime64[ns]")),
            },
            attrs={"title": "Iceberg tracks (processed for ML)", "classification": "synthetic_demo" if "demo" in Path(path).stem else "processed"},
        )
        write_netcdf(ds, path)
        return path
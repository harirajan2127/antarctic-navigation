"""Iceberg trajectory dataset preparation.

This module turns raw iceberg track tables into machine-learning sequences:

1. group by ``iceberg_id`` and sort chronologically
2. project each track into a local tangent plane (origin = first fix) -> X, Y km
3. derive kinematics (displacement, speed, direction, vx, vy)
4. attach ocean/weather drivers (uo, vo, u10, v10) at each fix
5. build sliding windows ``[N, seq_len, F]`` with next-position targets ``[N, 2]``
6. drop any sequence containing a non-finite value (invalid sequences)
7. provide deterministic chronological train/val/test splits and per-channel
   scalers fitted on the train split only

The primary input is ``backend/data/processed/features/feature_table.csv``,
produced by the preprocessing pipeline (it already carries the environment
lookups). An ``icebergs_clean.csv`` + ocean/weather NetCDF combo is also
supported via nearest-neighbour lookups.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from .utils import (
    LocalTangentPlane,
    bearing_deg,
    haversine_km,
    sin_cos_doy,
)

LOG = logging.getLogger(__name__)

# Feature order exposed to every learner.
FEATURE_COLS = [
    "x", "y",            # 0,1  projected easting/northing (km)
    "vx", "vy",          # 2,3  velocity (km/h)
    "uo", "vo",          # 4,5  ocean current (m/s)
    "u10", "v10",        # 6,7  wind (m/s)
    "length", "width",   # 8,9  iceberg dimensions (km -> normalised by scaler)
    "sin_doy", "cos_doy",  # 10,11 time features
]
N_FEATURES = len(FEATURE_COLS)
TARGET_IDX = (0, 1)


@dataclass
class IcebergTrack:
    """One chronologically sorted track in projected coordinates."""

    iceberg_id: str
    times: np.ndarray          # datetime64[us]
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray              # km
    y: np.ndarray              # km
    vx: np.ndarray             # km/h
    vy: np.ndarray             # km/h
    uo: np.ndarray
    vo: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    length: np.ndarray         # km
    width: np.ndarray          # km
    displacement_km: np.ndarray
    distance_traveled_km: np.ndarray
    speed_kmh: np.ndarray
    direction_deg: np.ndarray
    plane: LocalTangentPlane
    median_dt_hours: float = field(default=float("nan"))

    def __len__(self) -> int:
        return len(self.times)

    def to_local_features(self, i: int) -> np.ndarray:
        """Row i's feature vector before time features are appended by builder."""
        return np.array([
            self.x[i], self.y[i], self.vx[i], self.vy[i],
            self.uo[i], self.vo[i], self.u10[i], self.v10[i],
            self.length[i], self.width[i],
        ], dtype=np.float64)


class TrajectoryDataset:
    """Prepared, windowed iceberg trajectory data ready for ML.

    Attributes
    ----------
    seq : np.ndarray  [N, seq_len, F]  (std-scaled after ``fit_scalers``)
    seq_raw : np.ndarray
    targets : np.ndarray  [N, 2]  projected next position (km, unscaled)
    mask_meta : np.ndarray of dict entries per sequence
    tracks : dict[str, IcebergTrack]
    scalers : {mean: [F], std: [F]}
    """

    def __init__(
        self,
        tracks: dict[str, IcebergTrack],
        track_order: list[str],
        seq: np.ndarray,
        targets: np.ndarray,
        meta: list[dict[str, Any]],
        seq_len: int,
        horizon: int,
        scalers: dict[str, np.ndarray] | None = None,
    ):
        self.tracks = tracks
        self.track_order = track_order
        self.seq_raw = seq.astype(np.float64)
        self.seq = seq.astype(np.float32)
        self.targets = targets.astype(np.float64)
        self.meta = meta
        self.seq_len = seq_len
        self.horizon = horizon
        self.scalers = scalers or {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_features(
        cls,
        feature_csv: Path | str,
        seq_len: int = 5,
        horizon: int = 1,
        min_track_len: int | None = None,
        ocean_nc: Path | str | None = None,
        weather_nc: Path | str | None = None,
    ) -> "TrajectoryDataset":
        df = pd.read_csv(feature_csv, parse_dates=["timestamp"] if "timestamp" in _cols_of(feature_csv) else [])
        return _build(cls, df, seq_len, horizon, min_track_len, ocean_nc, weather_nc)

    @classmethod
    def from_parts(
        cls,
        tracks_csv: Path | str,
        ocean_nc: Path | str,
        weather_nc: Path | str,
        seq_len: int = 5,
        horizon: int = 1,
        min_track_len: int | None = None,
    ) -> "TrajectoryDataset":
        """Join a bare icebergs CSV with environment NetCDF grids."""
        df = pd.read_csv(tracks_csv)
        df = _join_env(df, ocean_nc, weather_nc)
        return _build(cls, df, seq_len, horizon, min_track_len, ocean_nc, weather_nc)

    # ------------------------------------------------------------------
    # Scalers & splits
    # ------------------------------------------------------------------
    def fit_scalers(self, train_idx: np.ndarray) -> None:
        x = self.seq_raw[train_idx].reshape(-1, N_FEATURES)  # [N*T, F]
        finite = np.isfinite(x)
        mean = np.zeros(N_FEATURES)
        std = np.ones(N_FEATURES)
        for f in range(N_FEATURES):
            v = x[finite[:, f], f]
            if v.size:
                mean[f] = v.mean()
                std[f] = max(v.std(), 1e-8)
        self.scalers = {"mean": mean, "std": std}

    def normalize(self, scalers: Mapping[str, Any] | None = None) -> None:
        stats = dict(scalers or self.scalers)
        if not stats:
            raise ValueError("Call fit_scalers() first.")
        flat = self.seq_raw.reshape(-1, N_FEATURES)
        flat = (flat - stats["mean"]) / stats["std"]
        self.seq = flat.reshape(self.seq_raw.shape).astype(np.float32)
        self.scalers = stats

    def split_chronological(self, train_frac: float = 0.7, val_frac: float = 0.15) -> dict[str, np.ndarray]:
        """Deterministic split by target timestamp (oldest -> train -> val -> test)."""
        times = np.array([m["target_time"] for m in self.meta], dtype="datetime64[us]")
        order = np.argsort(times, kind="stable", axis=0)
        n = len(times)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        idx = order
        return {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }

    def tabular(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flatten (sequence, step) for tree-based models: [n, seq_len*F]."""
        x = self.seq[idx].reshape(len(idx), self.seq_len * N_FEATURES)
        return x, self.targets[idx]

    def seq_tensor(self, idx: np.ndarray) -> tuple[Any, Any]:
        import torch

        return (
            torch.from_numpy(self.seq[idx]),
            torch.from_numpy(self.targets[idx].astype(np.float32)),
        )

    def track(self, iceberg_id: str) -> IcebergTrack:
        return self.tracks[iceberg_id]


# ---------------------------------------------------------------------------
# internal builders
# ---------------------------------------------------------------------------

def _cols_of(path) -> list[str]:
    import csv

    with Path(path).open("r", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _read_track_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    df["iceberg_id"] = df["iceberg_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["iceberg_id", "time", "latitude", "longitude"])
    df = df.sort_values(["iceberg_id", "time"]).drop_duplicates(["iceberg_id", "time"]).reset_index(drop=True)
    for col in ("uo", "vo", "u10", "v10"):
        if col not in df.columns:
            df[col] = np.nan
    for col in ("length_nm", "width_nm"):
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0
    return df


def _build(
    cls,
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    min_track_len: int | None,
    ocean_nc,
    weather_nc,
) -> TrajectoryDataset:
    df = _read_track_frame(df)
    if min_track_len is None:
        min_track_len = seq_len + horizon + 1
    if min_track_len < seq_len + horizon + 1:
        LOG.warning("min_track_len=%d raised to %d", min_track_len, seq_len + horizon + 1)
        min_track_len = seq_len + horizon + 1

    tracks: dict[str, IcebergTrack] = {}
    dropped = 0
    for iceberg_id, g in df.groupby("iceberg_id", sort=False):
        if len(g) < min_track_len:
            dropped += 1
            continue
        t = _make_track(iceberg_id, g)
        tracks[iceberg_id] = t
    if not tracks:
        raise ValueError(
            f"No iceberg has at least {min_track_len} observations "
            f"(need seq_len={seq_len} + horizon={horizon} + 1)."
        )
    LOG.info("tracks: %d kept, %d dropped too short", len(tracks), dropped)

    seqs, targets, meta = [], [], []
    for iceberg_id in tracks:
        tr = tracks[iceberg_id]
        n = len(tr)
        for i in range(0, n - seq_len - horizon + 1):
            win_end = i + seq_len
            target_row = i + seq_len + horizon - 1
            block = np.stack([tr.to_local_features(k) for k in range(i, win_end)], axis=0)
            sdt, cdt = sin_cos_doy(tr.times[i:win_end])
            block = np.concatenate([block, np.stack([sdt, cdt], axis=-1)], axis=-1)
            tgt = np.array([tr.x[target_row], tr.y[target_row]])
            if not (np.isfinite(block).all() and np.isfinite(tgt).all()):
                continue
            seqs.append(block)
            targets.append(tgt)
            meta.append({
                "iceberg_id": iceberg_id,
                "start_time": str(tr.times[i]),
                "target_time": str(tr.times[target_row]),
                "horizon_steps": horizon,
                "origin_lat": float(tr.plane.lat0),
                "origin_lon": float(tr.plane.lon0),
                "last_lat": float(tr.lat[win_end - 1]),
                "last_lon": float(tr.lon[win_end - 1]),
                "dt_hours": tr.median_dt_hours,
            })
    if not seqs:
        raise ValueError("No valid sequences after NaN removal (all sequences invalid).")

    seq_arr = np.stack(seqs)
    tgt_arr = np.stack(targets)
    LOG.info(
        "sequences: %d  shape=%s  targets=%s  (%.1f%% of candidate windows kept)",
        len(seq_arr), seq_arr.shape, tgt_arr.shape,
        100.0 * len(seq_arr) / max(1, len(targets)),
    )
    return cls(
        tracks=tracks,
        track_order=list(tracks),
        seq=seq_arr,
        targets=tgt_arr,
        meta=meta,
        seq_len=seq_len,
        horizon=horizon,
    )


def _make_track(iceberg_id: str, g: pd.DataFrame) -> IcebergTrack:
    lat = g["latitude"].to_numpy(dtype=np.float64)
    lon = g["longitude"].to_numpy(dtype=np.float64)
    times = g["time"].to_numpy(dtype="datetime64[us]")

    plane = LocalTangentPlane.origin_of(lat, lon)
    x, y = plane.forward(lat, lon)

    dt_h = np.diff(times.astype("datetime64[h]").astype(np.int64)).astype(np.float64)
    dt_h = np.concatenate([[float("nan")], dt_h])
    median_dt = float(np.nanmedian(dt_h[1:])) if len(dt_h) > 1 else float("nan")

    vx = np.full_like(x, np.nan)
    vy = np.full_like(y, np.nan)
    vx[1:] = np.diff(x) / dt_h[1:]
    vy[1:] = np.diff(y) / dt_h[1:]

    disp = np.full(len(lat), np.nan)
    for i in range(1, len(lat)):
        disp[i] = haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i])
    cum = np.nancumsum(np.nan_to_num(disp[1:]))
    cum = np.concatenate([[0.0], cum])
    speed = disp[1:] / dt_h[1:]
    speed = np.concatenate([[np.nan], speed])
    direction = np.full(len(lat), np.nan)
    for i in range(1, len(lat)):
        direction[i] = bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])

    length = g["length_nm"].to_numpy(dtype=np.float64) * 1.852  # nm -> km
    width = g["width_nm"].to_numpy(dtype=np.float64) * 1.852

    env = {
        c: g[c].to_numpy(dtype=np.float64)
        for c in ("uo", "vo", "u10", "v10")
    }
    return IcebergTrack(
        iceberg_id=iceberg_id,
        times=times,
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        uo=env["uo"],
        vo=env["vo"],
        u10=env["u10"],
        v10=env["v10"],
        length=length,
        width=width,
        displacement_km=disp,
        distance_traveled_km=cum,
        speed_kmh=speed,
        direction_deg=direction,
        plane=plane,
        median_dt_hours=median_dt,
    )


def _join_env(df: pd.DataFrame, ocean_nc, weather_nc) -> pd.DataFrame:
    """Nearest lat/lon/time lookup of uo/vo (ocean) and u10/v10 (weather)."""
    df = df.copy()
    times = pd.to_datetime(df["time"])
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    o = xr.open_dataset(ocean_nc)
    w = xr.open_dataset(weather_nc)
    try:
        df["uo"] = _lookup(o, "uo", times, lat, lon)
        df["vo"] = _lookup(o, "vo", times, lat, lon)
        df["u10"] = _lookup(w, "u10", times, lat, lon)
        df["v10"] = _lookup(w, "v10", times, lat, lon)
    finally:
        o.close()
        w.close()
    return df


def _lookup(ds: xr.Dataset, var: str, times, lat, lon) -> np.ndarray:
    da = ds[var]
    out = np.full(len(times), np.nan)
    for i, (t, la, lo) in enumerate(zip(times, lat, lon)):
        sel = da.sel(
            time=t,
            latitude=la,
            longitude=lo,
            method="nearest",
        )
        v = np.asarray(sel.values, dtype=np.float64)
        out[i] = float(np.ravel(v)[0])
    return out
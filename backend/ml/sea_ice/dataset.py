"""Dataset construction for sea-ice forecasting.

Reads the grids produced by the preprocessing layer
(``backend/data/processed/**``), regrids every feature onto the sea-ice
reference grid via nearest-neighbour lookup, and builds sliding windows of
shape ``[n_samples, input_steps, H, W, F]`` with a next-day target grid
``[n_samples, H, W]``.

Every row is either fully known or masked out (``mask`` array [n, H, W]): a
cell participates in loss/metrics only if all input channels **and** the
target are finite, so invalid grid cells never leak into training.

Splits are strictly chronological (oldest -> train -> val -> test) and are
computed without any randomness, which makes training and evaluation
reproducible across runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .utils import apply_scaler, fit_channel_scaler, sin_cos_doy

LOG = logging.getLogger(__name__)

# Channel order shown to every learner. Index 0 is always the sea-ice
# concentration (the predict target), the rest are drivers.
BASE_CHANNELS = [
    "sea_ice_concentration",  # 0  target channel
    "sic_change",             # 1  first difference along time
    "thetao",                 # 2  ocean temperature (K)
    "uo",                     # 3  ocean zonal current (m/s)
    "vo",                     # 4  ocean meridional current (m/s)
    "u10",                    # 5  wind zonal (m/s)
    "v10",                    # 6  wind meridional (m/s)
    "t2m",                    # 7  2m temperature (K)
    "msl",                    # 8  mean sea-level pressure (Pa)
]
TIME_CHANNELS = ["sin_doy", "cos_doy"]
ALL_CHANNELS = BASE_CHANNELS + TIME_CHANNELS

TARGET_CHANNEL = BASE_CHANNELS.index("sea_ice_concentration")
N_CHANNELS = len(ALL_CHANNELS)

# dataset name -> (path key, variables to pull from that file)
DATASET_VARS: dict[str, tuple[str, list[str]]] = {
    "sea_ice": ("sea_ice", ["sea_ice_concentration"]),
    "ocean": ("ocean", ["thetao", "uo", "vo", "so"]),
    "weather": ("weather", ["u10", "v10", "t2m", "msl"]),
}


def _nearest_sel(ds: xr.Dataset, var: str, time, lat, lon) -> np.ndarray:
    da = ds[var]
    try:
        arr = da.sel(time=time, latitude=lat, longitude=lon, method="nearest").values
    except (KeyError, ValueError):
        # grid has non-trivial/unusual dims: fall back to direct lookup by name
        return da.values.astype(np.float32)
    return arr.astype(np.float32)


@dataclass
class SeaIceDataset:
    """Loaded, aligned, windowed sea-ice data ready for ML.

    Attributes
    ----------
    X : [n_samples, input_steps, H, W, F]  normalized channel grid
    X_raw : [n_samples, input_steps, H, W, F]  unnormalized channel grid
    y : [n_samples, H, W]  next-day concentration target
    mask : [n_samples, H, W]  valid-cell mask (finite in all inputs + target)
    times : pandas.DatetimeIndex (n_samples)  target dates per sample
    lats, lons : spatial coordinates of the (subsampled) grid
    scalers : dict[str, np.ndarray] mean/std per channel (fit on train only)
    """

    X: np.ndarray
    X_raw: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    times: pd.DatetimeIndex
    lats: np.ndarray
    lons: np.ndarray
    input_steps: int
    horizon: int
    spatial_step: int
    scalers: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        files: Mapping[str, Path],
        input_steps: int = 7,
        horizon: int = 1,
        spatial_step: int = 2,
    ) -> "SeaIceDataset":
        ref = Path(files["sea_ice"])
        with xr.open_dataset(ref) as ds:
            times_all = pd.to_datetime(ds["time"].values)
            lats_all = ds["latitude"].values.astype(np.float64)
            lons_all = ds["longitude"].values.astype(np.float64)
            ref_name = "sea_ice_concentration"

        lats = lats_all[::spatial_step]
        lons = lons_all[::spatial_step]

        raw: dict[str, np.ndarray] = {}
        for dsname, (key, vars_) in DATASET_VARS.items():
            with xr.open_dataset(Path(files[key])) as ds:
                for var in vars_:
                    raw[var] = _nearest_sel(ds, var, times_all, lats, lons)
        # keep the source order for the stack
        order = [v for _, (_, vs) in DATASET_VARS.items() for v in vs]
        raw = {v: raw[v] for v in order}

        T = raw["sea_ice_concentration"].shape[0]
        sic = raw["sea_ice_concentration"]  # [T,H,W]
        if np.isscalar(sic) or sic.ndim != 3:
            raise ValueError("Sea-ice file must be a 3-D grid (time, latitude, longitude).")

        # first-difference follows the time axis
        sic_change = np.full_like(sic, np.nan)
        sic_change[1:] = sic[1:] - sic[:-1]

        sin_doy, cos_doy = sin_cos_doy(times_all)
        sin_arr = np.broadcast_to(sin_doy[:, None, None], sic.shape).astype(np.float32)
        cos_arr = np.broadcast_to(cos_doy[:, None, None], sic.shape).astype(np.float32)

        channels: dict[str, np.ndarray] = {
            "sea_ice_concentration": sic,
            "sic_change": sic_change,
            # skip 'so' (salinity) — not part of the learned driver stack
            "thetao": raw["thetao"],
            "uo": raw["uo"],
            "vo": raw["vo"],
            "u10": raw["u10"],
            "v10": raw["v10"],
            "t2m": raw["t2m"],
            "msl": raw["msl"],
            "sin_doy": sin_arr,
            "cos_doy": cos_arr,
        }
        missing = [c for c in BASE_CHANNELS if c not in channels]
        if missing:
            raise ValueError(f"Missing required channels: {missing}")

        grid = np.stack([channels[c] for c in ALL_CHANNELS], axis=-1).astype(np.float32)

        n = T - input_steps - horizon + 1
        if n < 2:
            raise ValueError(
                f"Not enough timesteps ({T}) for input_steps={input_steps} + "
                f"horizon={horizon}. Need at least {input_steps + horizon + 1}."
            )
        X_raw = np.stack([grid[s : s + input_steps] for s in range(n)])  # [n,Ti,H,W,F]
        y = np.stack([sic[s + input_steps : s + input_steps + horizon] for s in range(n)])[:, 0]
        mask = (
            np.all(np.isfinite(X_raw), axis=(1, 4))
            & np.isfinite(y)
        )
        times = pd.to_datetime(times_all[input_steps + horizon - 1 :])

        ds_ = cls(
            X_raw=X_raw.copy(),
            X=X_raw.copy(),
            y=y.astype(np.float32),
            mask=mask,
            times=times,
            lats=lats,
            lons=lons,
            input_steps=input_steps,
            horizon=horizon,
            spatial_step=spatial_step,
        )
        LOG.info(
            "SeaIceDataset ready: %d samples, %d steps, grid %dx%d, %d channels, %.1f%% of cells valid.",
            n, input_steps, ds_.lats.size, ds_.lons.size, N_CHANNELS,
            100.0 * mask.mean(),
        )
        return ds_

    # ------------------------------------------------------------------
    def split_chronological(
        self, train_frac: float = 0.7, val_frac: float = 0.15
    ) -> dict[str, np.ndarray]:
        """Contiguous, order-preserving index arrays (no shuffling/leakage)."""
        n = self.X.shape[0]
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        # absorb rounding into the test tail
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        n_test = n - n_train - n_val
        idx = np.arange(n)
        return {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }

    def fit_scalers(self, train_idx: np.ndarray) -> None:
        """Per-channel mean/std from train cells only (no data leakage)."""
        train_x = self.X_raw[train_idx]
        weights = self.mask[train_idx][:, None, :, :].astype(np.float64)
        self.scalers = fit_channel_scaler(train_x, weights)

    def normalize(self, scalers: Mapping[str, np.ndarray] | None = None) -> None:
        stats = dict(scalers or self.scalers)
        if not stats:
            raise ValueError("No scalers available; call fit_scalers() first.")
        flat = self.X_raw.reshape(-1, N_CHANNELS)
        flat_n = apply_scaler(flat, stats)
        self.X = flat_n.reshape(self.X_raw.shape).astype(np.float32)
        # Missing (unobserved) cells become a neutral 0.0 input so convolution
        # doesn't spill NaNs to neighbours; the per-sample ``mask`` still tells
        # the loss and the metrics which cells are actually observable.
        self.X = np.nan_to_num(self.X, nan=0.0)
        self.scalers = stats

    def tabular_cells(
        self, idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten (sample, cell) tabular features for tree-based baselines.

        Returns (X2 [n*h*w, input_steps*F], y2 [n*h*w], mask2 [n*h*w]).
        Rows are ordered (sample0, all cells), (sample1, ...), and the mask
        column tells the caller which cells are valid; nothing is dropped.
        """
        xsub = self.X[idx]
        n, ti, h, w, f = xsub.shape
        x = xsub.transpose(0, 2, 3, 1, 4).reshape(n * h * w, ti * f)
        y = self.y[idx].reshape(n * h * w)
        m = self.mask[idx].reshape(n * h * w)
        return x, y, m


class SeaIceWindowDataset(Dataset):
    """Minimal torch Dataset over SeaIceDataset windows."""

    def __init__(self, ds: SeaIceDataset, idx: np.ndarray):
        self.x = torch.from_numpy(ds.X[idx])
        self.y = torch.from_numpy(ds.y[idx])
        self.mask = torch.from_numpy(ds.mask[idx])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[i],
            "y": self.y[i],
            "mask": self.mask[i],
        }
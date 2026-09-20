"""Shared utilities for the sea-ice forecasting pipeline.

This module is deliberately self-contained (no dependency on the rest of the
``data_pipeline`` package) so the pipeline can be run, debugged and unit
tested in isolation.

Everything that touches the filesystem writes UTF-8 so reports are safe on
any platform (Windows console defaults to a non-UTF-8 codec).
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

LOG = logging.getLogger("sea_ice.ml")

DISCLAIMER_DEMO = (
    "DEMO MODE: trained/validated on SYNTHETIC demo data. No real observations "
    "were used. All numbers in this run are code-path checks and carry NO "
    "information about real-world forecasting skill."
)

_DISCLAIMER_SEED = "sea-ice-ml-v1"


def set_seed(seed: int) -> None:
    """Make sampling, numpy, torch and sklearn reproducible for a given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def configured_logger(verbose: bool = False) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger("sea_ice.ml")
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return root


def is_demo_dataset(paths: Iterable[Path | str]) -> bool:
    """True if any NetCDF input carries a synthetic/demo classification."""
    for p in paths:
        p = Path(p)
        if p.suffix != ".nc" or not p.exists():
            continue
        with xr.open_dataset(p) as ds:
            cls = str(ds.attrs.get("classification", "")).lower()
        if "demo" in cls or "synthetic" in cls:
            return True
    return False


def explain_data_situation(demo_data: bool, demo_flag: bool) -> None:
    """Clear, honest messaging about what can and cannot be trained.

    Raises SystemExit if the caller asks for a meaningful model while only
    demo data is available.
    """
    if demo_data and not demo_flag:
        raise SystemExit(
            "\n[sea-ice] The only available input files are SYNTHETIC (demo) data.\n"
            "  A real sea-ice model CANNOT be meaningfully trained on them.\n"
            "  Options:\n"
            "    1. Provide real NetCDF files (see data/processing pipeline and README).\n"
            "    2. Run the *demo smoke test* with  --demo  which exercises the full\n"
            "       code path but produces no meaningful skill numbers.\n"
        )
    if demo_data and demo_flag:
        LOG.warning("Running in DEMO mode on synthetic data. Metrics are NOT meaningful.")
    if not demo_data and demo_flag:
        LOG.warning("--demo was passed but inputs look real; the flag is ignored.")


def sin_cos_doy(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclic day-of-year features (sin, cos) in [0, 1] for numpy datetime64."""
    dts = pd.to_datetime(times)
    doy = np.array([d.timetuple().tm_yday for d in dts], dtype=np.float64)
    ang = 2.0 * np.pi * (doy - 1.0) / 366.0
    return (np.sin(ang) + 1.0) / 2.0, (np.cos(ang) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Masked metrics (per-sample arrays: [n, H, W])
# ---------------------------------------------------------------------------

def _valid(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask & np.isfinite(actual) & np.isfinite(pred)
    return pred[m], actual[m]


def masked_mae(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float:
    p, a = _valid(np.asarray(pred), np.asarray(actual), np.asarray(mask))
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - a)))


def masked_rmse(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float:
    p, a = _valid(np.asarray(pred), np.asarray(actual), np.asarray(mask))
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def masked_spatial_correlation_per_sample(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray
) -> list[float]:
    """Pearson correlation in the spatial domain, one value per sample."""
    pred = np.asarray(pred)
    actual = np.asarray(actual)
    mask = np.asarray(mask)
    out = []
    for s in range(pred.shape[0]):
        p, a = _valid(pred[s], actual[s], mask[s])
        if p.size < 2 or np.std(p) == 0 or np.std(a) == 0:
            out.append(float("nan"))
        else:
            out.append(float(np.corrcoef(p, a)[0, 1]))
    return out


def masked_spatial_correlation(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray
) -> float:
    vals = masked_spatial_correlation_per_sample(pred, actual, mask)
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE used as the ConvLSTM training loss (mean over valid cells).

    NaNs in ``pred`` (from convolutional spill-over of masked neighbours) are
    treated as zeros so they never contaminate the loss.
    """
    m = mask.float()
    diff = ((pred - target) ** 2) * m
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    denom = m.sum().clamp(min=1.0)
    return diff.sum() / denom


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_channel_scaler(x: np.ndarray, feature_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel mean/std on the (n, ..., F) array using valid cells only.

    ``feature_mask`` must broadcast against x.shape[:-1]; cells where it is
    True (train-valid + finite) participate, everywhere else is ignored.
    Returns {"mean": np.array[F], "std": np.array[F]}.
    """
    x = np.asarray(x, dtype=np.float64)
    feat = np.moveaxis(x, -1, 0)  # [F, n, Ti, H, W]
    f = feat.shape[0]
    m = np.broadcast_to(np.asarray(feature_mask, dtype=bool), feat.shape[1:])
    mean = np.zeros(f, dtype=np.float64)
    std = np.ones(f, dtype=np.float64)
    for i, ch in enumerate(feat):
        vals = ch[m & np.isfinite(ch)]
        if vals.size > 0:
            mean[i] = vals.mean()
            std[i] = max(vals.std(), 1e-8)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def apply_scaler(x: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - stats["mean"]) / stats["std"]


def save_scalers(scalers: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, path)
    return path


def load_scalers(path: Path | str) -> dict[str, Any]:
    return joblib.load(Path(path))


def save_json(obj: Any, path: Path | str, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return path


def _json_default(o: Any):
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime, np.datetime64, pd.Timestamp)):
        return str(o)
    return str(o)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_CBAR_LABEL = "sea-ice concentration (fraction)"


def _imshow_map(ax, lat2d, lon2d, arr, vmin, vmax, label):
    im = ax.pcolormesh(lon2d, lat2d, arr, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(label)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, label=_CBAR_LABEL)
    ax.set_aspect("auto")
    ax.grid(True, linestyle=":", alpha=0.3)
    return im


def plot_forecast_maps(
    actual: np.ndarray,
    predicted: np.ndarray,
    error: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    out_path: Path | str,
    timestamp: str = "",
) -> Path:
    """Actual / predicted / error map triple for the most recent valid sample."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lon2d, lat2d = np.meshgrid(lons, lats)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig.suptitle(f"Sea-ice forecast {timestamp}".strip())
    ax_m = np.ma.masked_invalid(actual)
    _imshow_map(axes[0], lat2d, lon2d, ax_m, actual.min(), actual.max(), "Actual")
    _imshow_map(axes[1], lat2d, lon2d, np.ma.masked_invalid(predicted), actual.min(), actual.max(), "Predicted")
    _imshow_map(axes[2], lat2d, lon2d, np.ma.masked_invalid(error), -0.5, 0.5, "Error (pred-actual)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_loss_curve(csv_path: Path | str, out_path: Path | str) -> Path:
    """Loss curve from the epoch CSV; expects columns epoch,train_loss,val_loss."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
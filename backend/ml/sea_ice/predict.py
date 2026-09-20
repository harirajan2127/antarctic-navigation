"""Sea-ice forecasting inference.

Usage (from ``backend/``):

    python ml/sea_ice/predict.py --out models/sea_ice/run

Loads the checkpoint saved by ``train.py`` and predicts the day after the
most recent observation, or a user-specified date if available.  Writes a
NetCDF file and a JSON summary to the ``<out>/predictions/`` directory.

If the input data is synthetic (demo), the outputs clearly say so and
contain **no meaningful skill** — they exist only to exercise the code path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from .dataset import N_CHANNELS, SeaIceDataset
from .model import build_model
from .train import TrainConfig
from .utils import (
    DISCLAIMER_DEMO,
    configured_logger,
    load_scalers,
    now_utc,
    save_json,
    set_seed,
)

LOG = logging.getLogger(__name__)


def run(
    out_dir: Path | str,
    prediction_date: str | None = None,
    out_dir_predictions: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_pred = Path(out_dir_predictions or out_dir / "predictions")
    out_pred.mkdir(parents=True, exist_ok=True)

    cfg_dict = _read_json(out_dir / "config.json")
    cfg = TrainConfig.from_dict(cfg_dict)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = {
        "sea_ice": Path(cfg.sea_ice),
        "ocean": Path(cfg.ocean),
        "weather": Path(cfg.weather),
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing input recorded in config: {p}")

    ds = SeaIceDataset.from_files(
        paths,
        input_steps=cfg.input_steps,
        horizon=cfg.horizon,
        spatial_step=cfg.spatial_step,
    )
    ds.normalize(load_scalers(out_dir / "scalers.joblib"))
    splits = ds.split_chronological(cfg.train_frac, cfg.val_frac)
    all_idx = splits["test"]  # the prediction target lives at the tail of time
    last_sample_idx = int(all_idx[-1]) if all_idx.size > 0 else int(ds.X.shape[0] - 1)
    target_time = ds.times[last_sample_idx]

    # --- optionally pick a specific date ---
    target_date_override: pd.Timestamp | None = None
    if prediction_date:
        target_date_override = pd.Timestamp(prediction_date)
        # locate the sample whose target time matches
        matches = np.where(ds.times == target_date_override)[0]
        if matches.size == 0:
            available = sorted(set(ds.times))
            raise ValueError(
                f"Requested date {prediction_date} is not a valid target in the dataset.\n"
                f"  Available target dates: {[str(t) for t in available[:6]]} ..."
            )
        last_sample_idx = int(matches[0])
        target_time = ds.times[last_sample_idx]

    # --- build model (ConvLSTM, or Random-Forest fallback) ---
    ckpt_path = out_dir / "model.pt"
    rf_path = out_dir / "rf.joblib"
    model_name = "convlstm"
    if ckpt_path.exists():
        model_kwargs = _load_model_kwargs(ckpt_path, cfg)
        model_kwargs["model_name"] = "convlstm"
        model = build_model("convlstm", model_kwargs, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    elif rf_path.exists():
        from .baselines import load_rf
        model = load_rf(rf_path)
        model_name = "random_forest"
    else:
        raise FileNotFoundError("No trained model found in the artifacts directory.")

    # --- predict ---
    if model_name == "convlstm":
        x = torch.from_numpy(ds.X[last_sample_idx:last_sample_idx + 1]).to(device)
        pred = model.predict_grid(x)[0, :, :, 0].cpu().numpy()  # [H, W]
    else:  # random forest regression on (cell, time-features) rows
        x2, _, m2 = ds.tabular_cells(np.array([last_sample_idx]))
        pred_flat = np.clip(model.predict(x2), 0.0, 1.0)
        pred = np.full((ds.lats.size, ds.lons.size), np.nan, dtype=np.float64)
        pred[m2.reshape(ds.lats.size, ds.lons.size)] = pred_flat[m2]
        pred = pred.squeeze()

    n_cells = ds.lats.size * ds.lons.size
    stats = {
        "mean": float(np.nanmean(pred)),
        "min": float(np.nanmin(pred)),
        "max": float(np.nanmax(pred)),
        "std": float(np.nanstd(pred)),
        "fraction_above_0_15": float(np.nanmean(pred > 0.15)),
    }

    # --- write NetCDF ---
    ds_out = xr.Dataset(
        {"sea_ice_concentration": (["latitude", "longitude"], pred.astype(np.float32))},
        coords={"latitude": ds.lats, "longitude": ds.lons},
        attrs={
            "title": "Sea-ice concentration forecast",
            "target_time": str(target_time),
            "model": model_name,
            "source": "prediction",
            "classification": "demo" if cfg.demo else "real",
            "created_utc": now_utc(),
            "disclaimer": DISCLAIMER_DEMO if cfg.demo else None,
        },
    )
    nc_path = out_pred / f"prediction_{str(target_time.date())}.nc"
    ds_out.to_netcdf(nc_path, encoding={
        "sea_ice_concentration": {"dtype": "float32"},
    })

    # --- JSON summary ---
    summary: dict[str, Any] = {
        "created_utc": now_utc(),
        "target_date": str(target_time),
        "prediction_source": "model.pt" if model_name == "convlstm" else "rf.joblib",
        "model_name": model_name,
        "inputs": {k: str(v) for k, v in paths.items()},
        "grid": {"H": ds.lats.size, "W": ds.lons.size,
                 "spatial_step": cfg.spatial_step},
        "statistics": stats,
        "output_nc": str(nc_path),
        "demo_only": bool(cfg.demo),
        "disclaimer": DISCLAIMER_DEMO if cfg.demo else None,
    }
    json_path = out_pred / f"prediction_{str(target_time.date())}.json"
    save_json(summary, json_path)
    LOG.info("Inference complete -> %s", nc_path)
    LOG.info("  Mean SIC: %.3f (min %.3f max %.3f)", stats["mean"], stats["min"], stats["max"])
    return summary


def _load_model_kwargs(ckpt_path: Path, cfg: TrainConfig) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kw = ckpt.get("model_kwargs", None)
    if kw is None:
        return {
            "in_channels": N_CHANNELS,
            "hidden_channels": cfg.hidden_channels,
            "kernel_size": cfg.kernel_size,
            "sigmoid_output": cfg.sigmoid_output,
        }
    kw = dict(kw)
    kw.setdefault("in_channels", N_CHANNELS)
    return kw


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Predict the next day's sea-ice concentration.")
    p.add_argument("--out", required=True, help="training artifacts directory")
    p.add_argument("--date", default=None, help="target date (YYYY-MM-DD); default: last available")
    p.add_argument("--predictions-out", default=None,
                   help="output directory for predictions (default: <out>/predictions)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configured_logger(verbose=args.verbose)
    run(Path(args.out), prediction_date=args.date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
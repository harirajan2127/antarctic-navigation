"""Evaluation for the sea-ice forecasting pipeline.

Loads the artifacts saved by ``train.py`` and recomputes masked MAE / RMSE /
spatial-correlation on the test split, then writes maps (actual, predicted,
error), the training-loss curve and a plain comparison of baselines.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import evaluate_persistence
from .dataset import SeaIceDataset
from .model import build_model
from .train import TrainConfig
from .utils import (
    DISCLAIMER_DEMO,
    configured_logger,
    load_scalers,
    masked_mae,
    masked_rmse,
    masked_spatial_correlation,
    plot_forecast_maps,
    plot_loss_curve,
    save_json,
)

LOG = logging.getLogger(__name__)


@torch.no_grad()
def predict_split(
    model: torch.nn.Module, ds: SeaIceDataset, idx: np.ndarray, device: torch.device
) -> np.ndarray:
    """Predict every sample of ``idx`` as one batch. Returns [n, H, W] clipped grids."""
    model.eval()
    x = torch.from_numpy(ds.X[idx]).to(device)
    out = model.predict_grid(x)  # [B, H, W, 1]
    return out.squeeze(-1).cpu().numpy()


def evaluate_convlstm_split(
    model: torch.nn.Module,
    ds: SeaIceDataset,
    idx: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    pred = predict_split(model, ds, idx, device)
    y = ds.y[idx]
    mask = ds.mask[idx]
    return {
        "mae": round(float(masked_mae(pred, y, mask)), 6),
        "rmse": round(float(masked_rmse(pred, y, mask)), 6),
        "spatial_correlation": round(
            float(masked_spatial_correlation(pred, y, mask)), 6
        ),
        "n_valid_cells": int(mask.sum()),
    }


def run(out_dir: Path | str) -> dict[str, Any]:
    out_dir = Path(out_dir)
    cfg = TrainConfig.from_dict(_read_json(out_dir / "config.json"))
    set_seed_quiet(cfg.seed)

    paths = {
        "sea_ice": Path(cfg.sea_ice),
        "ocean": Path(cfg.ocean),
        "weather": Path(cfg.weather),
    }
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing input recorded in config: {p}")

    ds = SeaIceDataset.from_files(
        paths, input_steps=cfg.input_steps, horizon=cfg.horizon,
        spatial_step=cfg.spatial_step,
    )
    scalers = load_scalers(out_dir / "scalers.joblib")
    ds.normalize(scalers)
    splits = ds.split_chronological(cfg.train_frac, cfg.val_frac)
    test = splits["test"]

    result: dict[str, Any] = {
        "created_utc": _now(),
        "demo_only": cfg.demo,
        "disclaimer": DISCLAIMER_DEMO if cfg.demo else None,
        "out_dir": str(out_dir),
        "config": cfg.to_dict(),
    }

    # Model A always
    res = evaluate_persistence(ds, test)
    result["persistence"] = _plain(res)

    # Model B if trained
    rf_path = out_dir / "rf.joblib"
    if rf_path.exists():
        from .baselines import evaluate_random_forest, load_rf
        result["model_b"] = {
            "model": "random_forest",
            **_plain(evaluate_random_forest(load_rf(rf_path), ds, test)),
        }
        LOG.info("Evaluated Model B (random forest) on test split.")

    # Model C if checkpoint exists
    ckpt_path = out_dir / "model.pt"
    model_c = None
    if ckpt_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = _read_ckpt_model_kwargs(ckpt_path, device)
        model = build_model("convlstm", model_kwargs, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        eval_c = evaluate_convlstm_split(model, ds, test, device)
        result["model_c"] = {"model": "convlstm", **eval_c}
        model_c = model
        LOG.info("Evaluated Model C (ConvLSTM) on test split: %s", eval_c)

    # ---- maps (use the most recent test sample) ----
    ev = out_dir / "evaluation"
    ev.mkdir(parents=True, exist_ok=True)
    last = int(test[-1])
    actual = ds.y[last]
    mask = ds.mask[last]

    if model_c is not None:
        pred = predict_split(model_c, ds, np.array([last]), torch.device("cpu"))
        pred_m = pred[0].copy()
    elif rf_path.exists():
        from .baselines import evaluate_random_forest, load_rf
        rf_res = evaluate_random_forest(load_rf(rf_path), ds, np.array([last]))
        pred_m = rf_res.prediction[0].copy()
    else:
        pred_m = evaluate_persistence(ds, np.array([last])).prediction[0].copy()

    err = pred_m - actual
    pred_m[~mask] = np.nan
    plot_forecast_maps(
        actual, pred_m, err, ds.lats, ds.lons,
        ev / "forecast_maps.png", timestamp=str(ds.times[last]),
    )
    if ckpt_path.exists() and (out_dir / "loss_history.csv").exists():
        plot_loss_curve(out_dir / "loss_history.csv", ev / "loss_curve.png")

    result["artifacts"] = {
        "maps": str(ev / "forecast_maps.png"),
        "loss_curve": str(ev / "loss_curve.png")
        if ckpt_path.exists() and (out_dir / "loss_history.csv").exists() else None,
        "latest_test_target_time": str(ds.times[last]),
    }
    save_json(result, out_dir / "evaluation" / "metrics.json")
    _write_plain_md(result, ev / "metrics.md")
    LOG.info("Evaluation artifacts -> %s", ev)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    import json
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_ckpt_model_kwargs(path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_kwargs" in ckpt:
        return dict(ckpt["model_kwargs"])
    from .dataset import N_CHANNELS
    cfg = TrainConfig.from_dict(_read_json(path.parent / "config.json"))
    return {
        "in_channels": N_CHANNELS,
        "hidden_channels": cfg.hidden_channels,
        "kernel_size": cfg.kernel_size,
        "sigmoid_output": cfg.sigmoid_output,
    }


def _plain(res) -> dict[str, Any]:
    return {
        "mae": round(float(res.mae), 6) if np.isfinite(res.mae) else None,
        "rmse": round(float(res.rmse), 6) if np.isfinite(res.rmse) else None,
        "spatial_correlation": round(float(res.spatial_correlation), 6)
        if np.isfinite(res.spatial_correlation) else None,
        "n_valid_cells": int(res.n_valid_cells),
    }


def _now() -> str:
    from .utils import now_utc
    return now_utc()


def set_seed_quiet(seed: int) -> None:
    from .utils import set_seed
    set_seed(seed)


def _write_plain_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sea-ice evaluation", "",
        f"- Created (UTC): `{result['created_utc']}`",
        f"- Demo-only run: `{str(result['demo_only']).upper()}`",
        "",
        "## Test-split metrics",
    ]
    if result.get("disclaimer"):
        lines += ["", "> **" + result["disclaimer"] + "**"]
    order = ["persistence", "model_b", "model_c"]
    for key in order:
        if key not in result:
            continue
        v = result[key]
        lines += [
            "",
            f"### {key.title()}",
            f"- MAE: **{v['mae']}**",
            f"- RMSE: **{v['rmse']}**",
            f"- Spatial correlation: **{v['spatial_correlation']}**",
            f"- Valid cells: {v.get('n_valid_cells')}",
        ]
    lines += ["", "## Artifacts"]
    for k, v in result.get("artifacts", {}).items():
        lines.append(f"- `{k}`: `{v}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Evaluate a trained sea-ice model.")
    p.add_argument("--out", required=True, help="training output directory")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configured_logger(verbose=args.verbose)
    run(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
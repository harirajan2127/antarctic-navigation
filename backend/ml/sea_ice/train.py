"""Training entry point for the sea-ice forecasting pipeline.

Usage (from ``backend/``):

    python ml/sea_ice/train.py --model random_forest --demo --spatial-step 2
    python ml/sea_ice/train.py --model convlstm    --demo --spatial-step 2

The run always evaluates Model A (persistence); Model B (random forest) is
trained when ``--model random_forest`` or ``--with-rf`` is given; Model C
(ConvLSTM) when ``--model convlstm`` (the default).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import (
    evaluate_persistence,
    fit_random_forest,
    evaluate_random_forest,
    save_rf,
)
from .dataset import N_CHANNELS, SeaIceDataset, SeaIceWindowDataset
from .model import build_model, count_parameters
from .utils import (
    DISCLAIMER_DEMO,
    configured_logger,
    explain_data_situation,
    is_demo_dataset,
    masked_mse_torch,
    now_utc,
    plot_loss_curve,
    save_json,
    save_scalers,
    set_seed,
)

LOG = logging.getLogger(__name__)

DEFAULT_FILES = {
    "sea_ice": "data/processed/sea_ice/sea_ice_clean.nc",
    "ocean": "data/processed/ocean/ocean_clean.nc",
    "weather": "data/processed/weather/weather_clean.nc",
}


@dataclasses.dataclass
class TrainConfig:
    seed: int = 7
    device: str = "auto"            # auto | cpu | cuda
    out_dir: str = "models/sea_ice/run"
    demo: bool = False              # allow training on synthetic data

    sea_ice: str = DEFAULT_FILES["sea_ice"]
    ocean: str = DEFAULT_FILES["ocean"]
    weather: str = DEFAULT_FILES["weather"]

    input_steps: int = 7
    horizon: int = 1
    spatial_step: int = 2           # 1 = full resolution
    train_frac: float = 0.7
    val_frac: float = 0.15

    model: str = "convlstm"         # convlstm | random_forest | persistence
    with_rf: bool = False

    hidden_channels: int = 16
    kernel_size: int = 3
    sigmoid_output: bool = True
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 4
    early_stop_patience: int = 5
    weight_decay: float = 0.0

    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    rf_subsample: int | None = 200_000

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if cfg.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _convlstm_kwargs(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "in_channels": N_CHANNELS,
        "hidden_channels": cfg.hidden_channels,
        "kernel_size": cfg.kernel_size,
        "sigmoid_output": cfg.sigmoid_output,
    }


def _train_convlstm(
    cfg: TrainConfig,
    ds: SeaIceDataset,
    splits: dict[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    model = build_model("convlstm", _convlstm_kwargs(cfg), device)
    LOG.info("ConvLSTM parameters: %d", count_parameters(model))

    train_ds = SeaIceWindowDataset(ds, splits["train"])
    val_ds = SeaIceWindowDataset(ds, splits["val"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    history_path = out_dir / "loss_history.csv"
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience_left = cfg.early_stop_patience
    header_written = False

    with history_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad()
                pred = model(x).squeeze(-1)
                if cfg.sigmoid_output:
                    pred = torch.sigmoid(pred)
                loss = masked_mse_torch(pred, y, mask)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = float(np.mean(train_losses))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    mask = batch["mask"].to(device)
                    pred = model(x).squeeze(-1)
                    if cfg.sigmoid_output:
                        pred = torch.sigmoid(pred)
                    val_losses.append(masked_mse_torch(pred, y, mask).item())
            val_loss = float(np.mean(val_losses))

            if not header_written:
                writer.writerow(["epoch", "train_loss", "val_loss"])
                header_written = True
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            fh.flush()

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                patience_left = cfg.early_stop_patience
                best_state = {
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                }
            else:
                patience_left -= 1

            LOG.info(
                "epoch %d/%d  train=%s  val=%s%s",
                epoch,
                cfg.epochs,
                f"{train_loss:.5f}",
                f"{val_loss:.5f}",
                "  (best)" if improved else "",
            )
            if patience_left <= 0:
                LOG.info("Early stopping at epoch %d.", epoch)
                break

    if best_state is None:
        raise RuntimeError("ConvLSTM produced no checkpoint during training.")

    torch.save(
        {**best_state, "config": cfg.to_dict(),
         "model_kwargs": _convlstm_kwargs(cfg), "seed": cfg.seed},
        out_dir / "model.pt",
    )
    plot_loss_curve(history_path, out_dir / "loss_curve.png")
    return {
        "model": "convlstm",
        "best_epoch": best_state["epoch"],
        "best_val_loss": best_state["val_loss"],
        "train_loss": best_state["train_loss"],
        "parameters": count_parameters(model),
    }


def run(cfg: TrainConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    paths = {k: Path(v) for k, v in cfg.to_dict().items() if k in ("sea_ice", "ocean", "weather")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    demo_data = is_demo_dataset(paths.values())
    explain_data_situation(demo_data, cfg.demo)
    cfg.demo = bool(cfg.demo or demo_data)  # persist the truth in artifacts

    ds = SeaIceDataset.from_files(
        paths, input_steps=cfg.input_steps, horizon=cfg.horizon,
        spatial_step=cfg.spatial_step,
    )
    splits = ds.split_chronological(cfg.train_frac, cfg.val_frac)
    ds.fit_scalers(splits["train"])
    ds.normalize()
    save_scalers(ds.scalers, out_dir / "scalers.joblib")

    LOG.info(
        "Splits (samples): train=%d val=%d test=%d",
        splits["train"].size, splits["val"].size, splits["test"].size,
    )

    # ---- Model A (always) ----
    persistence = evaluate_persistence(ds, splits["test"])
    LOG.info("Persistence test: MAE=%.4f RMSE=%.4f r=%.3f",
             persistence.mae, persistence.rmse, persistence.spatial_correlation)
    results: dict[str, Any] = {"persistence": _result_to_plain(persistence)}

    # ---- Model B ----
    rf_info = None
    if cfg.model == "random_forest" or cfg.with_rf:
        rf = fit_random_forest(
            ds, splits["train"],
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            subsample=cfg.rf_subsample,
            random_state=cfg.seed,
        )
        save_rf(rf, out_dir / "rf.joblib")
        rf_res = evaluate_random_forest(rf, ds, splits["test"])
        rf_info = {"model": "random_forest", **_result_to_plain(rf_res)}
        LOG.info("Random forest test: MAE=%.4f RMSE=%.4f r=%.3f",
                 rf_res.mae, rf_res.rmse, rf_res.spatial_correlation)

    # ---- Model C ----
    conv_info = None
    if cfg.model == "convlstm":
        device = _resolve_device(cfg)
        LOG.info("Device: %s", device)
        conv_info = _train_convlstm(cfg, ds, splits, device, out_dir)
        # evaluate best checkpoint on the test split
        model = build_model("convlstm", _convlstm_kwargs(cfg), device)
        ckpt = torch.load(out_dir / "model.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        from .evaluate import evaluate_convlstm_split  # circular-import safe at call time

        conv_eval = evaluate_convlstm_split(model, ds, splits["test"], device)
        conv_info.update({k: conv_eval[k] for k in ("mae", "rmse", "spatial_correlation")})
        LOG.info("ConvLSTM test: MAE=%.4f RMSE=%.4f r=%.3f",
                 conv_eval["mae"], conv_eval["rmse"], conv_eval["spatial_correlation"])
        results["model_c"] = {
            "model": "convlstm",
            **{k: conv_info[k] for k in ("best_epoch", "best_val_loss", "parameters")},
            **conv_eval,
        }

    if rf_info is not None:
        results["model_b"] = rf_info

    summary = {
        "created_utc": now_utc(),
        "demo_only": cfg.demo,
        "disclaimer": DISCLAIMER_DEMO if cfg.demo else None,
        "inputs": {k: str(v) for k, v in paths.items()},
        "dataset": {
            "n_samples": int(ds.X.shape[0]),
            "samples_train": int(splits["train"].size),
            "samples_val": int(splits["val"].size),
            "samples_test": int(splits["test"].size),
            "target_times": [str(t) for t in ds.times],
            "grid": {"H": int(ds.lats.size), "W": int(ds.lons.size),
                     "lat_min": float(ds.lats.min()), "lat_max": float(ds.lats.max())},
            "mask_valid_frac": float(ds.mask.mean()),
        },
        "config": cfg.to_dict(),
        "results": results,
    }
    save_json(summary, out_dir / "metrics.json")
    save_json(cfg.to_dict(), out_dir / "config.json")
    _write_summary_md(summary, out_dir)
    LOG.info("Artifacts written to %s", out_dir)
    return summary


def _result_to_plain(res) -> dict[str, Any]:
    return {
        "mae": round(float(res.mae), 6) if np.isfinite(res.mae) else None,
        "rmse": round(float(res.rmse), 6) if np.isfinite(res.rmse) else None,
        "spatial_correlation": round(float(res.spatial_correlation), 6)
        if np.isfinite(res.spatial_correlation) else None,
        "n_valid_cells": int(res.n_valid_cells),
    }


def _write_summary_md(summary: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Sea-ice training summary",
        "",
        f"- Created (UTC): `{summary['created_utc']}`",
        f"- Demo-only run: `{str(summary['demo_only']).upper()}`",
    ]
    if summary.get("disclaimer"):
        lines += ["", "> **" + summary["disclaimer"] + "**"]
    lines += [
        "",
        "## Dataset",
        f"- Samples: {summary['dataset']['n_samples']} "
        f"(train {summary['dataset']['samples_train']} / "
        f"val {summary['dataset']['samples_val']} / test {summary['dataset']['samples_test']})",
        f"- Grid: {summary['dataset']['grid']['H']}x{summary['dataset']['grid']['W']} "
        f"(spatial step {summary['config']['spatial_step']})",
        "",
        "## Model A - persistence (test split)",
        f"- MAE: **{summary['results']['persistence']['mae']}**",
        f"- RMSE: **{summary['results']['persistence']['rmse']}**",
        f"- Spatial correlation: **{summary['results']['persistence']['spatial_correlation']}**",
    ]
    if "model_b" in summary["results"]:
        lines += [
            "",
            "## Model B - random forest (test split)",
            f"- MAE: **{summary['results']['model_b']['mae']}**",
            f"- RMSE: **{summary['results']['model_b']['rmse']}**",
            f"- Spatial correlation: **{summary['results']['model_b']['spatial_correlation']}**",
        ]
    if "model_c" in summary["results"]:
        lines += [
            "",
            "## Model C - ConvLSTM (test split)",
            f"- MAE: **{summary['results']['model_c']['mae']}**",
            f"- RMSE: **{summary['results']['model_c']['rmse']}**",
            f"- Spatial correlation: **{summary['results']['model_c']['spatial_correlation']}**",
            f"- Best epoch: **{summary['results']['model_c']['best_epoch']}**",
        ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train sea-ice forecasting models.")
    p.add_argument("--out", default=TrainConfig.out_dir, help="output directory")
    p.add_argument("--demo", action="store_true", help="allow training on synthetic demo data")
    p.add_argument("--model", choices=["convlstm", "random_forest", "persistence"],
                   default=TrainConfig.model)
    p.add_argument("--with-rf", action="store_true", help="also train random forest (Model B)")
    p.add_argument("--sea-ice", default=TrainConfig.sea_ice)
    p.add_argument("--ocean", default=TrainConfig.ocean)
    p.add_argument("--weather", default=TrainConfig.weather)
    p.add_argument("--input-steps", type=int, default=TrainConfig.input_steps)
    p.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    p.add_argument("--spatial-step", type=int, default=TrainConfig.spatial_step)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--hidden-channels", type=int, default=TrainConfig.hidden_channels)
    p.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    p.add_argument("--no-sigmoid", action="store_true", help="raw-logit output head")
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default=TrainConfig.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--rf-estimators", type=int, default=TrainConfig.rf_n_estimators)
    p.add_argument("--rf-subsample", type=int, default=TrainConfig.rf_subsample)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    return TrainConfig(
        out_dir=args.out, demo=args.demo, model=args.model, with_rf=args.with_rf,
        sea_ice=args.sea_ice, ocean=args.ocean, weather=args.weather,
        input_steps=args.input_steps, horizon=args.horizon, spatial_step=args.spatial_step,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden_channels=args.hidden_channels, early_stop_patience=args.early_stop_patience,
        sigmoid_output=not args.no_sigmoid, seed=args.seed, device=args.device,
        rf_n_estimators=args.rf_estimators, rf_subsample=args.rf_subsample,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    configured_logger(verbose=False)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
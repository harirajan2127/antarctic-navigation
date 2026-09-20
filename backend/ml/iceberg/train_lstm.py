"""Train the PyTorch LSTM iceberg position model.

Usage (from backend/)::

    python -m ml.iceberg.train_lstm --seq-len 5 --horizon 4 --epochs 60 --out models/iceberg/run

Outputs into the run directory:
  * lstm.pt            best state_dict (early stopping)
  * lstm_last.pt       last-epoch state_dict
  * scalers.joblib     train-only feature scalers
  * loss_history.csv   per-epoch training log
  * loss_curve.png     validation RMSE curve
  * run.json           machine-readable fingerprint + train metrics
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _plot_loss(history: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["epoch"], df["train_rmse_km"], label="train RMSE", color="#1f77b4")
        ax.plot(df["epoch"], df["val_rmse_km"], label="val RMSE", color="#d62728")
        ax.set_xlabel("epoch")
        ax.set_ylabel("position error (km)")
        ax.set_title("LSTM training curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception:
        LOG.warning("Could not render loss curve (matplotlib missing?)", exc_info=True)


def main(argv: list[str] | None = None) -> None:
    from .cli_common import add_data_args, resolve_data, save_run_meta
    from .lstm_model import train_loop
    from .utils import configured_logger, save_json

    parser = argparse.ArgumentParser(description="Train LSTM iceberg trajectory model")
    add_data_args(parser)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    ctx = resolve_data(args)
    ds, splits, demo, out = ctx["ds"], ctx["splits"], ctx["demo_data"], ctx["out_dir"]
    configured_logger(args.verbose)

    ds.fit_scalers(splits["train"])
    ds.normalize()
    joblib.dump({"scalers": ds.scalers, "seq_len": ds.seq_len, "horizon": ds.horizon}, out / "scalers.joblib")

    algo = train_loop(
        ds,
        splits["train"],
        splits["val"],
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        out_dir=out,
    )
    history = algo["history"]
    pd.DataFrame(history).to_csv(out / "loss_history.csv", index=False)
    _plot_loss(history, out / "loss_curve.png")

    best = best_mae(ds, splits, out, algo)
    train_metrics = {
        "best_epoch": algo["best_epoch"],
        "best_val_rmse_km": round(algo["best_val_rmse_km"], 6),
        "best_val_mae_km": round(best, 6),
        "n_train": algo["n_train"],
        "n_val": algo["n_val"],
        "device": algo["device"],
        "final_epochs_trained": len(history),
    }
    save_run_meta(
        out,
        ds,
        demo,
        model="lstm",
        metrics={"train": train_metrics},
        tracks_path=args.tracks,
    )

    save_json({"feature_cols": _feature_cols()}, out / "feature_cols.json")
    LOG.info("LSTM done. best val RMSE %.4f km, MAE %.4f km", algo["best_val_rmse_km"], best)


def best_mae(ds, splits, out, algo) -> float:
    from .lstm_model import predict_batch_km

    model = algo["model"]
    Xva, Yva = ds.seq_tensor(splits["val"])
    pred = predict_batch_km(model, Xva.numpy(), device="cpu")
    return float(np.abs(pred - Yva.numpy()).mean())


def _feature_cols():
    from .trajectory_dataset import FEATURE_COLS

    return FEATURE_COLS


if __name__ == "__main__":
    main()
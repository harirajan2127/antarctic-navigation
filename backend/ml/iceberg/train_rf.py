"""Train the Random Forest iceberg position model.

Usage (from backend/)::

    python -m ml.iceberg.train_rf --seq-len 5 --horizon 4 --out models/iceberg/run

Baselines are *not* trained here; they are evaluated on the same held-out data
by ``evaluate.py`` so skill numbers stay honest.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    from .cli_common import add_data_args, resolve_data, save_run_meta
    from .random_forest_model import RandomForestTrajectoryModel
    from .utils import set_seed

    parser = argparse.ArgumentParser(description="Train RandomForest iceberg trajectory model")
    add_data_args(parser)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    args = parser.parse_args(argv)

    ctx = resolve_data(args)
    ds, splits, demo, out = ctx["ds"], ctx["splits"], ctx["demo_data"], ctx["out_dir"]
    set_seed(args.seed)

    ds.fit_scalers(splits["train"])
    ds.normalize()
    joblib.dump({"scalers": ds.scalers, "seq_len": ds.seq_len, "horizon": ds.horizon}, out / "scalers.joblib")

    Xtr, Ytr = ds.tabular(splits["train"])
    Xva, Yva = ds.tabular(splits["val"])
    model = RandomForestTrajectoryModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    model.fit(Xtr, Ytr)

    pred_va = model.predict(Xva)
    err = np.sqrt(((pred_va - Yva) ** 2).mean(axis=1))
    train_err = np.sqrt(((model.predict(Xtr) - Ytr) ** 2).mean(axis=1))
    train_metrics = {
        "train_position_error_km_mean": float(train_err.mean()),
        "train_position_error_km_median": float(np.median(train_err)),
        "val_position_error_km_mean": float(err.mean()),
        "val_position_error_km_median": float(np.median(err)),
        "val_n": int(len(Yva)),
    }
    model.save(out / "rf.joblib")
    save_run_meta(
        out,
        ds,
        demo,
        model="random_forest",
        metrics={"train": train_metrics},
        tracks_path=args.tracks,
    )
    LOG.info("val mean position error: %.2f km (median %.2f)", err.mean(), np.median(err))


if __name__ == "__main__":
    main()
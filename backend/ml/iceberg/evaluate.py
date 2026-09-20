"""Chronological evaluation of iceberg trajectory models.

Evaluates persistence, current-drift, Random Forest and LSTM on **held-out**
(chronologically latest) observations. For every test track, forecasts are
started from the tail of the track and compared against ground-truth fixes at
the times that actually occur, so each error is bucketable into:
  24 h error   (1-36 h ahead)
  48 h error   (36-60 h ahead)
  72 h error   (60-84 h ahead, only when the track supports it)

Reports per model:
  * latitude / longitude RMSE (degrees),
  * mean / median haversine position error (km and nautical miles),
  * 24 / 48 / 72 h buckets.

No metric is invented: anything a model cannot be scored on is omitted.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _load_ds(run_json: Path, tracks: Path, joblib_scalers: Path):
    from .trajectory_dataset import TrajectoryDataset
    from .utils import load_json

    cfg = load_json(run_json)
    seq_len = int(cfg.get("seq_len", 5))
    horizon = int(cfg.get("horizon_steps", 4))
    ds = TrajectoryDataset.from_features(tracks, seq_len=seq_len, horizon=horizon)
    stats = joblib.load(joblib_scalers)["scalers"]
    ds.scalers = stats
    ds.normalize(stats)
    return ds


def _valid_window(track, i, seq_len) -> bool:
    env = [track.uo, track.vo, track.u10, track.v10]
    lo = max(0, i - seq_len + 1)
    for arr in env:
        if not np.isfinite(arr[lo:i + 1]).all():
            return False
    return True


def _make_predictor(kind: str, ds, out_dir: Path):
    if kind == "random_forest":
        from .random_forest_model import RandomForestTrajectoryModel

        model = RandomForestTrajectoryModel.load(out_dir / "rf.joblib")
        return lambda seq: model.predict(seq.reshape(1, -1))
    if kind == "lstm":
        from .lstm_model import LSTMPositionModel, predict_batch_km

        model = LSTMPositionModel(input_size=ds.seq.shape[-1])
        model.load_state_dict(torch_load(out_dir / "lstm.pt"))
        return lambda seq: predict_batch_km(model, seq, device="cpu")
    raise ValueError(kind)


def torch_load(path):
    import torch

    return torch.load(Path(path), map_location="cpu")


def _score_off(preds_pts, actual_pts):
    """preds_pts/actual_pts: lists of (lat, lon)."""
    from .utils import haversine_km

    if not preds_pts or not actual_pts:
        return None
    lat_err = np.array([p[0] - a[0] for p, a in zip(preds_pts, actual_pts)])
    lon_err = np.array([p[1] - a[1] for p, a in zip(preds_pts, actual_pts)])
    dist_km = np.array([haversine_km(*p, *a) for p, a in zip(preds_pts, actual_pts)])
    return {
        "n": int(len(dist_km)),
        "lat_rmse_deg": float(np.sqrt((lat_err**2).mean())),
        "lon_rmse_deg": float(np.sqrt((lon_err**2).mean())),
        "position_error_km_mean": float(dist_km.mean()),
        "position_error_km_median": float(np.median(dist_km)),
        "position_error_nm_mean": float(dist_km.mean() / 1.852),
    }


def _bucket(hours: float) -> str | None:
    if hours <= 36:
        return "24h"
    if hours <= 60:
        return "48h"
    if hours <= 84:
        return "72h"
    return None


def run(out_dir: Path, tracks: Path | None):
    from .baselines import CurrentDriftModel, PersistenceModel, RecursiveForecaster
    from .cli_common import DEFAULT_TRACKS
    from .utils import load_json, save_json, save_md, DISCLAIMER_DEMO

    if tracks is None:
        tracks = DEFAULT_TRACKS
    cfg = load_json(out_dir / "run.json")
    ds = _load_ds(out_dir / "run.json", tracks, out_dir / "scalers.joblib")
    seq_len = ds.seq_len
    max_h = 12
    seq = []

    predictors = {
        "persistence": PersistenceModel(ds),
        "current_drift": CurrentDriftModel(ds),
    }
    rf_pred = _make_predictor("random_forest", ds, out_dir) if (out_dir / "rf.joblib").exists() else None
    lstm_pred = _make_predictor("lstm", ds, out_dir) if (out_dir / "lstm.pt").exists() else None
    if rf_pred is not None:
        predictors["random_forest"] = RecursiveForecaster(ds, rf_pred)
    if lstm_pred is not None:
        predictors["lstm"] = RecursiveForecaster(ds, lstm_pred)

    n_test = 0
    for iceberg_id in ds.track_order:
        tr = ds.track(iceberg_id)
        n = len(tr)
        if n < seq_len + max_h:
            continue
        n_test += 1
        # Rolling trials over the tail so 24/48/72 h are all populated.
        first_start = n - seq_len - max_h
        for start in range(first_start, n - seq_len + 1):
            i = start + seq_len - 1
            if not _valid_window(tr, i, seq_len):
                continue
            maxk = min(max_h, n - i - 1)
            if maxk <= 0:
                continue
            for model_name, eng in predictors.items():
                if eng is None:
                    continue
                try:
                    fwd = eng.forecast(tr, i, maxk)
                except Exception:
                    LOG.warning("model %s failed at %s", model_name, iceberg_id, exc_info=True)
                    continue
                if not fwd:
                    continue
                for k in range(1, maxk + 1):
                    hit = i + k
                    if hit >= n or not np.isfinite([tr.lat[hit], tr.lon[hit]]).all():
                        break
                    hours = float((tr.times[hit] - tr.times[i]) / np.timedelta64(1, "h"))
                    bucket = _bucket(hours)
                    if bucket is None:
                        continue
                    actual = (tr.lat[hit], tr.lon[hit])
                    pred = (fwd[k - 1]["lat"], fwd[k - 1]["lon"])
                    seq.append({
                        "model": model_name, "iceberg": iceberg_id,
                        "bucket": bucket, "hours": hours,
                        "start_time": str(tr.times[i]),
                        "pred_lat": pred[0], "pred_lon": pred[1],
                        "act_lat": actual[0], "act_lon": actual[1], "k": k,
                    })

    df = pd.DataFrame(seq)
    if df.empty:
        LOG.error("No evaluation trials could be produced.")
        return None

    by_model = {}
    per_bucket = {}
    for model_name, g in df.groupby("model"):
        m = _score_off(list(zip(g.pred_lat, g.pred_lon)), list(zip(g.act_lat, g.act_lon)))
        by_model[model_name] = m
        buckets = {}
        for bucket, gg in g.groupby("bucket"):
            buckets[bucket] = _score_off(list(zip(gg.pred_lat, gg.pred_lon)), list(zip(gg.act_lat, gg.act_lon)))
        per_bucket[model_name] = buckets

    out = {
        "data": {"n_test_tracks": n_test, "n_trials": int(len(df)), "model": cfg.get("model")},
        "models": by_model,
        "per_horizon_bucket": per_bucket,
        "demo_only": bool(cfg.get("demo_only", False)),
        "note_missing_models": [m for m in ("persistence", "current_drift", "random_forest", "lstm") if m not in by_model],
    }
    evals_dir = out_dir / "evaluation"
    evals_dir.mkdir(parents=True, exist_ok=True)
    save_json(out, evals_dir / "metrics.json")
    save_md(_render(out), evals_dir / "metrics.md")
    _plot_bars(per_bucket, evals_dir / "error_by_horizon.png")
    if cfg.get("demo_only"):
        save_md(DISCLAIMER_DEMO, evals_dir / "disclaimer.txt")
    LOG.info("evaluation: %d trials across %d test tracks", len(df), n_test)
    return out


def _render(out) -> str:
    lines = ["# Iceberg trajectory evaluation (chronological)", "", "## Models — overall position error (km)", ""]
    for m, s in out["models"].items():
        if s is None:
            lines.append(f"- **{m}**: no trials")
            continue
        lines.append(
            f"- **{m}**: {s['position_error_km_mean']:.2f} km mean | "
            f"{s['position_error_km_median']:.2f} km median | "
            f"lat RMSE {s['lat_rmse_deg']:.4f}° | lon RMSE {s['lon_rmse_deg']:.4f}° | n={s['n']}"
        )
    lines += ["", "## Errors by forecast horizon (km)", ""]
    for m, buckets in out["per_horizon_bucket"].items():
        line = f"- **{m}**: "
        line += " | ".join(f"{b}: {s['position_error_km_mean']:.2f}" for b, s in sorted(buckets.items()) if s)
        lines.append(line)
    lines.append("")
    if out.get("demo_only"):
        lines += ["> **DEMO MODE** — metrics are code-path checks on synthetic data, not skill.", ""]
    lines.append(f"_Run with {out['data']['n_trials']} trials on {out['data']['n_test_tracks']} test tracks._")
    return "\n".join(lines)


def _plot_bars(per_bucket, path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models, buckets = list(per_bucket), sorted({b for bs in per_bucket.values() for b in bs})
        x = np.arange(len(buckets))
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, m in enumerate(models):
            vals = [per_bucket[m][b]["position_error_km_mean"] if per_bucket[m].get(b) else np.nan for b in buckets]
            ax.plot(x, vals, marker="o", label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_ylabel("mean position error (km)")
        ax.set_xlabel("forecast horizon")
        ax.set_title("Iceberg trajectory forecast error by horizon")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception:
        LOG.warning("error_by_horizon plot failed", exc_info=True)


def main(argv: list[str] | None = None) -> None:
    from .cli_common import configured_logger

    p = argparse.ArgumentParser(description="Evaluate iceberg trajectory models")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "models" / "iceberg" / "run"))
    p.add_argument("--tracks", default=None, help="CSV used for training (defaults to feature_table.csv)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    configured_logger(args.verbose)
    run(Path(args.out), Path(args.tracks) if args.tracks else None)


if __name__ == "__main__":
    main()

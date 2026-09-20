"""Predict an iceberg trajectory forward in time.

Usage (from backend/)::

    python -m ml.iceberg.predict --iceberg-id DEMO-B000 --steps 4 --model random_forest
    python -m ml.iceberg.predict --iceberg-id DEMO-B000 --steps 8 --model lstm

Output: ``models/iceberg/run/predictions/prediction_{id}_{steps}steps.json`` and
``...csv`` containing the current location, forecast horizon, predicted
track (recursive multi-step), predicted movement, distance travelled and an
honest uncertainty statement (none implemented => stated explicitly).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _fmt(v) -> float:
    return float(np.round(v, 6))


def predict(
    out_dir: Path,
    iceberg_id: str,
    steps: int,
    model_kind: str,
    tracks: Path | None = None,
    compare_baselines: bool = False,
) -> dict:
    from .baselines import CurrentDriftModel, PersistenceModel, RecursiveForecaster
    from .cli_common import DEFAULT_TRACKS
    from .utils import load_json, save_json, haversine_km, DISCLAIMER_TRAJECTORY

    cfg = load_json(out_dir / "run.json")
    if tracks is None:
        tracks = DEFAULT_TRACKS

    from .trajectory_dataset import TrajectoryDataset

    ds = TrajectoryDataset.from_features(tracks, seq_len=int(cfg.get("seq_len", 5)),
                                         horizon=int(cfg.get("horizon_steps", 4)))
    import joblib

    ds.scalers = joblib.load(out_dir / "scalers.joblib")["scalers"]

    if iceberg_id not in ds.tracks:
        raise KeyError(
            f"Unknown iceberg_id '{iceberg_id}'. Available tracks: "
            f"{', '.join(sorted(ds.tracks))}"
        )
    tr = ds.track(iceberg_id)
    start_idx = len(tr) - ds.seq_len
    if start_idx < 0:
        raise ValueError("Track too short for the configured sequence length.")

    from .lstm_model import LSTMPositionModel, predict_batch_km
    from .random_forest_model import RandomForestTrajectoryModel

    engines = {}
    if model_kind in ("random_forest", "both") and (out_dir / "rf.joblib").exists():
        m = RandomForestTrajectoryModel.load(out_dir / "rf.joblib")
        engines["random_forest"] = RecursiveForecaster(ds, lambda s: m.predict(s.reshape(1, -1)))
    if model_kind in ("lstm", "both") and (out_dir / "lstm.pt").exists():
        m = LSTMPositionModel(input_size=ds.seq.shape[-1])
        m.load_state_dict(torch_load(out_dir / "lstm.pt"))
        engines["lstm"] = RecursiveForecaster(ds, lambda s: predict_batch_km(m, s, device="cpu"))

    if not engines:
        raise ValueError(
            f"Model kind '{model_kind}' has no saved checkpoint in {out_dir}. "
            "Train it first (train_rf.py / train_lstm.py)."
        )
    if compare_baselines:
        engines["persistence"] = PersistenceModel(ds)
        engines["current_drift"] = CurrentDriftModel(ds)

    current = {"time": str(tr.times[start_idx + ds.seq_len - 1]),
               "lat": _fmt(tr.lat[start_idx + ds.seq_len - 1]),
               "lon": _fmt(tr.lon[start_idx + ds.seq_len - 1])}
    dt_h = tr.median_dt_hours if np.isfinite(tr.median_dt_hours) else 6.0

    forecasts = {}
    for name, eng in engines.items():
        fwd = eng.forecast(tr, start_idx + ds.seq_len - 1, steps)
        pts = [{"time": p["time"], "hours_ahead": _fmt(p["hours_from_current"]),
                "lat": _fmt(p["lat"]), "lon": _fmt(p["lon"]),
                "distance_km_from_prev": _fmt(p["distance_km_from_prev"])} for p in fwd]
        dist = sum(p["distance_km_from_prev"] for p in pts)
        bearing = None
        if len(pts) >= 1:
            bearing = _bearing(current["lat"], current["lon"], pts[-1]["lat"], pts[-1]["lon"])
        forecasts[name] = {
            "predicted_track": pts,
            "distance_travelled_km": _fmt(dist),
            "net_displacement_km": _fmt(haversine_km(float(current["lat"]), float(current["lon"]),
                                                      float(pts[-1]["lat"]), float(pts[-1]["lon"]))),
            "final_bearing_deg": bearing,
        }

    result = {
        "iceberg_id": iceberg_id,
        "current_location": current,
        "forecast_horizon": {"steps": steps, "hours": _fmt(steps * dt_h), "step_dt_hours": _fmt(dt_h)},
        "prediction_uncertainty": {
            "type": "not_implemented",
            "note": "No distributional/uncertainty model is implemented. "
                    "Point forecasts only; treat as advisory.",
        },
        "models": forecasts,
        "demo_only": bool(cfg.get("demo_only", False)),
        "disclaimer": DISCLAIMER_TRAJECTORY,
    }

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, pred_dir / f"prediction_{iceberg_id}_{steps}steps.json")
    rows = []
    for name, f in forecasts.items():
        for p in f["predicted_track"]:
            rows.append({"model": name, **p})
    pd.DataFrame(rows).to_csv(pred_dir / f"prediction_{iceberg_id}_{steps}steps.csv", index=False)
    return result


def _bearing(lat1, lon1, lat2, lon2):
    from .utils import bearing_deg

    return _fmt(bearing_deg(lat1, lon1, lat2, lon2))


def torch_load(path):
    import torch

    return torch.load(Path(path), map_location="cpu")


def main(argv: list[str] | None = None) -> None:
    from .cli_common import configured_logger

    p = argparse.ArgumentParser(description="Forecast an iceberg trajectory")
    p.add_argument("--iceberg-id", required=True)
    p.add_argument("--steps", type=int, default=4, help="steps ahead (step ~ 6 h in demo data)")
    p.add_argument("--model", choices=["random_forest", "lstm", "both"], default="random_forest")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "models" / "iceberg" / "run"))
    p.add_argument("--tracks", default=None)
    p.add_argument("--compare-baselines", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    configured_logger(args.verbose)
    r = predict(Path(args.out), args.iceberg_id, args.steps, args.model,
                Path(args.tracks) if args.tracks else None, args.compare_baselines)
    print(_render_cli(r))


def _render_cli(r: dict) -> str:
    lines = [
        f"Iceberg {r['iceberg_id']}  @ {r['current_location']['time']}",
        f"  current: {r['current_location']['lat']:.4f}, {r['current_location']['lon']:.4f}",
        f"  horizon: {r['forecast_horizon']['hours']:g} h ({r['forecast_horizon']['steps']} steps)",
    ]
    for name, f in r["models"].items():
        lines.append(f"  [{name}] final {f['predicted_track'][-1]['lat']:.4f}, "
                     f"{f['predicted_track'][-1]['lon']:.4f} | "
                     f"travelled {f['distance_travelled_km']:.2f} km | "
                     f"bearing {f['final_bearing_deg']} deg")
    lines.append("  uncertainty: not implemented (advisory only)")
    if r.get("demo_only"):
        lines.append("  DEMO MODE: synthetic data — not a real forecast.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

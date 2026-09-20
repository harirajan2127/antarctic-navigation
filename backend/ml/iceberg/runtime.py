"""In-memory inference runtime for trained iceberg models.

Bridges ``ml.iceberg`` (RF / LSTM training pipeline) and the live REST service.
Handles:

* loading the artifacts saved by ``train_rf.py`` / ``train_lstm.py``
  (``rf.joblib`` / ``lstm.pt`` + ``scalers.joblib`` + ``run.json``),
* recursive multi-step look-ahead forecasting for one iceberg, and
* an honest single-position contract for the API response.

If artifacts are missing or an iceberg is not in the training feature table,
callers must keep using the persistence baseline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("dss.ml.iceberg.runtime")

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = BACKEND_DIR / "models" / "iceberg" / "run_real"
FEATURE_TABLE = BACKEND_DIR / "data" / "processed" / "features" / "feature_table.csv"
REAL_FEATURE_TABLE = BACKEND_DIR / "data" / "processed" / "features" / "feature_table_real.csv"
LEGACY_RUN_DIR = BACKEND_DIR / "models" / "iceberg" / "run"

# Demo-tracks step cadence used to translate horizon hours -> forecast steps.
_REFERENCE_STEP_HOURS = 6.0


@dataclass
class IcebergRuntime:
    """Lazily-loaded trained iceberg model runtime (one per model kind)."""

    model_kind: str
    out_dir: Path
    available: bool = False
    loaded: bool = False
    error: str | None = None
    _cfg: dict = field(default_factory=dict)
    _ds: object = None
    _engine: object = None
    _custom_model: object = None
    _custom_scaler: object = None
    _custom_features: list[str] = field(default_factory=list)
    _custom_tracks: object = None

    def _load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        try:
            cfg_path = self.out_dir / "run.json"
            if not cfg_path.exists() and self.out_dir != LEGACY_RUN_DIR:
                alt_dir = LEGACY_RUN_DIR if LEGACY_RUN_DIR.exists() else None
                if alt_dir is not None:
                    self.out_dir = alt_dir
                    cfg_path = self.out_dir / "run.json"
            if not cfg_path.exists():
                self.error = f"No run.json in {self.out_dir}"
                return
            self._cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ckpt = self.out_dir / "rf.joblib" if self.model_kind == "random_forest" \
                else (self.out_dir / "lstm.pt")
            if not ckpt.exists():
                self.error = f"No checkpoint for '{self.model_kind}' in {self.out_dir}"
                return

            # The real-data trainer uses a compact tabular contract rather than
            # the legacy 12-channel sequence contract used by demo artifacts.
            # Load that contract directly so a real model cannot be mistaken for
            # a demo model or silently fall back because of schema differences.
            preprocessor_path = self.out_dir / "preprocessor.joblib"
            if preprocessor_path.exists():
                import joblib
                import pandas as pd

                preprocessor = joblib.load(preprocessor_path)
                self._custom_model = joblib.load(ckpt)
                self._custom_scaler = preprocessor["scaler"]
                self._custom_features = list(preprocessor["feature_columns"])
                tracks_path = Path(self._cfg["tracks_resolved"])
                if not tracks_path.is_absolute():
                    tracks_path = BACKEND_DIR / tracks_path
                if not tracks_path.is_file():
                    packaged_tracks = REAL_FEATURE_TABLE if REAL_FEATURE_TABLE.exists() else FEATURE_TABLE
                    if packaged_tracks.is_file():
                        LOG.warning(
                            "Ignoring unavailable recorded track path %s; using packaged feature table %s",
                            tracks_path,
                            packaged_tracks,
                        )
                        tracks_path = packaged_tracks
                    else:
                        self.error = f"No feature-table tracks available for '{self.model_kind}'."
                        return
                tracks = pd.read_csv(tracks_path, parse_dates=["timestamp"])
                self._custom_tracks = tracks.sort_values(["iceberg_id", "timestamp"])
                self.available = True
                return

            from ml.iceberg.baselines import RecursiveForecaster
            from ml.iceberg.lstm_model import LSTMPositionModel, predict_batch_km
            from ml.iceberg.random_forest_model import RandomForestTrajectoryModel
            from ml.iceberg.trajectory_dataset import TrajectoryDataset

            import joblib

            preferred_tracks = REAL_FEATURE_TABLE if REAL_FEATURE_TABLE.exists() else FEATURE_TABLE
            tracks = preferred_tracks if preferred_tracks.exists() else self._cfg.get("tracks_resolved")
            if not tracks or not str(tracks).endswith(".csv"):
                self.error = "No feature-table tracks available for the trained model."
                return

            ds = TrajectoryDataset.from_features(
                str(tracks),
                seq_len=int(self._cfg.get("seq_len", 5)),
                horizon=int(self._cfg.get("horizon_steps", 4)),
            )
            scalers = joblib.load(self.out_dir / "scalers.joblib")
            ds.scalers = scalers.get("scalers") if isinstance(scalers, dict) else scalers
            self._ds = ds

            if self.model_kind == "random_forest":
                m = RandomForestTrajectoryModel.load(self.out_dir / "rf.joblib")
                self._engine = RecursiveForecaster(
                    ds, lambda s: m.predict(np.asarray(s).reshape(1, -1))
                )
            else:  # lstm
                m = LSTMPositionModel(input_size=ds.seq.shape[-1])
                import torch

                state = torch.load(self.out_dir / "lstm.pt", map_location="cpu")
                m.load_state_dict(
                    state["model_state"] if isinstance(state, dict) and "model_state" in state
                    else state
                )
                m.eval()
                self._engine = RecursiveForecaster(
                    ds, lambda s: predict_batch_km(m, s, device="cpu")
                )
            self.available = True
        except Exception as exc:  # noqa: BLE001 - runtime must never crash 500s
            LOG.exception("Failed to load iceberg trained model '%s': %s", self.model_kind, exc)
            self.error = str(exc)

    @property
    def summary(self) -> dict:
        out = {
            "name": self.model_kind,
            "available": self.available,
            "trained_on_demo": bool(self._cfg.get("demo_only", True)) if self._cfg else None,
        }
        if self.error:
            out["error"] = self.error
        return out

    def step_hours_for(self, tr) -> float:
        dt = getattr(tr, "median_dt_hours", None) if tr is not None else None
        if dt is None or not float(dt) == float(dt):  # NaN guard
            return _REFERENCE_STEP_HOURS
        return max(2.0, float(dt))

    def predict_iceberg(self, iceberg_id: str, horizon_hours: int) -> dict | None:
        """Return the predicted track for ``iceberg_id`` or ``None`` if unusable."""
        self._load()
        if self._custom_model is not None:
            return self._predict_custom(iceberg_id, horizon_hours)
        if not self.available or self._ds is None or self._engine is None:
            return None
        ds = self._ds
        if iceberg_id not in ds.tracks:
            return None
        tr = ds.track(iceberg_id)
        start_idx = len(tr) - ds.seq_len
        if start_idx < 0:
            return None
        steps = max(1, round(horizon_hours / self.step_hours_for(tr)))
        fwd = self._engine.forecast(tr, start_idx + ds.seq_len - 1, steps)
        if not fwd:
            return None
        return {
            "iceberg_id": iceberg_id,
            "current": {
                "lat": float(tr.lat[start_idx + ds.seq_len - 1]),
                "lon": float(tr.lon[start_idx + ds.seq_len - 1]),
            },
            "steps": steps,
            "track": [
                {
                    "time": p.get("time"),
                    "hours_ahead": float(p.get("hours_from_current", 0.0)),
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                    "distance_km_from_prev": float(p.get("distance_km_from_prev", 0.0)),
                }
                for p in fwd
            ],
            "final": {
                "lat": float(fwd[-1]["lat"]),
                "lon": float(fwd[-1]["lon"]),
            },
            "demo_only": bool(self._cfg.get("demo_only", True)),
            "confidence_note": (
                "Trained {} model. Point forecast only; no distributional "
                "uncertainty is implemented.".format(self.model_kind)
            ),
        }

    def _predict_custom(self, iceberg_id: str, horizon_hours: int) -> dict | None:
        import pandas as pd

        tracks = self._custom_tracks
        if tracks is None:
            return None
        history = tracks[tracks["iceberg_id"].astype(str) == str(iceberg_id)].copy()
        if history.empty:
            return None
        row = history.iloc[-1].copy()
        current_lat = float(row["latitude"])
        current_lon = float(row["longitude"])
        step_hours = max(1.0, float(row.get("target_horizon_hours", 24.0)))
        steps = max(1, round(float(horizon_hours) / step_hours))
        forecast = []
        for step in range(steps):
            values = pd.DataFrame([{name: row[name] for name in self._custom_features}])
            scaled = self._custom_scaler.transform(values)
            delta = self._custom_model.predict(scaled)[0]
            next_lat = current_lat + float(delta[0])
            next_lon = current_lon + float(delta[1])
            forecast.append({
                "time": str(pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=step_hours * (step + 1))),
                "hours_from_current": step_hours * (step + 1),
                "lat": next_lat,
                "lon": next_lon,
                "distance_km_from_prev": 0.0,
            })
            current_lat, current_lon = next_lat, next_lon
            row["prev_latitude"] = current_lat
            row["prev_longitude"] = current_lon
            row["latitude"] = current_lat
            row["longitude"] = current_lon
        return {
            "iceberg_id": str(iceberg_id),
            "current": {"lat": float(history.iloc[-1]["latitude"]), "lon": float(history.iloc[-1]["longitude"])},
            "steps": steps,
            "track": forecast,
            "final": {"lat": forecast[-1]["lat"], "lon": forecast[-1]["lon"]},
            "demo_only": bool(self._cfg.get("demo_only", True)),
            "confidence_note": "Trained random_forest model on real processed observations. Point forecast only; no distributional uncertainty is implemented.",
        }


_runtimes: dict[str, IcebergRuntime] = {}


def get_runtime(model_kind: str, out_dir: Path | str | None = None) -> IcebergRuntime:
    """Return the (cached) trained iceberg runtime for ``model_kind``."""
    d = Path(out_dir or DEFAULT_OUT_DIR)
    key = f"{model_kind}:{d}"
    runtime = _runtimes.get(key)
    if runtime is None:
        ckpt = (d / "rf.joblib") if model_kind == "random_forest" else (d / "lstm.pt")
        if (d / "run.json").exists() and ckpt.exists():
            runtime = IcebergRuntime(model_kind=model_kind, out_dir=d)
            runtime._load()
        else:
            runtime = IcebergRuntime(
                model_kind=model_kind,
                out_dir=d,
                available=False,
                error=f"No trained artifacts for '{model_kind}' in {d}",
            )
        _runtimes[key] = runtime
    return runtime
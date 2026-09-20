"""In-memory inference runtime for trained sea-ice models.

Bridges the offline ML training pipeline (``ml.sea_ice``) and the live REST
service. ``grid_predict`` loads the artifacts saved by ``train.py`` and runs
the *same* inference code path as ``ml.sea_ice.predict``, but returns arrays
in memory instead of writing NetCDF/JSON files.

A trained runtime is only ever reported when the checkpoint and its matching
``config.json`` (and scalers) all exist. If anything is missing or the loading
fails, ``available`` is ``False`` and callers must keep using persistence.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOG = logging.getLogger("dss.ml.sea_ice.runtime")

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = BACKEND_DIR / "models" / "sea_ice" / "run"

# The trained models produced by ``ml.sea_ice.train`` forecast a fixed step
# (daily for the current demo grids). We only claim the runtime for horizons
# that match this step so we never re-label another model's field.
STEP_HOURS = 24


@dataclass
class SeaIceRuntime:
    """Lazily-loaded trained sea-ice model runtime."""

    model_name: str
    out_dir: Path
    available: bool = False
    loaded: bool = False
    error: str | None = None
    cfg: dict = field(default_factory=dict)
    _state: dict | None = None

    def _load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        from ml.sea_ice.dataset import SeaIceDataset
        from ml.sea_ice.model import build_model
        from ml.sea_ice.train import TrainConfig
        from ml.sea_ice.utils import load_scalers

        import torch

        try:
            cfg = json.loads((self.out_dir / "config.json").read_text(encoding="utf-8"))
            self.cfg = TrainConfig.from_dict(cfg).__dict__
            config = dict(cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            paths = {
                "sea_ice": Path(config.get("sea_ice", "")),
                "ocean": Path(config.get("ocean", "")),
                "weather": Path(config.get("weather", "")),
            }
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                self.error = f"Training inputs missing: {', '.join(missing)}"
                return

            ds = SeaIceDataset.from_files(
                paths,
                input_steps=int(config.get("input_steps", 7)),
                horizon=int(config.get("horizon", 1)),
                spatial_step=int(config.get("spatial_step", 1)),
            )
            ds.normalize(load_scalers(self.out_dir / "scalers.joblib"))

            model_name = self.model_name
            if model_name == "random_forest":
                ckpt = self.out_dir / "rf.joblib"
                if not ckpt.exists():
                    self.error = "No rf.joblib checkpoint found."
                    return
                from ml.sea_ice.baselines import load_rf

                model = load_rf(ckpt)
                kind = "random_forest"
            elif model_name == "convlstm":
                ckpt = self.out_dir / "model.pt"
                if not ckpt.exists():
                    self.error = "No model.pt checkpoint found."
                    return
                save = torch.load(ckpt, map_location="cpu", weights_only=False)
                kw = dict(save.get("model_kwargs") or {})
                kw.setdefault("in_channels", ds.X.shape[-1])
                kw.setdefault("hidden_channels", int(config.get("hidden_channels", 16)))
                kw.setdefault("kernel_size", int(config.get("kernel_size", 3)))
                kw.setdefault("sigmoid_output", bool(config.get("sigmoid_output", True)))
                model = build_model("convlstm", kw, device)
                model.load_state_dict(save["model_state"])
                kind = "convlstm"
            else:
                self.error = f"Unsupported trained model: {self.model_name}"
                return

            self._state = {
                "ds": ds,
                "model": model,
                "model_name": kind,
                "demo_only": bool(config.get("demo", True)),
            }
            self.available = True
        except Exception as exc:  # noqa: BLE001 - runtime must never crash 500s
            LOG.exception("Failed to load sea-ice trained model '%s': %s", self.model_name, exc)
            self.error = str(exc)

    @property
    def summary(self) -> dict:
        """Short metadata for API responses (no internals)."""
        base = {
            "model_name": self.model_name,
            "available": self.available,
            "demo_only": bool(self.cfg.get("demo", True)) if self.cfg else None,
            "provided_24h_step": STEP_HOURS,
        }
        if self.error:
            base["error"] = self.error
        return base

    def grid_predict(self) -> dict:
        """Return the next-step concentration field + lon/lat for this model."""
        self._load()
        if not self.available or self._state is None:
            raise RuntimeError(f"Trained sea-ice runtime unavailable: {self.error}")
        ds = self._state["ds"]
        model = self._state["model"]
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            if self._state["model_name"] == "convlstm":
                x = torch.from_numpy(ds.X[-1:]).to(device)
                pred = model.predict_grid(x)[0, :, :, 0].cpu().numpy()
            else:
                x2, _, m2 = ds.tabular_cells(np.array([ds.X.shape[0] - 1]))
                pred_flat = np.clip(np.asarray(model.predict(x2), dtype=float), 0.0, 1.0)
                pred = np.full((ds.lats.size, ds.lons.size), np.nan, dtype=np.float64)
                pred[m2.reshape(ds.lats.size, ds.lons.size)] = pred_flat[m2]

        lat = np.round(np.asarray(ds.lats, dtype=float), 4).tolist()
        lon = np.round(np.asarray(ds.lons, dtype=float), 4).tolist()
        return {
            "concentration": np.asarray(pred, dtype=float).tolist(),
            "lat": lat,
            "lon": lon,
            "model": self._state["model_name"],
            "model_used_real": not self._state["demo_only"],
            "demo_only": self._state["demo_only"],
            "skill_note": (
                "Trained sea-ice model loaded from {} ({} steps input). "
                "Valid only for the model's native horizon.".format(
                    self.out_dir.name, self.cfg.get("input_steps", "?")
                )
            ),
        }


_runtimes: dict[str, SeaIceRuntime] = {}


def get_runtime(model_name: str, out_dir: Path | str | None = None) -> SeaIceRuntime:
    """Return the (cached) trained sea-ice runtime for ``model_name``."""
    key = f"{model_name}:{Path(out_dir or DEFAULT_OUT_DIR)}"
    runtime = _runtimes.get(key)
    if runtime is None:
        d = Path(out_dir or DEFAULT_OUT_DIR)
        # Config + at least one checkpoint must exist or we skip loading.
        if (d / "config.json").exists() and (
            (d / "model.pt").exists() or (d / "rf.joblib").exists()
        ):
            runtime = SeaIceRuntime(model_name=model_name, out_dir=d)
            runtime._load()
        else:
            runtime = SeaIceRuntime(
                model_name=model_name,
                out_dir=d,
                available=False,
                error=f"No trained artifacts found in {d}",
            )
        _runtimes[key] = runtime
    return runtime
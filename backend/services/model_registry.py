"""Model registry — aggregates trained-model status for the UI/API.

Reads the ML artifacts on disk (``models/{sea_ice,iceberg}/run``) and reports,
for both pipelines, which models are trained, when, on which data (demo vs
real), and where the checkpoints are. Pure read-only; never fabricates status.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger("dss.services.model_registry")

BACKEND_DIR = Path(__file__).resolve().parents[1]

SEA_ICE_RUN = BACKEND_DIR / "models" / "sea_ice" / "run"
ICEBERG_RUN = BACKEND_DIR / "models" / "iceberg" / "run_real"
LEGACY_ICEBERG_RUN = BACKEND_DIR / "models" / "iceberg" / "run"
if not ICEBERG_RUN.exists():
    ICEBERG_RUN = LEGACY_ICEBERG_RUN


@dataclass(frozen=True)
class TrainedModel:
    pipeline: str
    name: str
    available: bool
    trained_on_demo: bool | None
    created_utc: str | None
    artifact: str | None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "name": self.name,
            "available": self.available,
            "trained_on_demo": self.trained_on_demo,
            "created_utc": self.created_utc,
            "artifact": self.artifact,
            "error": self.error,
        }


def _sea_ice_models() -> list[TrainedModel]:
    models: list[TrainedModel] = []
    if not SEA_ICE_RUN.exists():
        return models
    cfg = _read_json(SEA_ICE_RUN / "config.json")
    demo = bool(cfg.get("demo", True)) if isinstance(cfg, dict) else None
    created = (cfg or {}).get("created_utc") or _file_mtime(SEA_ICE_RUN / "metrics.json")
    if (SEA_ICE_RUN / "model.pt").exists():
        models.append(
            TrainedModel(
                pipeline="sea_ice",
                name="convlstm",
                available=True,
                trained_on_demo=demo,
                created_utc=created,
                artifact="model.pt",
            )
        )
    if (SEA_ICE_RUN / "rf.joblib").exists():
        models.append(
            TrainedModel(
                pipeline="sea_ice",
                name="random_forest",
                available=True,
                trained_on_demo=demo,
                created_utc=created,
                artifact="rf.joblib",
            )
        )
    return models


def _iceberg_models() -> list[TrainedModel]:
    models: list[TrainedModel] = []
    if not ICEBERG_RUN.exists():
        return models
    run = _read_json(ICEBERG_RUN / "run.json")
    demo = bool(run.get("demo_only", True)) if isinstance(run, dict) else None
    created = (run or {}).get("created_utc") or _file_mtime(ICEBERG_RUN / "run.json")
    if (ICEBERG_RUN / "rf.joblib").exists():
        models.append(
            TrainedModel(
                pipeline="iceberg",
                name="random_forest",
                available=True,
                trained_on_demo=demo,
                created_utc=created,
                artifact="rf.joblib",
            )
        )
    if (ICEBERG_RUN / "lstm.pt").exists():
        models.append(
            TrainedModel(
                pipeline="iceberg",
                name="lstm",
                available=True,
                trained_on_demo=demo,
                created_utc=created,
                artifact="lstm.pt",
            )
        )
    return models


def model_registry() -> dict:
    """Full registry snapshot (no secrets, JSON-safe)."""
    sea = _sea_ice_models()
    ice = _iceberg_models()
    all_models = sea + ice
    real_trained = [m.as_dict() for m in all_models if m.trained_on_demo is False]
    return {
        "models": [m.as_dict() for m in all_models],
        "count": len(all_models),
        "any_real_trained": len(real_trained) > 0,
        "config": {
            "sea_ice_model": settings.SEA_ICE_MODEL,
            "iceberg_model": settings.ICEBERG_MODEL,
        },
        "demo_mode": False,
        "warning": (
            "No real-trained model checkpoint is available in the current runtime."
            if not real_trained and all_models
            else None
        ),
    }


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _file_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except Exception:  # noqa: BLE001
        return None
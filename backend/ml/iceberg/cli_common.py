"""Shared CLI helpers for the iceberg training / evaluation / prediction scripts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .trajectory_dataset import TrajectoryDataset
from .utils import configured_logger, is_demo_dataset, save_json

LOG = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRACKS = BACKEND_DIR / "data" / "processed" / "features" / "feature_table.csv"


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracks",
        default=str(DEFAULT_TRACKS),
        help="CSV with iceberg_id,timestamp,latitude,longitude,uo,vo,u10,v10[,length_nm,width_nm] "
             "(default: preprocessing feature_table.csv)",
    )
    parser.add_argument("--ocean", help="NetCDF ocean grid (uo/vo) for nearest-neighbour lookup", default=None)
    parser.add_argument("--weather", help="NetCDF weather grid (u10/v10) lookup", default=None)
    parser.add_argument("--seq-len", type=int, default=5, help="observations per input window (3-7 typical)")
    parser.add_argument("--horizon", type=int, default=4, help="forecast horizon in steps ahead")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(BACKEND_DIR / "models" / "iceberg" / "run"))
    parser.add_argument("--demo", action="store_true", help="allow training on synthetic/demo data")
    parser.add_argument("--verbose", action="store_true")


def resolve_data(args: argparse.Namespace):
    """Returns dict: ds, splits, demo_data, out_dir."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configured_logger(args.verbose)

    tracks_csv = Path(args.tracks)
    demo_data = is_demo_dataset([tracks_csv, args.ocean, args.weather])
    if not demo_data and tracks_csv.exists():
        try:
            head = pd.read_csv(tracks_csv, usecols=["iceberg_id"], nrows=20)
            demo_data = any("demo" in str(i).lower() for i in head["iceberg_id"])
        except Exception:
            pass

    if demo_data and not args.demo:
        from .utils import explain_data_situation

        explain_data_situation(True, False)  # exits

    if args.ocean and args.weather:
        ds = TrajectoryDataset.from_parts(tracks_csv, args.ocean, args.weather,
                                          seq_len=args.seq_len, horizon=args.horizon)
    else:
        ds = TrajectoryDataset.from_features(tracks_csv, seq_len=args.seq_len,
                                             horizon=args.horizon,
                                             ocean_nc=args.ocean, weather_nc=args.weather)
    splits = ds.split_chronological()
    LOG.info("split sizes: %s", {k: len(v) for k, v in splits.items()})
    return {"ds": ds, "splits": splits, "demo_data": demo_data, "out_dir": out_dir}


def save_run_meta(
    out_dir: Path,
    ds: TrajectoryDataset,
    demo_data: bool,
    model: str,
    metrics: dict,
    extra: dict | None = None,
    tracks_path: str | Path | None = None,
) -> Path:
    """Persist a machine-readable run fingerprint used by evaluate/predict."""
    meta = {
        "model": model,
        "demo_only": bool(demo_data),
        "tracks_resolved": str(Path(tracks_path) if tracks_path else DEFAULT_TRACKS),
        "seq_len": ds.seq_len,
        "horizon_steps": ds.horizon,
        "n_tracks": len(ds.tracks),
        "n_sequences": len(ds.seq_raw),
        "n_features": ds.seq_raw.shape[-1],
        "metrics": metrics,
    }
    if extra:
        meta.update(extra)
    if demo_data:
        from .utils import DISCLAIMER_DEMO

        meta["disclaimer"] = DISCLAIMER_DEMO
    return save_json(meta, out_dir / "run.json")


def std_report(metrics: dict) -> str:
    lines = ["# Iceberg trajectory forecasting — run report", ""]
    for sec in ("data", "train", "test"):
        if sec in metrics:
            lines.append(f"## {sec.title()}")
            for k, v in metrics[sec].items():
                lines.append(f"- {k}: {v}")
            lines.append("")
    for sec in ("baselines", "models"):
        if sec in metrics:
            lines.append(f"## {sec.title()}")
            rows = metrics[sec] if isinstance(metrics[sec], list) else list(metrics[sec].items())
            for row in rows:
                if isinstance(row, dict):
                    lines.append("- " + " | ".join(f"{k}={v}" for k, v in row.items()))
                else:
                    lines.append(f"- {row}")
            lines.append("")
    return "\n".join(lines)
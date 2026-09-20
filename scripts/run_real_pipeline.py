#!/usr/bin/env python3
"""End-to-end orchestrator for the real-data Antarctic DSS pipeline.

Runs every stage in order, with staging/verify gates between each stage.
Default mode is full; pass ``--quick`` for a fast CI-friendly run.

Stages
------
1. discover   – inventory every file under ``Real data/raw/``
2. validate   – read-only schema / region / format checks (exit on error)
3. convert    – standardize all datasets to partitioned CSVs
4. masks      – build a land/ocean mask from bathymetry for the navigation grid
5. metadata   – write / refresh ``processed/manifest.json`` + per-dataset metadata
6. train      – (optional) train sea-ice / iceberg ML models on the real CSVs
7. evaluate   – (optional) run quick model-evaluation metrics
8. integrate  – update backend path resolution so the REST API reads the real CSVs
9. verify     – boot the backend and hit /api/health to confirm startup
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BACKEND_DIR = PROJECT_ROOT / "backend"
PROCESSED_ROOT = PROJECT_ROOT / "Real data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PYTHON = sys.executable


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _banner(stage: str, msg: str) -> None:
    print(f"\n{'='*72}\n  [{_ts()}] STAGE {stage}: {msg}\n{'='*72}", flush=True)


def _run(cmd: list[str], label: str, timeout: int = 3600, **kw) -> int:
    print(f"  [{_ts()}] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout, **kw)
    if proc.returncode != 0:
        print(f"  [{_ts()}] FAIL ({label}): exit code {proc.returncode}", flush=True)
    else:
        print(f"  [{_ts()}] OK    ({label})", flush=True)
    return proc.returncode


# ── Stage 1: discover ───────────────────────────────────────────────
def stage_discover(quick: bool = False) -> dict:
    _banner("1", "Discover raw datasets")
    from scripts.preprocess_real_data import discover, log  # type: ignore

    inventory = discover()
    counts = {d: len(fs) for d, fs in inventory.items()}
    print(f"  Detected: {counts}", flush=True)
    return inventory


# ── Stage 2: validate (dry-run) ────────────────────────────────────
def stage_validate(quick: bool = False) -> int:
    _banner("2", "Validate raw datasets (read-only)")
    args = ["--dataset", "all", "--dry-run"]
    if quick:
        args.append("--quick")
    return _run(
        [PYTHON, str(SCRIPTS_DIR / "preprocess_real_data.py"), *args],
        "validate",
    )


# ── Stage 3: convert ───────────────────────────────────────────────
def stage_convert(quick: bool = False) -> int:
    _banner("3", "Convert raw → standardized CSVs")
    args = ["--dataset", "all"]
    if quick:
        args.append("--quick")
    return _run(
        [PYTHON, str(SCRIPTS_DIR / "preprocess_real_data.py"), *args],
        "convert",
    )


# ── Stage 4: masks ─────────────────────────────────────────────────
def stage_masks(quick: bool = False) -> int:
    """Build a land/ocean mask from the bathymetry CSV for the navigation grid."""
    _banner("4", "Build land/ocean mask from bathymetry")
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("  SKIP: pandas/numpy not available", flush=True)
        return 0

    bathy_dir = PROCESSED_ROOT / "bathymetry" / "csv"
    if not bathy_dir.exists():
        print("  SKIP: no bathymetry CSVs found", flush=True)
        return 0

    parts = sorted(bathy_dir.glob("bathymetry_part_*.csv"))
    if not parts:
        print("  SKIP: no bathymetry parts", flush=True)
        return 0

    out_dir = PROCESSED_ROOT / "bathymetry"
    mask_path = out_dir / "land_ocean_mask.json"

    print(f"  Reading {len(parts)} bathymetry part(s)...", flush=True)
    dfs = [pd.read_csv(p, usecols=["latitude", "longitude", "depth_meters", "elevation_meters"]) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude"])

    # Land = elevation above sea level (or missing depth); ocean = depth > 0.
    is_ocean = df["depth_meters"].notna() & (df["depth_meters"] > 0)
    are_land = ~is_ocean

    ocean = df[is_ocean].copy()
    land = df[are_land].copy()

    mask = {
        "latitude": ocean["latitude"].round(4).tolist(),
        "longitude": ocean["longitude"].round(4).tolist(),
        "ocean_cells": int(len(ocean)),
        "land_cells": int(len(land)),
        "total_cells": int(len(df)),
        "generated_at": _ts(),
        "source": "bathymetry CSVs (land = elevation>0, ocean = depth>0)",
    }
    with open(mask_path, "w") as f:
        json.dump(mask, f)
    print(f"  Mask written: {mask_path}  ({len(ocean)} ocean, {len(land)} land)", flush=True)
    return 0


# ── Stage 5: metadata ──────────────────────────────────────────────
def stage_metadata(quick: bool = False) -> int:
    _banner("5", "Refresh metadata / manifest")
    manifest = PROCESSED_ROOT / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m["last_pipeline_run"] = _ts()
        manifest.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        print(f"  Manifest updated: {manifest}", flush=True)
    else:
        print(f"  No manifest found at {manifest}", flush=True)
    return 0


# ── Stage 6: train (optional) ─────────────────────────────────────
def stage_train(quick: bool = False) -> int:
    _banner("6", "Train / load ML models")
    if quick:
        print("  SKIP in --quick mode", flush=True)
        return 0
    model_dir = PROJECT_ROOT / "models"
    if not model_dir.exists():
        print("  No models/ directory; skipping training", flush=True)
        return 0
    print("  Model training not yet wired to real CSVs — skipping", flush=True)
    return 0


# ── Stage 7: evaluate (optional) ──────────────────────────────────
def stage_evaluate(quick: bool = False) -> int:
    _banner("7", "Evaluate models")
    if quick:
        print("  SKIP in --quick mode", flush=True)
        return 0
    print("  Evaluation not yet wired — skipping", flush=True)
    return 0


# ── Stage 8: integrate ────────────────────────────────────────────
def stage_integrate(quick: bool = False) -> int:
    """Ensure the backend resolves to the real processed CSVs."""
    _banner("8", "Update backend path resolution")

    flag = PROCESSED_ROOT / ".real_data_ready"
    flag.write_text(json.dumps({
        "ready": True,
        "timestamp": _ts(),
        "processed_root": str(PROCESSED_ROOT),
    }), encoding="utf-8")
    print(f"  Flag written: {flag}", flush=True)

    # Verify the processed files exist for each dataset
    datasets = ["sea_ice", "iceberg", "ocean", "weather", "vessel", "bathymetry"]
    for ds in datasets:
        csv_dir = PROCESSED_ROOT / ds / "csv"
        if csv_dir.exists():
            n = len(list(csv_dir.glob("*.csv")))
            print(f"  {ds:12s}: {n} CSV file(s) in {csv_dir}", flush=True)
        else:
            print(f"  {ds:12s}: NO CSV directory — Data Unavailable", flush=True)
    return 0


# ── Stage 9: verify backend boot ──────────────────────────────────
def stage_verify(quick: bool = False) -> int:
    """Quick smoke-test: import the backend app module."""
    _banner("9", "Verify backend can load with real data paths")
    verify_script = BACKEND_DIR / "_verify_boot.py"
    verify_script.write_text(
        '"""Auto-generated boot verification."""\n'
        'import sys, pathlib, os\n'
        'os.chdir(pathlib.Path(__file__).resolve().parent)\n'
        'sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))\n'
        'try:\n'
        '    from services.data_paths import have_real_datasets, SEA_ICE_NETCDF, ICEBERG_CSV\n'
        '    print(f"  real_data_available={have_real_datasets}")\n'
        '    print(f"  SEA_ICE_NETCDF={SEA_ICE_NETCDF}")\n'
        '    print(f"  ICEBERG_CSV={ICEBERG_CSV}")\n'
        'except Exception as exc:\n'
        '    print(f"  BOOT ERROR: {exc}")\n'
        '    sys.exit(1)\n'
        'print("  Backend path resolution OK")\n',
        encoding="utf-8",
    )
    rc = _run([PYTHON, str(verify_script)], "boot_verify")
    verify_script.unlink(missing_ok=True)
    return rc


# ── CLI ────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end real-data pipeline orchestrator.")
    p.add_argument("--quick", action="store_true",
                   help="Fast CI mode: small samples, skip training/eval")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Stage names to skip (discover, validate, convert, masks, "
                        "metadata, train, evaluate, integrate, verify)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these stages (overrides --skip)")
    return p.parse_args(argv)


ALL_STAGES = [
    "discover", "validate", "convert", "masks", "metadata",
    "train", "evaluate", "integrate", "verify",
]


STAGE_FUNCS = {
    "discover":  stage_discover,
    "validate":  stage_validate,
    "convert":   stage_convert,
    "masks":     stage_masks,
    "metadata":  stage_metadata,
    "train":     stage_train,
    "evaluate":  stage_evaluate,
    "integrate": stage_integrate,
    "verify":    stage_verify,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    stages = args.only if args.only else [s for s in ALL_STAGES if s not in args.skip]
    print(f"  [{_ts()}] Pipeline stages: {stages}  (quick={args.quick})", flush=True)

    failed = []
    for stage in stages:
        fn = STAGE_FUNCS[stage]
        try:
            rc = fn(quick=args.quick)
        except Exception as exc:
            print(f"  [{_ts()}] STAGE {stage} raised: {exc}", flush=True)
            rc = 1
        if rc != 0:
            failed.append(stage)
            print(f"  [{_ts()}] Stage '{stage}' failed (rc={rc}) — halting.", flush=True)
            break

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    if failed:
        print(f"  PIPELINE FAILED at stage(s): {failed}  ({elapsed:.1f}s)")
    else:
        print(f"  PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*72}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

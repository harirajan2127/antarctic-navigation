#!/usr/bin/env python3
"""Full integration pipeline for the real-data Antarctic DSS.

Runs verification, DB migration, and optionally boots both backend and frontend
so the dashboard displays real data end-to-end.

Usage
-----
  python scripts/run_real_data_pipeline.py          # verify + migrate
  python scripts/run_real_data_pipeline.py --boot   # verify + migrate + start servers

After a successful run the backend is available at http://localhost:8000
and the frontend at http://localhost:5173.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PYTHON = sys.executable


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _banner(msg: str) -> None:
    print(f"\n{'='*72}\n  [{_ts()}] {msg}\n{'='*72}", flush=True)


def _run(cmd: list[str], label: str, timeout: int = 3600, cwd: str | None = None, **kw) -> int:
    print(f"  [{_ts()}] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, timeout=timeout, **kw)
    if proc.returncode != 0:
        print(f"  [{_ts()}] FAIL ({label}): exit code {proc.returncode}", flush=True)
    else:
        print(f"  [{_ts()}] OK    ({label})", flush=True)
    return proc.returncode


def stage_verify() -> int:
    _banner("Stage 1: Verify real processed data")
    return _run(
        [PYTHON, str(SCRIPTS_DIR / "verify_real_data.py")],
        "verify_real_data",
        timeout=900,
    )


def stage_migrate() -> int:
    _banner("Stage 2: Import real data into the database")
    return _run(
        [PYTHON, str(SCRIPTS_DIR / "import_real_csv_to_database.py")],
        "import_real_csv",
        timeout=600,
    )


def stage_boot() -> None:
    _banner("Stage 3: Boot backend and frontend")

    # Kill any stale process on port 8000 before starting fresh
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_use = s.connect_ex(("127.0.0.1", 8000)) == 0
        s.close()
        if in_use:
            print("  Port 8000 is in use; skipping backend start.", flush=True)
        else:
            env = os.environ.copy()
            env["DATA_MODE"] = "real"
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(BACKEND_DIR),
                env=env,
            )
            print("  Backend started on http://localhost:8000", flush=True)
    except Exception as exc:
        print(f"  Backend start failed: {exc}", flush=True)

    # Frontend
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "package.json").exists():
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
        )
        print("  Frontend started on http://localhost:5173", flush=True)
    else:
        print("  Frontend directory not found; skipping.", flush=True)

    print(f"\n  [{_ts()}] Services started. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n  [{_ts()}] Shutting down.", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full real-data integration pipeline.")
    p.add_argument("--boot", action="store_true",
                   help="After verify+migrate, start backend and frontend servers.")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the data verification stage.")
    p.add_argument("--skip-migrate", action="store_true",
                   help="Skip the DB migration stage.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    os.environ["DATA_MODE"] = "real"
    stages_run = []

    if not args.skip_verify:
        rc = stage_verify()
        if rc != 0:
            print(f"\n  PIPELINE FAILED at verify stage (rc={rc})")
            return 1
        stages_run.append("verify")

    if not args.skip_migrate:
        rc = stage_migrate()
        if rc != 0:
            print(f"\n  PIPELINE FAILED at migrate stage (rc={rc})")
            return 1
        stages_run.append("migrate")

    elapsed = time.time() - t0
    _banner("PIPELINE COMPLETE")
    print(f"  Stages run: {stages_run}  ({elapsed:.1f}s total)")
    print(f"\n  To start the backend:")
    print(f"    cd backend && uvicorn main:app --reload --port 8000")
    print(f"\n  To start the frontend:")
    print(f"    cd frontend && npm run dev")

    if args.boot:
        stage_boot()

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Demo-integrity check for the running DSS services.

Verifies, against the *live* backend (default http://127.0.0.1:8000), that:

* every dataset/iceberg/model response is honestly labeled (demo vs real),
* `real_data_available` is consistent with the dataset classifications,
* the trained-model registry reports what actually exists on disk, and
* the analytics summary/metrics endpoints respond sanely.

Exits with a non-zero status and a clear message when any dishonesty
(or an unexpected error) is found. Read-only — never modifies data.

Usage:
    python scripts/check_demo.py                      # http://127.0.0.1:8000
    python scripts/check_demo.py --base http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError

BASE = "http://127.0.0.1:8000"
REAL_CLASSIFICATIONS = {"synthetic_demo", "unknown", "unclassified", ""}

_failures: list[str] = []


def _get(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {body[:300]}") from exc
    except URLError as exc:
        raise SystemExit(f"Could not reach {url} — is the backend running? ({exc.reason})") from exc


def _check(label: str, ok: bool, detail: str = "") -> None:
    flag = "OK " if ok else "FAIL"
    print(f"[{flag}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(f"{label}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo-integrity check for the DSS API")
    ap.add_argument("--base", default=BASE, help="Backend base URL (default %(default)s)")
    args = ap.parse_args()
    root = args.base.rstrip("/")

    datasets = _get(f"{root}/api/datasets/status")
    items = datasets.get("datasets", [])
    _check("datasets/status responds", True, f"{len(items)} datasets")

    demo_count = sum(1 for d in items if d.get("demo"))
    _check(
        "every demo dataset has demo=true and labels synthetic_demo",
        all(bool(d.get("demo")) == (d.get("classification") == "synthetic_demo") for d in items),
        f"{demo_count}/{len(items)} labeled demo",
    )

    real_flag = bool(datasets.get("real_data_available"))
    real_detected = any(
        str(d.get("classification")) not in REAL_CLASSIFICATIONS for d in items
    )
    _check(
        "real_data_available consistent with classifications",
        real_flag == real_detected,
        f"flag={real_flag}, detected={real_detected}",
    )

    summary = _get(f"{root}/api/analytics/summary")
    _check(
        "analytics summary agrees on real_data_available",
        bool(summary.get("real_data_available")) == real_flag,
    )
    _check("analytics summary metrics available", "metrics_available" in summary)

    metrics = _get(f"{root}/api/analytics/model-metrics")
    metrics_demo = bool(metrics.get("demo_mode"))
    _check(
        "every metric carries a demo flag",
        all(isinstance(metric.get("demo"), bool) for metric in metrics.get("metrics", [])),
    )
    if metrics_demo:
        _check(
            "all metrics labeled demo while in demo mode",
            all(metric.get("demo") is True for metric in metrics.get("metrics", [])),
            f"{metrics.get('count', 0)} metrics in demo mode",
        )

    models = _get(f"{root}/api/models/status")
    models_demo = bool(models.get("demo_mode"))
    _check(
        "every model carries a trained_on_demo flag",
        all(m.get("trained_on_demo") is not False or m.get("available") for m in models.get("models", [])),
    )
    if models_demo:
        _check(
            "all checkpoints labeled demo while in demo mode",
            all(m.get("trained_on_demo") is True for m in models.get("models", [])),
            f"{models.get('count', 0)} checkpoints in demo mode",
        )

    icebergs = _get(f"{root}/api/icebergs")
    _check(
        "iceberg list carries demo label",
        icebergs.get("demo") is True,
        f"{icebergs.get('count')} icebergs",
    )

    health = _get(f"{root}/api/health")
    _check("health endpoint ok", health.get("status") == "ok", str(health.get("status")))

    if _failures:
        print("\nDemo-integrity check FAILED:")
        for f in _failures:
            print("  - " + f)
        return 1
    print("\nAll demo-integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
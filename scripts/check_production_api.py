"""Check the deployed DSS API without changing data or application state."""
from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    started = time.perf_counter()
    result: dict = {"endpoint": path, "method": method}
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}{path}", data=body, method=method, headers=headers), timeout=120) as response:
            raw = response.read()
            result.update({"status": response.status, "response_time_ms": round((time.perf_counter() - started) * 1000, 1), "response_size": len(raw)})
            try:
                result["data"] = json.loads(raw)
            except json.JSONDecodeError:
                result["error"] = "Response was not JSON"
    except HTTPError as exc:
        raw = exc.read()
        result.update({"status": exc.code, "response_time_ms": round((time.perf_counter() - started) * 1000, 1), "response_size": len(raw), "error": raw.decode(errors="replace")[:500]})
    except (URLError, TimeoutError, OSError) as exc:
        result.update({"status": None, "response_time_ms": round((time.perf_counter() - started) * 1000, 1), "response_size": 0, "error": str(exc)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a hosted Antarctic Navigation DSS backend")
    parser.add_argument("base_url", help="Public HTTPS backend URL, without /api")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    results = [
        request(base_url, "GET", path)
        for path in (
            "/api/health",
            "/api/system/status",
            "/api/datasets/status",
            "/api/sea-ice/current",
            "/api/icebergs",
            "/api/vessels",
            "/api/analytics/summary",
            "/api/models/status",
            "/api/v1/config/ports",
            "/api/v1/config/research-centers",
        )
    ]

    iceberg_data = next((r.get("data", {}) for r in results if r["endpoint"] == "/api/icebergs" and isinstance(r.get("data"), dict)), {})
    icebergs = iceberg_data.get("icebergs", [])
    if icebergs:
        iceberg_id = icebergs[0].get("iceberg_id")
        results.append(request(base_url, "GET", f"/api/icebergs/{iceberg_id}/trajectory"))

    ports = next((r.get("data", []) for r in results if r["endpoint"] == "/api/v1/config/ports"), [])
    centers = next((r.get("data", []) for r in results if r["endpoint"] == "/api/v1/config/research-centers"), [])
    vessels = next((r.get("data", {}).get("vessels", []) for r in results if r["endpoint"] == "/api/vessels" and isinstance(r.get("data"), dict)), [])
    if ports and centers and vessels:
        results.append(request(base_url, "POST", "/api/routes/optimize", {
            "start_latitude": ports[0]["latitude"],
            "start_longitude": ports[0]["longitude"],
            "destination_latitude": centers[0]["latitude"],
            "destination_longitude": centers[0]["longitude"],
            "vessel_id": vessels[0]["vessel_id"],
            "preference": "recommended",
        }))

    for result in results:
        data = result.get("data")
        result["required_data_present"] = bool(data) and "error" not in result
        result.pop("data", None)
    print(json.dumps({"base_url": base_url, "results": results}, indent=2))
    return 0 if all(r.get("status") and 200 <= r["status"] < 400 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
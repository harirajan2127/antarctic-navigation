"""End-to-end demo workflow for the Antarctic Navigation DSS.

Drives the *live* legacy journey API (`/api/v1/navigation/journey`) through the
accepted demo workflow:

    1. backend running (skipped here — run uvicorn app.main:app :8001)
    2. load/validate demo data and list departure ports + research centers
    3. select a departure port and a research center
    4. start an outbound journey
    5. advance two simulated hours -> vessel position changes
    6. route recalculates from the current position (rolling)
    7. advance again; the route keeps recomputing on the live risk grid
    8. loop until arrival (or --steps advances)
    9. show the final route metrics
   10. run a return journey (--return) back to the departure port

Read-only over HTTP. Uses only the Python standard library.

Usage:
    python scripts/demo_workflow.py
    python scripts/demo_workflow.py --port cape_town_za --center bharati_in --return
    python scripts/demo_workflow.py --api-base http://127.0.0.1:8001 --steps 6
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError

DEFAULT_BASE = "http://127.0.0.1:8001"


def _request(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', 'replace')[:400]}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach {url} — is the legacy app running? ({exc.reason})") from exc


def _nav(base: str, path: str, method: str = "GET", **params) -> str:
    url = f"{base}/api/v1/navigation/{path}"
    if params:
        from urllib.parse import urlencode

        url += f"?{urlencode({k: str(v) for k, v in params.items()})}"
    return url


def _start(base: str, port: str, center: str, journey_mode: str) -> dict:
    return _request(
        "POST", _nav(base, "journey"),
        {"departure_port_id": port, "destination_id": center,
         "journey_mode": journey_mode, "position_mode": "simulation"},
    )


def _print_route(route: dict, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    print(f"    origin       : {route['origin']}")
    print(f"    destination  : {route['destination']}")
    print(f"    mode         : {route['journey_mode']}")
    print(f"    vessel       : ({route['current_vessel_lat']:.3f}, {route['current_vessel_lon']:.3f})")
    print(f"    distance     : {route['total_distance_nm']:.1f} nm")
    print(f"    est. duration: {route['estimated_duration_hours']:.1f} h")
    print(f"    est. fuel    : {route['total_fuel_tons']:.1f} t")
    print(f"    max risk     : {route['max_risk_level']}")
    print(f"    waypoints    : {len(route['waypoints'])}")
    print(f"    route updates: {route['route_update_count']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end demo workflow driver")
    ap.add_argument("--api-base", default=DEFAULT_BASE)
    ap.add_argument("--port", default="hobart_au", help="departure port id")
    ap.add_argument("--center", default="casey_au", help="research center id")
    ap.add_argument("--steps", type=int, default=0, help="advance steps (0 = until arrival)")
    ap.add_argument("--return", dest="run_return", action="store_true",
                    help="also run the return journey")
    args = ap.parse_args()
    base = args.api_base.rstrip("/")

    print("==> [1/10] contacting legacy journey API")
    health = _request("GET", f"{base}/health")
    print(f"      backend: {health.get('status')} v{health.get('version')}")

    print("==> [2/10] listing data & fleet")
    ports = _request("GET", f"{base}/api/v1/config/ports")
    centers = _request("GET", f"{base}/api/v1/config/research-centers")
    print(f"      {len(ports)} ports, {len(centers)} research centers loaded")

    port_ids = {p["port_id"] for p in ports}
    center_ids = {c["center_id"] for c in centers}
    if args.port not in port_ids:
        raise SystemExit(f"Unknown port {args.port!r}; choose from {sorted(port_ids)}")
    if args.center not in center_ids:
        raise SystemExit(f"Unknown center {args.center!r}; choose from {sorted(center_ids)}")

    print("==> [3/10] departure selection")
    port = next(p for p in ports if p["port_id"] == args.port)
    print(f"      departing {port['name']} ({port['latitude']}, {port['longitude']})")

    print("==> [4/10] research center selection")
    center = next(c for c in centers if c["center_id"] == args.center)
    print(f"      destination {center['name']} ({center['latitude']}, {center['longitude']})")

    print("==> [5/10] starting outbound journey")
    route = _start(base, args.port, args.center, "outbound")
    journey_id = route["journey_id"]
    _print_route(route, "initial route")
    start_pos = (route["current_vessel_lat"], route["current_vessel_lon"])

    step = 0
    while True:
        print(f"==> [6-8/10] advancing 2 simulated hours (step {step + 1})")
        route = _request("POST", _nav(base, f"journey/{journey_id}/advance"))
        new_pos = (route["current_vessel_lat"], route["current_vessel_lon"])
        moved = new_pos != start_pos
        print(f"      position changed : {moved}")
        print(f"      vessel           : ({new_pos[0]:.3f}, {new_pos[1]:.3f})")
        print(f"      route updates    : {route['route_update_count']}")
        print(f"      remaining        : {route['total_distance_nm']:.1f} nm")
        start_pos = new_pos
        status = _request("GET", _nav(base, f"journey/{journey_id}/status"))
        print(f"      progress         : {status['progress_percent']:.1f}%  "
              f"time {status['current_time_hours']:.1f}h  complete={status['is_complete']}")
        step += 1
        if status["is_complete"]:
            break
        if args.steps and step >= args.steps:
            break

    print("==> [9/10] final route metrics")
    if status["is_complete"]:
        print(f"      journey {journey_id} COMPLETED after {status['current_time_hours']:.1f} h "
              f"({step} two-hour updates)")
        _print_route(route, "final route")
    else:
        _print_route(route, "route so far")
        print(f"      (stopped after {step} steps; run with --steps 0 to reach the destination)")

    if args.run_return:
        print("==> [10/10] return journey back to the departure port")
        ret = _start(base, args.port, args.center, "return")
        ret_id = ret["journey_id"]
        _print_route(ret, "return route")
        ret_status = _request("GET", _nav(base, f"journey/{ret_id}/status"))
        print(f"      return journey {ret_id} ready — origin {ret['origin']}, "
              f"destination {ret['destination']}")

    print("\nDemo workflow completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
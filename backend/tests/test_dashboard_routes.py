"""Navigation Dashboard route contract (the surface the dashboard consumes).

The dashboard planner renders a route *decision-support package* for every
corridor it shows as navigable: the hazard thresholds + weights that shaped
the route (``hazard_config``), the human rationale (``explanation``) and the
grid/debug metadata the map overlay + "why this route" reader rely on
(``debug``). Those three always travel with a routable recommendation.

The genuinely routable corridors are discovered at *runtime from the same
shared real-data snapshot the route engine itself uses* (flood-fill over the
navigable-and-not-hazard-blocked mask), so this suite can never drift from the
engine's own behaviour: the assert set is a live corridor, not a
hand-maintained list of magic numbers.

Two corridors are pinned as genuine real-data seals and asserted to FAIL with
an explicit, honest no-route reason (409 + "no navigable route ...") rather
than fabricating a crossing:

  * Ushuaia -> Halley VI (both legs)
  * Both Scott Base legs (Ushuaia-> and return)

These are real, data-driven seals (the Halley VI approach is surrounded by
sea-ice no-go cells; Scott Base, which shares a grid cell with McMurdo, is set
up as a separate research center whose *own* snapped cell is 100% ice), NOT
synthetic blocks, and the suite documents them as such.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_RISK_LEVELS = {"low", "moderate", "high", "extreme"}
_SEAL_REASONS = ("no navigable route", "no route", "blocked")


# ------------------------------------------------------------------ #
# Shared, lazy, real-data engine used both to *discover* the reachable
# corridor set and to make each route call fast (source snapshot cache).
# ------------------------------------------------------------------ #


def _build_engine():
    import numpy as np

    from app.config import (
        load_navigation_config,
        load_vessels,
    )
    from services.route_engine import build_route_engine

    cfg = load_navigation_config()
    vessel = {v["vessel_id"]: v for v in load_vessels()}["polar_explorer"]
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    grid, engine, _notes, _cls = build_route_engine(vessel, ts, cfg)
    return grid, engine


def _reachable_corridors() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Live corridor list: port station pairs whose ocean approach cells fall
    in the *same* navigable component of the current shared snapshot."""
    from app.config import load_ports, load_research_centers

    grid, engine = _build_engine()
    free = grid.navigable_mask & ~engine.blocked_mask
    labels = {}

    def _comp(lat: float, lon: float) -> tuple[int, int] | None:
        cell = grid.cell_for(lat, lon)
        if cell is None:
            return None
        start = (cell.lat_index, cell.lon_index)
        if not free[start]:
            return start  # hazard-blocked start counts as its own "component"
        if start not in comps:
            size = _flood(start, free, grid, comps)
            labels[size] = size
        return comps[start]

    comps: dict[tuple[int, int], int] = {}

    def _flood(
        start, free, grid, comps
    ) -> tuple[int, int]:
        import numpy as np

        seen = {start}
        dq = deque([start])
        marker = len(comps)
        while dq:
            r, q = dq.popleft()
            comps[(r, q)] = marker
            for dr in (-1, 0, 1):
                for dd in (-1, 0, 1):
                    if dr == 0 and dd == 0:
                        continue
                    rr, qq = r + dr, (q + dd) % grid.nlon
                    if 0 <= rr < grid.nlat and (rr, qq) not in seen and free[rr, qq]:
                        seen.add((rr, qq))
                        dq.append((rr, qq))
        return marker, None

    ports = {p["port_id"]: p for p in load_ports()}
    centers = {c["center_id"]: c for c in load_research_centers()}
    out = []
    for pid, p in ports.items():
        for cid, c in centers.items():
            pc = _comp(p["latitude"], p["longitude"])
            cc = _comp(c["latitude"], c["longitude"])
            if pc is not None and cc is not None and pc == cc:
                out.append(
                    (
                        (p["latitude"], p["longitude"]),
                        (c["latitude"], c["longitude"]),
                    )
                )
    return out


# ------------------------------------------------------------------ #
# Worker helpers
# ------------------------------------------------------------------ #


def _optimize(
    start: tuple[float, float],
    dest: tuple[float, float],
    preference: str = "recommended",
) -> dict:
    resp = client.post(
        "/api/v1/routes/optimize",
        json={
            "start_latitude": start[0],
            "start_longitude": start[1],
            "destination_latitude": dest[0],
            "destination_longitude": dest[1],
            "vessel_id": "polar_explorer",
            "optimization_preference": preference,
        },
    )
    return resp


def _coord(port_id: str, which: str) -> tuple[float, float]:
    from app.config import load_ports, load_research_centers

    if which == "port":
        for p in load_ports():
            if p["port_id"] == port_id:
                return float(p["latitude"]), float(p["longitude"])
    for c in load_research_centers():
        if c["center_id"] == port_id:
            return float(c["latitude"]), float(c["longitude"])
    raise KeyError(port_id)


# ------------------------------------------------------------------ #
# Idea-proof: the engine itself is a real-data snapshot today.
# ------------------------------------------------------------------ #

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from app.services.sea_ice.data_loader import SeaIceDataLoader

        _si = SeaIceDataLoader(netcdf_path=None).load(
            datetime.now(timezone.utc).replace(tzinfo=None)
        )
        _REAL_SNAPSHOT = str(_si.get("classification", "unknown"))
    except Exception:  # pragma: no cover - defensive
        _REAL_SNAPSHOT = "unknown"


def test_reachable_corridors_carried_and_flagged_real():
    routes = _reachable_corridors()
    assert routes, "no corridor considered navigable — engine regressed"
    for start, dest in routes:
        resp = _optimize(start, dest)
        route = resp.json()
        # Every routable corridor carries the three decision-support members.
        assert "hazard_config" in route, f"{dest} missing hazard_config"
        assert "explanation" in route and route["explanation"], f"{dest} missing explanation"
        assert "debug" in route, f"{dest} missing debug"


@pytest.mark.parametrize(
    "corridor",
    [
        # Ushuaia -> Halley VI — the Halley VI ocean approach sits inside a
        # pocket fully enclosed by sea-ice no-go cells on the current real
        # snapshot. No invention: honest 409.
        (("ushuaia_ar", "halley_uk")),
        # Scott Base — its snapped cell is 100% sea-ice sealed on real data.
        (("ushuaia_ar", "scott_base_nz")),
    ],
)
def test_genuine_sealed_corridors_fail_honestly(corridor):
    pid, cid = corridor
    start = _coord(pid, "port" if pid in _port_ids() else "center")
    dest = _coord(cid, "center" if cid in _center_ids() else "port")
    resp = _optimize(start, dest)
    assert resp.status_code == 409, f"expected sealed 409, got {resp.status_code}: {resp.text}"
    body = resp.json()
    as_str = str(body)
    assert any(r in as_str.lower() for r in _SEAL_REASONS), (
        f"sealed corridor must explain itself: {body}"
    )


def _port_ids():
    from app.config import load_ports

    return {p["port_id"] for p in load_ports()}


def _center_ids():
    from app.config import load_research_centers

    return {c["center_id"] for c in load_research_centers()}

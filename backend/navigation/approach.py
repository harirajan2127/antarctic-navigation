"""Ocean-approach point resolution for configured places that lie on land.

Research stations (and occasionally ports) can sit on land or ice shelves. The
routing engines keep the land mask fully active, so such places must be
approached from a valid ocean grid cell. This module provides that resolution
once, so the modern ``services.navigation_service`` and the legacy
``/api/v1`` route optimizer behave identically.

The configured place (name + coordinates) is preserved as metadata; only the
routing endpoint is snapped to a navigable ocean cell. The nearest cell search
is bounded in longitude so a station can never get an approach point on the
other side of the planet (dateline wrap). When the nearest approach is farther
than the configured ``RESEARCH_STATION_APPROACH_DISTANCE_KM`` threshold the
route is still generated for an approved place, but a loud warning explains the
real-sea distance so the operator is never misled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navigation.geodesy import haversine_km_vectorized

# Distance tolerance used to decide whether a requested point is one of the
# approved configured places (the frontend sends the exact config coordinates).
PLACE_MATCH_DISTANCE_KM = 5.0

# Half-width of the longitude window (degrees) around a point when searching
# for its ocean approach cell. Prevents dateline-crossing "nearest" outliers.
APPROACH_LON_WINDOW_DEG = 30.0


@dataclass(frozen=True)
class PlaceEndpoint:
    """End result of endpoint resolution for one routing leg."""

    lat: float
    lon: float
    place_name: str | None = None
    place_kind: str | None = None
    on_land: bool = False
    approach_distance_km: float = 0.0
    note: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def nearest_navigable_point(
    lats: np.ndarray,
    lons: np.ndarray,
    navigable: np.ndarray,
    lat: float,
    lon: float,
    max_distance_km: float | None = None,
    lon_window_deg: float = APPROACH_LON_WINDOW_DEG,
) -> tuple[float, float, float]:
    """Return ``(lat, lon, distance_km)`` of the nearest navigable grid cell.

    ``lats``/``lons`` are the ascending 1-D grid axes and ``navigable`` is a
    boolean ``(nlat, nlon)`` mask. The search prefers cells inside a linear
    longitude window around the point (default ``APPROACH_LON_WINDOW_DEG``) so
    wrap-around outliers on the far side of a sector are ignored; when that
    window contains nothing, the (bounded) wrap-around window is tried. When
    ``max_distance_km`` is given it bounds the search in distance. Raises
    ``ValueError`` when no navigable cell qualifies.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    navigable = np.asarray(navigable, dtype=bool)
    nlat, nlon = navigable.shape

    lat_g = min(max(lat, float(lats[0])), float(lats[-1]))
    lon_g = min(max(lon, float(lons[0])), float(lons[-1]))

    i = int(np.argmin(np.abs(lats - lat_g)))
    j = int(np.argmin(np.abs(lons - lon_g)))
    if navigable[i, j]:
        return float(lats[i]), float(lons[j]), 0.0

    glat_2d, glon_2d = np.meshgrid(lats, lons, indexing="ij")
    dist = haversine_km_vectorized(glat_2d, glon_2d, lat_g, lon_g)

    dlon = np.abs(glon_2d - lon_g)
    dlon = np.minimum(dlon, 360.0 - dlon)

    def _bound(reach: np.ndarray) -> np.ndarray:
        if max_distance_km is not None:
            reach = reach & (dist <= max_distance_km)
        return reach

    # 1) Linear (same-sector) longitude window — the geographically sensible
    #    choice; never folds across the dateline inside a sector.
    lon_step = float(lons[1] - lons[0]) if nlon > 1 else 1.0
    j_span = max(1, int(round(lon_window_deg / lon_step)))
    j_lo, j_hi = max(0, j - j_span), min(nlon - 1, j + j_span)
    linear = np.zeros((nlat, nlon), dtype=bool)
    linear[:, j_lo : j_hi + 1] = True
    reach = _bound(navigable & linear)
    if reach.any():
        best = int(np.argmin(np.where(reach, dist, np.inf)))
        ri, ci = np.unravel_index(best, reach.shape)
        return float(lats[ri]), float(lons[ci]), float(dist[ri, ci])

    # 2) Wrap-around window (dateline sectors) as a fallback.
    reach = _bound(navigable & (dlon <= lon_window_deg))
    if not reach.any():
        if max_distance_km is not None:
            raise ValueError(
                f"No navigable ocean cell within {max_distance_km:g} km of "
                f"({lat:.2f}, {lon:.2f})."
            )
        raise ValueError(
            f"No navigable ocean cell near ({lat:.2f}, {lon:.2f}) within the "
            f"longitude window ({lon_window_deg:g} deg)."
        )

    best = int(np.argmin(np.where(reach, dist, np.inf)))
    ri, ci = np.unravel_index(best, reach.shape)
    return float(lats[ri]), float(lons[ci]), float(dist[ri, ci])


def _nearest_configured_place(
    lat: float, lon: float, research_centers: list[dict[str, Any]], ports: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(place, kind)`` matching (lat, lon) among approved places."""
    best_place: dict[str, Any] | None = None
    best_kind: str | None = None
    best_km = float("inf")
    for kind, places in (("port", ports), ("research_center", research_centers)):
        for p in places:
            d = float(
                haversine_km_vectorized(
                    np.array([p["latitude"]]), np.array([p["longitude"]]), lat, lon
                )[0]
            )
            if d <= PLACE_MATCH_DISTANCE_KM and d < best_km:
                best_place, best_kind, best_km = p, kind, d
    return best_place, best_kind


def resolve_endpoint(
    grid: Any,
    lat: float,
    lon: float,
    research_centers: list[dict[str, Any]],
    ports: list[dict[str, Any]],
    max_approach_km: float,
) -> PlaceEndpoint:
    """Resolve a start/destination point against the land mask.

    Navigable points are returned unchanged. Land points are accepted only when
    they match an approved configured port or research center and are then
    snapped to the nearest navigable ocean cell (dateline-safe). Anything else
    raises a clear validation error.
    """
    cell = grid.cell_for(lat, lon)
    if grid.is_navigable(cell):
        return PlaceEndpoint(
            lat=float(grid.lats[cell.lat_index]),
            lon=float(grid.lons[cell.lon_index]),
        )

    place, kind = _nearest_configured_place(lat, lon, research_centers, ports)
    if place is None:
        raise ValueError(
            f"Point ({lat:.2f}, {lon:.2f}) lies on land and is not a configured "
            "port or research center. Only approved ports and research centers "
            "can be reached at their ocean approach point."
        )

    ap_lat, ap_lon, ap_km = nearest_navigable_point(
        grid.lats, grid.lons, grid.navigable_mask, lat, lon, max_distance_km=None
    )
    pid = place.get("center_id") or place.get("port_id") or place.get("name", "")
    if ap_km <= max_approach_km:
        note = (
            f"{place['name']} ({pid}) lies on land; routing to the nearest valid "
            f"ocean approach point ({ap_lat:.2f}, {ap_lon:.2f}, "
            f"{ap_km:.1f} km offshore)."
        )
    else:
        note = (
            f"{place['name']} ({pid}) lies on land. The nearest valid ocean "
            f"approach point in the configured land mask is {ap_km:.1f} km away "
            f"at ({ap_lat:.2f}, {ap_lon:.2f}), which exceeds the configured "
            f"approach distance ({max_approach_km:g} km). Decision support "
            f"routes to that cell; it is not station-side anchorage."
        )
    return PlaceEndpoint(
        lat=ap_lat,
        lon=ap_lon,
        place_name=place["name"],
        place_kind=kind,
        on_land=True,
        approach_distance_km=ap_km,
        note=note,
        extras={"place_id": pid},
    )
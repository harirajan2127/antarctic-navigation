"""Distance calculations for icebergs, vessels and routes.

All functions use the WGS-84 mean Earth radius and spherical haversine
distance. Outputs are rounded to sensible precision so downstream code never
needs to handle extra float noise.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "haversine_distance_km",
    "haversine_distance_nm",
    "distance_km",
    "distance_nm",
    "distance_between_icebergs",
    "vessel_to_iceberg",
    "minimum_distance_to_route",
]

_R_KM = 6371.0088
_NM_PER_KM = 1.0 / 1.852
_SAMPLES_PER_SEGMENT = 50


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two points."""
    r1 = np.radians(lat1)
    r2 = np.radians(lat2)
    dr = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dr / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return float(round(2 * _R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))), 10))


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    return round(haversine_distance_km(lat1, lon1, lat2, lon2) * _NM_PER_KM, 10)


distance_km = haversine_distance_km
distance_nm = haversine_distance_nm


def distance_between_icebergs(lat1: float, lon1: float, lat2: float, lon2: float) -> dict[str, float]:
    """Distance in km and nautical miles between two iceberg (or any) points."""
    km = haversine_distance_km(lat1, lon1, lat2, lon2)
    return {"distance_km": km, "distance_nm": round(km * _NM_PER_KM, 10)}


def vessel_to_iceberg(
    vessel_lat: float,
    vessel_lon: float,
    iceberg_lat: float,
    iceberg_lon: float,
) -> dict[str, float]:
    """Distance from a vessel to an iceberg."""
    return distance_between_icebergs(vessel_lat, vessel_lon, iceberg_lat, iceberg_lon)


def minimum_distance_to_route(
    route_coords: Sequence[tuple[float, float]] | Sequence[list[float]],
    iceberg_lat: float,
    iceberg_lon: float,
) -> dict[str, float]:
    """Minimum distance from an iceberg to any point along a vessel route.

    Parameters
    ----------
    route_coords
        Ordered list of ``(latitude, longitude)`` waypoints. Each segment is
        sampled at 50 interior points so a segment passing close to (but not
        through) the iceberg is still detected.
    iceberg_lat, iceberg_lon
        Iceberg position in degrees.

    Returns
    -------
    dict with ``distance_km`` and ``distance_nm``.
    """
    if len(route_coords) < 2:
        raise ValueError("route_coords must contain at least 2 waypoints.")
    lats = np.asarray([c[0] for c in route_coords], dtype=np.float64)
    lons = np.asarray([c[1] for c in route_coords], dtype=np.float64)
    rm = np.radians(iceberg_lat)
    best = float("inf")
    for i in range(len(route_coords) - 1):
        slats = np.linspace(lats[i], lats[i + 1], _SAMPLES_PER_SEGMENT, endpoint=False)
        slons = np.linspace(lons[i], lons[i + 1], _SAMPLES_PER_SEGMENT, endpoint=False)
        a = (
            np.sin(np.radians(slats - iceberg_lat) / 2) ** 2
            + np.cos(rm)
            * np.cos(np.radians(slats))
            * np.sin(np.radians(slons - iceberg_lon) / 2) ** 2
        )
        d = 2 * _R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        best = min(best, float(np.min(d)))
    # Include the final waypoint endpoint.
    a_end = (
        np.sin(np.radians(lats[-1] - iceberg_lat) / 2) ** 2
        + np.cos(rm)
        * np.cos(np.radians(lats[-1]))
        * np.sin(np.radians(lons[-1] - iceberg_lon) / 2) ** 2
    )
    best = min(best, 2 * _R_KM * np.arcsin(np.sqrt(np.clip(a_end, 0.0, 1.0))))
    km = round(float(best), 10)
    return {"distance_km": km, "distance_nm": round(km * _NM_PER_KM, 10)}
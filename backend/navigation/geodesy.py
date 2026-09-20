"""Spherical geodesy helpers for the navigation decision-support engine.

Kept dependency-free (stdlib + numpy only) so the routing engine can be
tested in isolation from the FastAPI application layer.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_KM = 6371.0088
NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlat = (lat2 - lat1) * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in nautical miles."""
    return haversine_km(lat1, lon1, lat2, lon2) * KM_TO_NM


def haversine_km_vectorized(
    lat: np.ndarray, lon: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Vectorized great-circle distance (km) from arrays of points to one point."""
    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat_r
    dlon = lon2_r - lon_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r) * math.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (forward azimuth) in degrees from point 1 to point 2."""
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)

    bearing = math.atan2(y, x) * RAD_TO_DEG
    return (bearing + 360.0) % 360.0


def normalize_longitude(lon: float) -> float:
    """Normalize a longitude into the range [-180, 180)."""
    while lon >= 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def km_per_degree_lat() -> float:
    """Meridional arc length of one degree of latitude (km), spherical model."""
    return math.pi * EARTH_RADIUS_KM / 180.0


def km_per_degree_lon(lat: float) -> float:
    """Zonal arc length of one degree of longitude at a latitude (km), spherical."""
    return km_per_degree_lat() * math.cos(lat * DEG_TO_RAD)
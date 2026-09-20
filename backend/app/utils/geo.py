"""Geospatial utility functions for the Antarctic Navigation DSS."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class LatLon(NamedTuple):
    latitude: float
    longitude: float


EARTH_RADIUS_NM = 3440.065
EARTH_RADIUS_KM = 6371.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in nautical miles."""
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlat = (lat2 - lat1) * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_NM * c


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlat = (lat2 - lat1) * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def haversine_distance_km_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Vectorized great-circle distance from an array of points to one point (km)."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * math.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial bearing (forward azimuth) in degrees from point 1 to point 2."""
    lat1_r = lat1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    dlon = (lon2 - lon1) * DEG_TO_RAD

    y = math.sin(dlon) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)

    bearing = math.atan2(y, x) * RAD_TO_DEG
    return (bearing + 360) % 360


def intermediate_point(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> LatLon:
    """Calculate an intermediate point along the great circle at a given fraction."""
    lat1_r = lat1 * DEG_TO_RAD
    lon1_r = lon1 * DEG_TO_RAD
    lat2_r = lat2 * DEG_TO_RAD
    lon2_r = lon2 * DEG_TO_RAD

    d = 2 * math.asin(
        math.sqrt(
            math.sin((lat2_r - lat1_r) / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin((lon2_r - lon1_r) / 2) ** 2
        )
    )

    if d < 1e-10:
        return LatLon(lat1, lon1)

    a = math.sin((1 - fraction) * d) / math.sin(d)
    b = math.sin(fraction * d) / math.sin(d)

    x = a * math.cos(lat1_r) * math.cos(lon1_r) + b * math.cos(lat2_r) * math.cos(lon2_r)
    y = a * math.cos(lat1_r) * math.sin(lon1_r) + b * math.cos(lat2_r) * math.sin(lon2_r)
    z = a * math.sin(lat1_r) + b * math.sin(lat2_r)

    lat = math.atan2(z, math.sqrt(x**2 + y**2)) * RAD_TO_DEG
    lon = math.atan2(y, x) * RAD_TO_DEG

    return LatLon(lat, lon)


def normalize_longitude(lon: float) -> float:
    """Normalize longitude to the range [-180, 180]."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def is_in_antarctic(lat: float) -> bool:
    """Check if a latitude is in the Antarctic region (south of -50)."""
    return lat <= -50.0


def grid_coverage(
    lat_min: float = -90.0,
    lat_max: float = -50.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
    resolution: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Generate latitude and longitude grid arrays."""
    lats = []
    lat = lat_min
    while lat <= lat_max:
        lats.append(round(lat, 4))
        lat += resolution

    lons = []
    lon = lon_min
    while lon <= lon_max:
        lons.append(round(lon, 4))
        lon += resolution

    return lats, lons

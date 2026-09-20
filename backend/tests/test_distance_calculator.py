"""Unit tests for backend.services.distance_calculator."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.distance_calculator import (
    distance_between_icebergs,
    distance_km,
    distance_nm,
    haversine_distance_km,
    haversine_distance_nm,
    minimum_distance_to_route,
    vessel_to_iceberg,
)

# WGS-84 haversine:  1 degree latitude ≈ 111.1949 km
_DEG_LAT_KM = 2 * 6371.0088 * math.pi / 360.0


class TestHaversineKM:
    def test_deg_one_lat(self):
        d = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
        assert abs(d - _DEG_LAT_KM) < 0.01

    def test_deg_ten_lat(self):
        d = haversine_distance_km(0.0, 0.0, 10.0, 0.0)
        assert abs(d - 10 * _DEG_LAT_KM) < 0.2

    def test_equator_10deg_lon(self):
        d = haversine_distance_km(0.0, 0.0, 0.0, 10.0)
        assert abs(d - 10 * _DEG_LAT_KM) < 0.2

    def test_symmetry(self):
        d1 = haversine_distance_km(-71.0, 160.0, -69.5, 162.0)
        d2 = haversine_distance_km(-69.5, 162.0, -71.0, 160.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_same_point(self):
        assert haversine_distance_km(71.5, 0.0, 71.5, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_known_landmark_madrid_rome(self):
        """Madrid (40.42, -3.70) to Rome (41.90, 12.50) ≈ 1365 km."""
        d = haversine_distance_km(40.4168, -3.7038, 41.9028, 12.4964)
        assert 1362 < d < 1370


class TestHaversineNM:
    def test_nm_conversion(self):
        km = 10.0
        nm = haversine_distance_nm(0.0, 0.0, 0.0, km / _DEG_LAT_KM)
        assert abs(nm - km / 1.852) < 1e-4

    def test_known_value(self):
        nm = haversine_distance_nm(0.0, 0.0, 1.0, 0.0)
        assert abs(nm - _DEG_LAT_KM / 1.852) < 0.01


class TestDistanceHelpers:
    def test_distance_km_nm(self):
        assert distance_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(haversine_distance_km(0, 0, 1, 0))
        assert distance_nm(0.0, 0.0, 1.0, 0.0) == pytest.approx(haversine_distance_nm(0, 0, 1, 0))

    def test_distance_between_icebergs(self):
        d = distance_between_icebergs(0.0, 0.0, 2.0, 2.0)
        assert set(d) == {"distance_km", "distance_nm"}
        assert d["distance_km"] > 0
        assert d["distance_nm"] == pytest.approx(d["distance_km"] / 1.852, rel=1e-9)

    def test_vessel_to_iceberg(self):
        d = vessel_to_iceberg(-71.0, 160.0, -71.0, 161.0)
        assert d["distance_km"] > 0
        assert set(d) == {"distance_km", "distance_nm"}


class TestMinimumDistanceToRoute:
    def test_route_passing_through_point(self):
        route = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
        d = minimum_distance_to_route(route, 1.0, 0.0)
        assert d["distance_km"] == pytest.approx(0.0, abs=1e-3)

    def test_iceberg_off_route_nearest_segment_midpoint(self):
        route = [(0.0, 0.0), (0.0, 2.0)]
        d = minimum_distance_to_route(route, 1.0, 1.0)
        assert abs(d["distance_km"] - _DEG_LAT_KM) < 0.1

    def test_route_two_segments(self):
        route = [(-70.0, 160.0), (-70.5, 161.0), (-71.0, 162.0)]
        d = minimum_distance_to_route(route, -70.2, 160.5)
        assert 0.0 < d["distance_km"] < 50.0

    def test_symmetric_distance(self):
        route1 = [(0.0, 0.0), (2.0, 0.0)]
        route2 = [(2.0, 0.0), (0.0, 0.0)]
        d1 = minimum_distance_to_route(route1, 1.0, 1.0)
        d2 = minimum_distance_to_route(route2, 1.0, 1.0)
        assert d1["distance_km"] == pytest.approx(d2["distance_km"], rel=1e-6)

    def test_too_short_route(self):
        with pytest.raises(ValueError):
            minimum_distance_to_route([(0.0, 0.0)], 1.0, 0.0)

    def test_nm_matches_km(self):
        d = minimum_distance_to_route([(0.0, 0.0), (2.0, 0.0)], 1.0, 1.0)
        assert d["distance_nm"] == pytest.approx(d["distance_km"] / 1.852, rel=1e-9)
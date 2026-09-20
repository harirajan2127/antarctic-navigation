"""Persistence baseline iceberg trajectory predictor.

Assumes the iceberg continues in a straight line at its last observed
velocity. This is the standard incremental baseline.
"""
from __future__ import annotations

import math
from typing import Any

from app.utils.geo import EARTH_RADIUS_KM, RAD_TO_DEG, intermediate_point


class PersistenceIcebergPredictor:
    """Straight-line drift prediction from last known velocity."""

    def __init__(self) -> None:
        self.is_trained = True  # persistence requires no training

    def predict_position(
        self, iceberg: dict[str, Any], horizon_hours: int
    ) -> tuple[float, float]:
        """Predict iceberg (lat, lon) after horizon_hours of straight-line drift."""
        lat = iceberg.get("latitude", 0.0)
        lon = iceberg.get("longitude", 0.0)
        speed_kph = iceberg.get("drift_speed_km_per_hour", 1.0)
        direction_deg = iceberg.get("drift_direction_deg", 90.0)

        # Fall back to a west-southwest drift typical of the Antarctic
        # Coastal Current when direction is not supplied.
        if "drift_direction_deg" not in iceberg:
            direction_deg = 225.0

        distance_km = speed_kph * horizon_hours
        lat2, lon2 = self._destination_point(lat, lon, direction_deg, distance_km)
        return lat2, lon2

    @staticmethod
    def _destination_point(
        lat: float, lon: float, bearing_deg: float, distance_km: float
    ) -> tuple[float, float]:
        """Compute the destination point given a bearing and distance."""
        angular = distance_km / EARTH_RADIUS_KM
        bearing = math.radians(bearing_deg)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)

        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular)
            + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )

        return math.degrees(lat2), math.degrees(lon2)

    def confidence_note(self) -> str:
        return (
            "Persistence baseline: uncertainty grows with forecast horizon. "
            "No learned confidence estimate."
        )
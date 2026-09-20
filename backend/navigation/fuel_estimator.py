"""Vessel fuel and travel-time estimation powered by vessel configuration.

No hard-coded consumption rates: all quantities come from the vessel record
(``fuel_consumption_tons_per_day`` phase rates or ``fuel_rate_lph``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FUEL_DENSITY_TONS_PER_CUBIC_METER = 0.85

_SPEED_PHASES = ("slow_speed", "cruise_speed", "max_speed")


@dataclass(frozen=True)
class FuelEstimate:
    """Travel-time and fuel estimate for a distance at a given speed."""

    distance_km: float
    distance_nm: float
    speed_knots: float
    travel_time_hours: float
    fuel_tons: float
    fuel_rate_tons_per_day: float
    rate_source: str


class FuelEstimator:
    """Estimates travel time and fuel from distance, speed, and a vessel record."""

    NM_TO_KM = 1.852

    @classmethod
    def travel_time_hours(cls, distance_km: float, speed_knots: float) -> float:
        """Hours to cover a distance in km at a speed in knots."""
        if speed_knots <= 0:
            raise ValueError(f"Speed must be positive, got {speed_knots}")
        distance_nm = distance_km / cls.NM_TO_KM
        return distance_nm / speed_knots

    @classmethod
    def rate_tons_per_day(cls, vessel: dict[str, Any], speed_knots: float) -> tuple[float, str]:
        """Resolve a vessel's fuel rate (tons/day) at a given speed.

        Priority:
        1. ``fuel_consumption_tons_per_day`` as numeric-keyed mapping
           (nearest speed).
        2. ``fuel_consumption_tons_per_day`` as phase-keyed mapping
           (``slow_speed``/``cruise_speed``/``max_speed``) compared against the
           vessel's configured cruise and max speeds.
        3. ``fuel_rate_lph`` converted via fuel density.
        """
        rate_map = vessel.get("fuel_consumption_tons_per_day") or {}
        if rate_map:
            keys = list(rate_map)
            if all(_is_number(k) for k in keys):
                breakpoints = sorted(float(k) for k in keys)
                nearest = min(breakpoints, key=lambda bp: abs(bp - speed_knots))
                # Keys may be strings; resolve the matching key.
                nearest_key = next(k for k in keys if float(k) == nearest)
                return float(rate_map[nearest_key]), f"fuel_consumption_tons_per_day[{nearest:g}]"

            if all(k in _SPEED_PHASES for k in keys):
                cruise = float(vessel.get("cruise_speed_knots") or speed_knots)
                max_speed = float(vessel.get("max_speed_knots") or cruise)
                if max_speed > cruise and speed_knots >= max_speed:
                    return float(rate_map["max_speed"]), "fuel_consumption_tons_per_day[max_speed]"
                if speed_knots >= cruise:
                    return float(rate_map["cruise_speed"]), "fuel_consumption_tons_per_day[cruise_speed]"
                return float(rate_map["slow_speed"]), "fuel_consumption_tons_per_day[slow_speed]"

        if vessel.get("fuel_rate_lph"):
            rate = float(vessel["fuel_rate_lph"]) * 24.0 * FUEL_DENSITY_TONS_PER_CUBIC_METER / 1000.0
            return rate, "fuel_rate_lph (converted)"

        raise ValueError(
            "Vessel record has no usable fuel configuration "
            "(fuel_consumption_tons_per_day or fuel_rate_lph)."
        )

    @classmethod
    def fuel_for_distance(
        cls, distance_km: float, speed_knots: float, vessel: dict[str, Any]
    ) -> FuelEstimate:
        """Estimate travel time and fuel for a distance using the vessel config."""
        rate_tons_per_day, source = cls.rate_tons_per_day(vessel, speed_knots)
        travel_hours = cls.travel_time_hours(distance_km, speed_knots)
        return FuelEstimate(
            distance_km=float(distance_km),
            distance_nm=float(distance_km / cls.NM_TO_KM),
            speed_knots=float(speed_knots),
            travel_time_hours=travel_hours,
            fuel_tons=rate_tons_per_day * travel_hours / 24.0,
            fuel_rate_tons_per_day=rate_tons_per_day,
            rate_source=source,
        )

    @classmethod
    def estimate(
        cls, distance_km: float, speed_knots: float, vessel: dict[str, Any]
    ) -> FuelEstimate:
        """Alias for :meth:`fuel_for_distance`."""
        return cls.fuel_for_distance(distance_km, speed_knots, vessel)


def _is_number(value: str | float | int) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
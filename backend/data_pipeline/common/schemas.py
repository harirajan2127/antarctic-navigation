"""Dataset schemas: required fields, units, and coordinate ranges.

A single source of truth used by both generation (demo) and validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VariableSpec:
    name: str
    units: str
    long_name: str
    valid_range: tuple[float, float] | None = None
    science_note: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str                   # identifier used for filenames
    title: str
    variables: tuple[VariableSpec, ...]
    cadence_hours_default: int      # scheduled availability cadence
    required_coords: tuple[str, ...] = ("time", "latitude", "longitude")

    def variable(self, name: str) -> VariableSpec | None:
        for v in self.variables:
            if v.name == name:
                return v
        return None


SEA_ICE_SPEC = DatasetSpec(
    dataset="sea_ice",
    title="Antarctic sea-ice concentration",
    cadence_hours_default=24,
    variables=(
        VariableSpec(
            "sea_ice_concentration",
            "fraction",
            "Sea-ice concentration",
            valid_range=(0.0, 1.0),
            science_note="Grid-cell fraction of sea-ice cover.",
        ),
    ),
)

OCEAN_SPEC = DatasetSpec(
    dataset="ocean_surface",
    title="Antarctic ocean surface fields",
    cadence_hours_default=24,
    variables=(
        VariableSpec(
            "uo",
            "m s-1",
            "Eastward current velocity at surface",
            valid_range=(-5.0, 5.0),
        ),
        VariableSpec(
            "vo",
            "m s-1",
            "Northward current velocity at surface",
            valid_range=(-5.0, 5.0),
        ),
        VariableSpec(
            "thetao",
            "K",
            "Sea surface temperature",
            valid_range=(-3.0, 308.0),
        ),
        VariableSpec(
            "so",
            "psu",
            "Sea surface salinity (PSS-78)",
            valid_range=(0.0, 42.0),
        ),
    ),
)

WEATHER_SPEC = DatasetSpec(
    dataset="weather_surface",
    title="ERA5 surface weather fields",
    cadence_hours_default=1,
    variables=(
        VariableSpec(
            "u10",
            "m s-1",
            "10-metre eastward wind",
            valid_range=(-80.0, 80.0),
        ),
        VariableSpec(
            "v10",
            "m s-1",
            "10-metre northward wind",
            valid_range=(-80.0, 80.0),
        ),
        VariableSpec(
            "t2m",
            "K",
            "2-metre temperature",
            valid_range=(150.0, 330.0),
        ),
        VariableSpec(
            "msl",
            "Pa",
            "Mean sea-level pressure",
            valid_range=(90000.0, 105000.0),
        ),
    ),
)

ICEBERG_SPEC = DatasetSpec(
    dataset="icebergs",
    title="Iceberg tracking records",
    cadence_hours_default=6,
    required_coords=(),
    variables=(
        VariableSpec("iceberg_id", "-", "Iceberg identifier"),
        VariableSpec("timestamp", "ISO8601", "Observation timestamp"),
        VariableSpec("latitude", "degrees_north", "Iceberg latitude"),
        VariableSpec("longitude", "degrees_east", "Iceberg longitude"),
        VariableSpec("length_nm", "nautical_miles", "Iceberg length"),
        VariableSpec("width_nm", "nautical_miles", "Iceberg width"),
    ),
)

ALL_SPECS: dict[str, DatasetSpec] = {
    spec.dataset: spec for spec in (SEA_ICE_SPEC, OCEAN_SPEC, WEATHER_SPEC, ICEBERG_SPEC)
}
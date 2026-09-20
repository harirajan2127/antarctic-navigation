"""Navigation risk grid construction from sea-ice, iceberg, and weather inputs."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.utils.geo import haversine_distance_km_vectorized


class NavigationRiskGrid:
    """Builds a multi-layer navigation risk grid over the Antarctic region.

    Risk contributions:
        - sea-ice concentration
        - iceberg proximity
        - weather severity (when data available)

    Final risk is a weighted combination, bounded to [0,1].
    """

    def __init__(
        self,
        lat_min: float = -85.0,
        lat_max: float = -55.0,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
        resolution: float = 0.5,
        risk_weights: dict[str, float] | None = None,
    ) -> None:
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution
        self.risk_weights = risk_weights or {
            "sea_ice_concentration": 0.35,
            "iceberg_proximity": 0.30,
            "weather_severity": 0.20,
            "distance_penalty": 0.15,
        }

        self.lats = np.arange(lat_min, lat_max + resolution, resolution)
        self.lons = np.arange(lon_min, lon_max + resolution, resolution)
        self.nlat = len(self.lats)
        self.nlon = len(self.lons)

        self.ice_risk = np.zeros((self.nlat, self.nlon))
        self.iceberg_risk = np.zeros((self.nlat, self.nlon))
        self.weather_risk = np.zeros((self.nlat, self.nlon))
        self.total_risk = np.zeros((self.nlat, self.nlon))

    def add_sea_ice(self, lat: list[float], lon: list[float], concentration: list[list[float]]) -> None:
        """Interpolate ice concentration onto the grid (nearest neighbor) and convert to risk."""
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)
        conc = np.asarray(concentration, dtype=float)

        i_idx = np.argmin(np.abs(lat_arr[:, None] - self.lats[None, :]), axis=0)
        j_idx = np.argmin(np.abs(lon_arr[:, None] - self.lons[None, :]), axis=0)
        sampled = conc[np.ix_(i_idx, j_idx)]
        self.ice_risk = self._ice_risk(sampled)

    @staticmethod
    def _ice_risk(concentration: float | np.ndarray) -> float | np.ndarray:
        """Map sea-ice concentration fraction to a risk score in [0,1] (vectorized)."""
        c = np.asarray(concentration, dtype=float)
        conditions = [
            c <= 0.15,
            c <= 0.4,
            c <= 0.6,
            c <= 0.8,
        ]
        choices = [
            0.0,
            0.2 * (c - 0.15) / 0.25,
            0.2 + 0.4 * (c - 0.4) / 0.2,
            0.6 + 0.3 * (c - 0.6) / 0.2,
        ]
        result = np.select(conditions, choices, default=0.9 + 0.1 * (c - 0.8) / 0.2)
        result = np.clip(result, 0.0, 1.0)
        if not np.ndim(concentration):
            return float(result)
        return result

    def add_icebergs(self, icebergs: list[dict[str, Any]]) -> None:
        """Add risk from iceberg proximity using a Gaussian falloff.

        Accepts both raw iceberg records (``latitude``/``longitude``) and
        trajectory prediction records (``current_lat``/``current_lon``, using
        the predicted position when present).
        """
        exclusion_km = 20.0
        reach_km = 150.0

        glat_2d, glon_2d = np.meshgrid(self.lats, self.lons, indexing="ij")

        iceberg_risk = np.zeros((self.nlat, self.nlon))
        for ice in icebergs:
            ilat = ice.get("latitude", ice.get("current_lat"))
            ilon = ice.get("longitude", ice.get("current_lon"))
            if "predicted_lat" in ice and "predicted_lon" in ice:
                ilat, ilon = ice["predicted_lat"], ice["predicted_lon"]
            if ilat is None or ilon is None:
                continue

            d = haversine_distance_km_vectorized(glat_2d, glon_2d, float(ilat), float(ilon))
            risk = np.zeros_like(d)
            risk[d < exclusion_km] = 1.0
            ring = (d >= exclusion_km) & (d < reach_km)
            risk[ring] = 0.5 * np.maximum(0.0, 1.0 - (d[ring] - exclusion_km) / (reach_km - exclusion_km))
            iceberg_risk = np.maximum(iceberg_risk, risk)

        self.iceberg_risk = np.maximum(self.iceberg_risk, iceberg_risk)

    def add_weather(self, severity_field: list[list[float]] | None = None) -> None:
        """Add weather severity risk if a field is provided."""
        if severity_field is not None:
            arr = np.asarray(severity_field, dtype=float)
            self.weather_risk = np.clip(arr, 0.0, 1.0)

    def compute_total(self) -> np.ndarray:
        """Combine risk layers using configured weights, clipped to [0,1]."""
        total = (
            self.risk_weights["sea_ice_concentration"] * self.ice_risk
            + self.risk_weights["iceberg_proximity"] * self.iceberg_risk
            + self.risk_weights["weather_severity"] * self.weather_risk
        )
        self.total_risk = np.clip(total, 0.0, 1.0)
        return self.total_risk

    def risk_at(self, lat: float, lon: float) -> float:
        """Return the total risk at an arbitrary lat/lon via nearest grid cell."""
        i = int(np.round((lat - self.lat_min) / self.resolution))
        j = int(np.round((lon - self.lon_min) / self.resolution))
        i = max(0, min(self.nlat - 1, i))
        j = max(0, min(self.nlon - 1, j))
        return float(self.total_risk[i, j])

    def to_dict(self) -> dict[str, Any]:
        """Export the grid as JSON-serializable data."""
        return {
            "lat": self.lats.tolist(),
            "lon": self.lons.tolist(),
            "total_risk": self.total_risk.tolist(),
            "ice_risk": self.ice_risk.tolist(),
            "iceberg_risk": self.iceberg_risk.tolist(),
            "weather_risk": self.weather_risk.tolist(),
            "risk_weights": self.risk_weights,
        }
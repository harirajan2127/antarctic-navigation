"""Multi-layer navigation risk engine over the Antarctic grid.

Risk layers:
- sea-ice concentration (piecewise concentration -> risk mapping)
- iceberg proximity (exclusion buffer, safety buffer, outer reach falloff)
- weather severity (wind speed / wave height or a supplied severity field)
- vessel constraints (ice class -> maximum safe sea-ice concentration)

The final per-cell risk is the weighted combination of the layers, clipped to
[0, 1], with land and vessel-blocked cells forced to maximum risk (blocked).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navigation.geodesy import haversine_km_vectorized
from navigation.grid import AntarcticGrid

DEFAULT_RISK_WEIGHTS = {
    "sea_ice_concentration": 0.40,
    "iceberg_proximity": 0.35,
    "weather_severity": 0.25,
}

DEFAULT_ICE_CLASS_LIMITS = {
    "PC1": 1.0,
    "PC2": 1.0,
    "PC3": 1.0,
    "PC4": 0.8,
    "PC5": 0.6,
    "PC6": 0.45,
    "PC7": 0.3,
    "NO": 0.2,
}

DEFAULT_OBJECTIVE_WEIGHTS = {"safety": 0.6, "distance": 0.4}


def _optional_float(value, default):
    """Coerce an optional numeric config value to float, or ``default`` when absent."""
    if value in (None, "", "null", "none"):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _fmt_km(value: float | None) -> str:
    return f"{value:g} km" if value is not None else "not configured"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "not configured"


@dataclass(frozen=True)
class RiskConfig:
    """Configurable risk thresholds and weights for the risk engine."""

    risk_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RISK_WEIGHTS))
    max_safe_sea_ice_concentration: float = 0.6
    sea_ice_block_threshold: float | None = 0.85
    sea_ice_high_risk_threshold: float | None = 0.6
    sea_ice_moderate_risk_threshold: float | None = 0.3
    iceberg_exclusion_radius_km: float = 20.0
    iceberg_safety_buffer_km: float = 30.0
    iceberg_reach_km: float = 150.0
    hard_iceberg_block_km: float | None = None
    iceberg_block_radius_km: float | None = None
    iceberg_warning_radius_km: float | None = None
    iceberg_prediction_horizon_hours: int = 24
    min_clearance_km: float = 10.0
    max_wind_speed_knots: float = 50.0
    max_wave_height_m: float = 8.0
    max_risk_ratio: float = 0.95
    ice_class_limits: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ICE_CLASS_LIMITS)
    )
    objective_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS)
    )

    @classmethod
    def from_dict(cls, config: dict | None) -> "RiskConfig":
        """Build a RiskConfig from a navigation configuration dictionary."""
        if not config:
            return cls()
        config = dict(config)
        thresholds = config.get("thresholds", {}) or {}
        return cls(
            risk_weights=dict(config.get("risk_weights", DEFAULT_RISK_WEIGHTS)),
            max_safe_sea_ice_concentration=float(
                thresholds.get(
                    "max_safe_sea_ice_concentration",
                    cls.max_safe_sea_ice_concentration,
                )
            ),
            sea_ice_block_threshold=_optional_float(
                thresholds.get("sea_ice_block_threshold"), cls.sea_ice_block_threshold
            ),
            sea_ice_high_risk_threshold=_optional_float(
                thresholds.get("sea_ice_high_risk_threshold"),
                cls.sea_ice_high_risk_threshold,
            ),
            sea_ice_moderate_risk_threshold=_optional_float(
                thresholds.get("sea_ice_moderate_risk_threshold"),
                cls.sea_ice_moderate_risk_threshold,
            ),
            iceberg_exclusion_radius_km=float(
                thresholds.get("iceberg_exclusion_radius_km", cls.iceberg_exclusion_radius_km)
            ),
            iceberg_safety_buffer_km=float(
                thresholds.get("iceberg_safety_buffer_km", cls.iceberg_safety_buffer_km)
            ),
            iceberg_reach_km=float(
                thresholds.get("iceberg_reach_km", cls.iceberg_reach_km)
            ),
            hard_iceberg_block_km=_optional_float(
                thresholds.get("hard_iceberg_block_km"), None
            ),
            iceberg_block_radius_km=_optional_float(
                thresholds.get("iceberg_block_radius_km"), None
            ),
            iceberg_warning_radius_km=_optional_float(
                thresholds.get("iceberg_warning_radius_km"), None
            ),
            iceberg_prediction_horizon_hours=int(
                thresholds.get(
                    "iceberg_prediction_horizon_hours",
                    cls.iceberg_prediction_horizon_hours,
                )
            ),
            min_clearance_km=float(
                thresholds.get("min_clearance_km", cls.min_clearance_km)
            ),
            max_wind_speed_knots=float(
                thresholds.get("max_wind_speed_knots", cls.max_wind_speed_knots)
            ),
            max_wave_height_m=float(
                thresholds.get("max_wave_height_m", cls.max_wave_height_m)
            ),
            max_risk_ratio=float(
                thresholds.get("max_risk_ratio", cls.max_risk_ratio)
            ),
            ice_class_limits=dict(
                config.get("ice_class_limits", DEFAULT_ICE_CLASS_LIMITS)
            ),
            objective_weights=dict(
                config.get("objective_weights", DEFAULT_OBJECTIVE_WEIGHTS)
            ),
        )


def sea_ice_risk(concentration: np.ndarray) -> np.ndarray:
    """Map sea-ice concentration fraction to a risk score in [0, 1] (vectorized).

    Non-finite (nodata / missing) concentrations map to neutral zero risk so a
    missing coastal observation can never poison the total-risk field; missing
    cells remain reported as ``None`` by the sampler.
    """
    c = np.asarray(concentration, dtype=float)
    c = np.where(np.isfinite(c), c, 0.0)
    conditions = [c <= 0.15, c <= 0.4, c <= 0.6, c <= 0.8]
    choices = [
        0.0,
        0.2 * (c - 0.15) / 0.25,
        0.2 + 0.4 * (c - 0.4) / 0.2,
        0.6 + 0.3 * (c - 0.6) / 0.2,
    ]
    result = np.select(conditions, choices, default=0.9 + 0.1 * (c - 0.8) / 0.2)
    return np.clip(result, 0.0, 1.0)


class RiskEngine:
    """Builds and combines the navigation risk layers over an AntarcticGrid."""

    def __init__(self, grid: AntarcticGrid, config: RiskConfig | None = None) -> None:
        self.grid = grid
        self.config = config or RiskConfig()

        shape = (grid.nlat, grid.nlon)
        self.sea_ice_concentration = np.full(shape, np.nan, dtype=float)
        self.ice_risk = np.zeros(shape, dtype=float)
        self.iceberg_risk = np.zeros(shape, dtype=float)
        self.weather_risk = np.zeros(shape, dtype=float)
        self.iceberg_distance = np.full(shape, np.inf, dtype=float)
        self.total_risk = np.zeros(shape, dtype=float)

        self.blocked_mask = np.zeros(shape, dtype=bool)
        self.hard_iceberg_block_applied = False
        self.iceberg_count = 0
        self.max_sea_ice_for_vessel: float | None = None
        self.vessel_clearance_km: float = self.config.min_clearance_km
        self._computed = False
        self.notes: list[str] = []

    # ------------------------------------------------------------------ #
    # Field internals
    # ------------------------------------------------------------------ #
    def _nearest(self, lat, lon, field) -> np.ndarray:
        """Nearest-neighbor sample a (lat, lon, field) record onto the grid."""
        lat_arr = np.asarray(lat, dtype=float)
        lon_arr = np.asarray(lon, dtype=float)
        field = np.asarray(field, dtype=float)

        i_idx = np.argmin(np.abs(lat_arr[:, None] - self.grid.lats[None, :]), axis=0)
        j_idx = np.argmin(np.abs(lon_arr[:, None] - self.grid.lons[None, :]), axis=0)
        return field[np.ix_(i_idx, j_idx)]

    def add_sea_ice(self, lat, lon, concentration) -> None:
        """Interpolate a sea-ice concentration field onto the grid as risk."""
        conc = np.clip(self._nearest(lat, lon, concentration), 0.0, 1.0)
        missing = int(np.count_nonzero(~np.isfinite(conc)))
        self.sea_ice_concentration = conc
        self.ice_risk = sea_ice_risk(conc)
        if missing:
            self.notes.append(
                f"{missing} grid cells have no sea-ice observation (nodata). "
                "They are treated as neutral open water, not as land."
            )
        self._computed = False

    def add_icebergs(self, icebergs: list[dict[str, Any]]) -> None:
        """Add iceberg proximity risk from a list of iceberg records.

        Accepts raw records (``latitude``/``longitude``) and trajectory
        prediction records (``current_lat``/``current_lon``, preferring
        ``predicted_lat``/``predicted_lon`` when present).
        """
        glat_2d, glon_2d = np.meshgrid(self.grid.lats, self.grid.lons, indexing="ij")

        distance = np.full((self.grid.nlat, self.grid.nlon), np.inf, dtype=float)
        tracked = 0
        for iceberg in icebergs:
            ilat, ilon = self._iceberg_position(iceberg)
            if ilat is None or ilon is None:
                continue
            tracked += 1
            d = haversine_km_vectorized(glat_2d, glon_2d, float(ilat), float(ilon))
            distance = np.minimum(distance, d)

        if tracked:
            self.iceberg_distance = distance
            self.iceberg_risk = np.maximum(self.iceberg_risk, self._iceberg_risk(distance))
            self.iceberg_count = tracked
        self._computed = False

    @staticmethod
    def _iceberg_position(iceberg: dict[str, Any]) -> tuple[float | None, float | None]:
        ilat = iceberg.get("latitude", iceberg.get("current_lat"))
        ilon = iceberg.get("longitude", iceberg.get("current_lon"))
        if iceberg.get("predicted_lat") is not None and iceberg.get("predicted_lon") is not None:
            ilat, ilon = iceberg["predicted_lat"], iceberg["predicted_lon"]
        return ilat, ilon

    def _iceberg_risk(self, distance: np.ndarray) -> np.ndarray:
        """Map distance-from-iceberg (km) to a risk score in [0, 1]."""
        cfg = self.config
        excl = cfg.iceberg_exclusion_radius_km
        buffer_km = cfg.iceberg_safety_buffer_km
        reach = cfg.iceberg_reach_km

        risk = np.zeros_like(distance)
        if excl <= 0:
            risk[distance < reach] = 1.0
            risk[distance >= reach] = 0.0
            return risk

        risk[distance < excl] = 1.0
        inner = (distance >= excl) & (distance < buffer_km)
        if buffer_km > excl:
            risk[inner] = 1.0 - 0.5 * (distance[inner] - excl) / (buffer_km - excl)
        else:
            risk[inner] = 1.0
        outer = (distance >= buffer_km) & (distance < reach)
        if reach > buffer_km:
            risk[outer] = 0.5 * (1.0 - (distance[outer] - buffer_km) / (reach - buffer_km))
        else:
            risk[outer] = 0.0
        return risk

    def add_weather_severity(self, severity_grid: np.ndarray | list[list[float]]) -> None:
        """Add a weather severity field already aligned to the grid."""
        arr = np.asarray(severity_grid, dtype=float)
        if arr.shape != (self.grid.nlat, self.grid.nlon):
            raise ValueError(
                f"Weather severity shape {arr.shape} does not match grid "
                f"{(self.grid.nlat, self.grid.nlon)}"
            )
        self.weather_risk = np.clip(arr, 0.0, 1.0)
        self._computed = False

    def add_weather_wind_wave(
        self, wind_knots, wave_height_m, lat, lon
    ) -> None:
        """Compute weather risk from wind speed and wave height fields."""
        wind = self._nearest(lat, lon, wind_knots)
        wave = self._nearest(lat, lon, wave_height_m)
        wind_ratio = np.clip(wind / self.config.max_wind_speed_knots, 0.0, 1.0)
        wave_ratio = np.clip(wave / self.config.max_wave_height_m, 0.0, 1.0)
        self.weather_risk = np.clip(np.maximum(wind_ratio, wave_ratio), 0.0, 1.0)
        self._computed = False

    def set_vessel_constraints(self, vessel: dict[str, Any]) -> None:
        """Record vessel ice class -> max sea-ice concentration and clearance."""
        ice_class = vessel.get("ice_class")
        limits = self.config.ice_class_limits
        if ice_class in limits:
            self.max_sea_ice_for_vessel = float(limits[ice_class])
        else:
            self.max_sea_ice_for_vessel = self.config.max_safe_sea_ice_concentration
            self.notes.append(
                f"Ice class {ice_class!r} not configured; using default safe "
                f"limit of {self.config.max_safe_sea_ice_concentration:.0%}."
            )

        vessel_safety_km = float(vessel.get("safety_distance_nm") or 0.0) * 1.852
        self.vessel_clearance_km = max(self.config.min_clearance_km, vessel_safety_km)
        self._computed = False

    def apply_vessel_constraints(self) -> None:
        """Block cells whose sea-ice concentration exceeds the vessel limit.

        A vessel constraint is an additional No-Go layer on top of any hazard
        zones already blocked (iceberg exclusion, sea-ice no-go), never a
        replacement of them.
        """
        if self.max_sea_ice_for_vessel is None:
            return
        conc = self.sea_ice_concentration
        finite = np.isfinite(conc)
        over = finite & (conc > self.max_sea_ice_for_vessel)
        self.blocked_mask = self.blocked_mask | over
        self._computed = False

    def apply_iceberg_hard_block(self) -> None:
        """Force cells within a configured radius of any iceberg to be blocked.

        When ``RiskConfig.hard_iceberg_block_km`` is set (and > 0), iceberg
        exclusion zones become true no-go areas for path finding — not just a
        risk-weight penalty that a weighted cost function can cross. The
        dedicated ``RiskConfig.iceberg_block_radius_km`` takes precedence when
        present so the hazard avoids zones an operator configures explicitly.
        """
        block_km = self._effective_iceberg_block_radius()
        if block_km is None or block_km <= 0:
            return
        finite = np.isfinite(self.iceberg_distance)
        zone = finite & (self.iceberg_distance < float(block_km))
        self.blocked_mask = self.blocked_mask | zone
        self.hard_iceberg_block_applied = True
        self.notes.append(
            f"Iceberg exclusion zones are hard-blocked within "
            f"{block_km:g} km (No-Go)."
        )
        self._computed = False

    def _effective_iceberg_block_radius(self) -> float | None:
        """The No-Go radius around icebergs, config-precedence-aware."""
        if self.config.iceberg_block_radius_km is not None:
            return self.config.iceberg_block_radius_km
        return self.config.hard_iceberg_block_km

    def apply_sea_ice_hazard_blocks(self) -> None:
        """Hard-block cells whose sea-ice concentration exceeds a configured cap.

        ``RiskConfig.sea_ice_block_threshold`` (default 0.85) defines the
        concentration above which a cell is a No-Go for path finding,
        independent of the per-vessel ice-class limit. This is the explicit
        sea-ice forecasting-zone avoidance rule surfaced to the dashboard.
        """
        threshold = self.config.sea_ice_block_threshold
        if threshold is None:
            return
        conc = self.sea_ice_concentration
        finite = np.isfinite(conc)
        over = finite & (conc > threshold)
        newly_blocked = over & ~self.blocked_mask
        self.blocked_mask = self.blocked_mask | over
        self._computed = False
        if newly_blocked.any():
            self.notes.append(
                f"Sea-ice No-Go zones applied: {int(newly_blocked.sum())} cells "
                f"above {threshold:.0%} concentration are avoided."
            )

    # ------------------------------------------------------------------ #
    # Combination
    # ------------------------------------------------------------------ #
    def compute_total(self) -> np.ndarray:
        """Combine all risk layers, clip to [0, 1], force land/vessel-blocks to 1."""
        w = self.config.risk_weights
        total = (
            w.get("sea_ice_concentration", 0.0) * self.ice_risk
            + w.get("iceberg_proximity", 0.0) * self.iceberg_risk
            + w.get("weather_severity", 0.0) * self.weather_risk
        )
        total = np.where(np.isfinite(total), total, 0.0)
        blocked = self.grid.land_mask | self.blocked_mask
        total[blocked] = 1.0
        self.total_risk = np.clip(total, 0.0, 1.0)
        self._computed = True
        return self.total_risk

    def _ensure_computed(self) -> None:
        if not self._computed:
            self.compute_total()

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def risk_at(self, lat: float, lon: float) -> float:
        """Total risk at an arbitrary (lat, lon) via the nearest cell."""
        cell = self.grid.cell_for(lat, lon)
        self._ensure_computed()
        return float(self.total_risk[cell.lat_index, cell.lon_index])

    def sample_at(self, lat: float, lon: float) -> dict[str, float]:
        """Sample the risk inputs and composite score at a coordinate."""
        cell = self.grid.cell_for(lat, lon)
        i, j = cell.lat_index, cell.lon_index
        self._ensure_computed()
        return {
            "sea_ice_concentration": float(self.sea_ice_concentration[i, j]),
            "iceberg_distance_km": float(self.iceberg_distance[i, j]),
            "risk_score": float(self.total_risk[i, j]),
        }

    def distance_to_nearest_iceberg(self, lat: float, lon: float) -> float:
        """Distance (km) from a coordinate to the nearest iceberg."""
        cell = self.grid.cell_for(lat, lon)
        return float(self.iceberg_distance[cell.lat_index, cell.lon_index])

    @staticmethod
    def risk_level(value: float) -> str:
        """Bucket a risk score into a human-readable level."""
        if value >= 0.8:
            return "extreme"
        if value >= 0.6:
            return "high"
        if value >= 0.3:
            return "moderate"
        return "low"

    def obstacle_mask(self, threshold: float | None = None) -> np.ndarray:
        """Boolean mask of non-traversable cells (risk exceeds threshold or land)."""
        self._ensure_computed()
        threshold = self.config.max_risk_ratio if threshold is None else threshold
        return self.total_risk >= threshold

    def waypoint_warnings(
        self, coordinates: list[tuple[float, float]]
    ) -> list[str]:
        """Generate human-readable risk warnings along a route."""
        cfg = self.config
        warnings: list[str] = []
        self._ensure_computed()

        for lat, lon in coordinates:
            sample = self.sample_at(lat, lon)
            level = self.risk_level(sample["risk_score"])
            if level in ("high", "extreme"):
                warnings.append(
                    f"{level.title()} risk (score {sample['risk_score']:.2f}) at "
                    f"({lat:.2f}, {lon:.2f})."
                )
            iceberg_km = sample["iceberg_distance_km"]
            if np.isfinite(iceberg_km) and iceberg_km < self.vessel_clearance_km:
                warnings.append(
                    f"Iceberg within minimum clearance of "
                    f"{self.vessel_clearance_km:.0f} km at ({lat:.2f}, {lon:.2f}) "
                    f"({iceberg_km:.1f} km)."
                )
            conc = sample["sea_ice_concentration"]
            if (
                self.max_sea_ice_for_vessel is not None
                and np.isfinite(conc)
                and conc > self.max_sea_ice_for_vessel
            ):
                warnings.append(
                    f"Sea-ice concentration {conc:.0%} exceeds vessel limit "
                    f"{self.max_sea_ice_for_vessel:.0%} at ({lat:.2f}, {lon:.2f})."
                )
            high = self.config.sea_ice_high_risk_threshold
            if high is not None and np.isfinite(conc) and conc >= high:
                warnings.append(
                    f"High sea-ice concentration {conc:.0%} (forecasting-zone "
                    f"threshold {high:.0%}) at ({lat:.2f}, {lon:.2f})."
                )
            warn_km = self.config.iceberg_warning_radius_km
            if (
                warn_km is not None
                and np.isfinite(iceberg_km)
                and iceberg_km < warn_km
            ):
                warnings.append(
                    f"Iceberg tracking hazard within {warn_km:.0f} km at "
                    f"({lat:.2f}, {lon:.2f}) ({iceberg_km:.1f} km)."
                )

        # De-duplicate while preserving order.
        return list(dict.fromkeys(warnings))

    # ------------------------------------------------------------------ #
    # Decision-support explanation / hazard configuration
    # ------------------------------------------------------------------ #
    def hazard_config(self) -> dict[str, Any]:
        """The hazard-avoidance configuration surfaced to the dashboard."""
        cfg = self.config
        return {
            "sea_ice": {
                "block_threshold": cfg.sea_ice_block_threshold,
                "high_risk_threshold": cfg.sea_ice_high_risk_threshold,
                "moderate_risk_threshold": cfg.sea_ice_moderate_risk_threshold,
                "max_safe_concentration": cfg.max_safe_sea_ice_concentration,
                "vessel_max_concentration": self.max_sea_ice_for_vessel,
            },
            "iceberg": {
                "block_radius_km": self._effective_iceberg_block_radius(),
                "warning_radius_km": cfg.iceberg_warning_radius_km or cfg.iceberg_safety_buffer_km,
                "prediction_horizon_hours": cfg.iceberg_prediction_horizon_hours,
                "tracked_icebergs": self.iceberg_count,
            },
            "risk_weights": dict(cfg.risk_weights),
            "objective_weights": dict(cfg.objective_weights),
            "max_risk_ratio": cfg.max_risk_ratio,
            "hard_iceberg_block_applied": self.hard_iceberg_block_applied,
        }

    def explanation(self) -> list[str]:
        """Readable reasons describing how hazards shaped this route grid."""
        lines = list(self.notes)
        if self.config.hard_iceberg_block_km is not None or (
            self.config.iceberg_block_radius_km is not None
        ):
            lines.append(
                "Iceberg tracking hazards are avoided with a hard no-go zone "
                f"({_fmt_km(self._effective_iceberg_block_radius())})."
            )
        if self.config.sea_ice_block_threshold is not None:
            lines.append(
                "Sea-ice forecasting zones above "
                f"{self.config.sea_ice_block_threshold:.0%} concentration are "
                "avoided as no-go; high-risk crossings ("
                f"{_fmt_pct(self.config.sea_ice_high_risk_threshold)}) are warned."
            )
        return list(dict.fromkeys(lines))
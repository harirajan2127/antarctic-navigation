"""Antarctic geographic grid and projection for the navigation engine.

Defines the rectangular lat/lon search space over the Antarctic region,
converts coordinates to a local projection for geometric work, and tracks
which grid cells are navigable (i.e. not land).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from navigation.geodesy import km_per_degree_lat, km_per_degree_lon

ANTARCTIC_REGION_DEFAULT = {
    "lat_min": -85.0,
    "lat_max": -55.0,
    "lon_min": -180.0,
    "lon_max": 180.0,
}
DEFAULT_RESOLUTION_DEG = 0.5


class GridCell(NamedTuple):
    """A single node on the Antarctic grid, referenced by index."""
    lat_index: int
    lon_index: int


class GridProjection:
    """Equirectangular tangent-plane projection around an origin (km).

    ``project`` returns easting/northing in kilometers relative to an origin
    point; ``unproject`` reverses the transform. This is a local, planar
    approximation used for geometric work, not a certified map projection.
    """

    def __init__(self, lat0: float, lon0: float = 0.0) -> None:
        self.lat0 = lat0
        self.lon0 = lon0
        self.km_per_deg_lat = km_per_degree_lat()
        self.km_per_deg_lon = km_per_degree_lon(lat0)

    def project(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert (lat, lon) to (easting_km, northing_km)."""
        easting = (lon - self.lon0) * self.km_per_deg_lon
        northing = (lat - self.lat0) * self.km_per_deg_lat
        return easting, northing

    def unproject(self, easting_km: float, northing_km: float) -> tuple[float, float]:
        """Convert (easting_km, northing_km) back to (lat, lon)."""
        lon = self.lon0 + easting_km / self.km_per_deg_lon
        lat = self.lat0 + northing_km / self.km_per_deg_lat
        return lat, lon


class AntarcticGrid:
    """Rectangular lat/lon grid over the Antarctic region.

    Each ``(lat_index, lon_index)`` node is a candidate navigable position.
    Land cells are tracked separately and excluded from navigation.
    """

    def __init__(
        self,
        lat_min: float = -85.0,
        lat_max: float = -55.0,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
        resolution_deg: float = DEFAULT_RESOLUTION_DEG,
    ) -> None:
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError(
                f"Invalid region bounds: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]"
            )
        if resolution_deg <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution_deg}")

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution_deg

        # Nodes are placed at the grid points themselves (like the risk grid).
        self.lats = np.arange(lat_min, lat_max + resolution_deg * 0.5, resolution_deg)
        self.lons = np.arange(lon_min, lon_max + resolution_deg * 0.5, resolution_deg)
        self.nlat = len(self.lats)
        self.nlon = len(self.lons)

        self.land_mask = np.zeros((self.nlat, self.nlon), dtype=bool)
        self.projection = GridProjection(lat0=(lat_min + lat_max) * 0.5, lon0=0.0)

    @classmethod
    def from_config(cls, config: dict) -> "AntarcticGrid":
        """Build a grid from a navigation configuration dictionary."""
        region = config.get("region", ANTARCTIC_REGION_DEFAULT)
        return cls(
            lat_min=region.get("lat_min", ANTARCTIC_REGION_DEFAULT["lat_min"]),
            lat_max=region.get("lat_max", ANTARCTIC_REGION_DEFAULT["lat_max"]),
            lon_min=region.get("lon_min", ANTARCTIC_REGION_DEFAULT["lon_min"]),
            lon_max=region.get("lon_max", ANTARCTIC_REGION_DEFAULT["lon_max"]),
            resolution_deg=config.get("grid_resolution_degrees", DEFAULT_RESOLUTION_DEG),
        )

    # ------------------------------------------------------------------ #
    # Coordinate / cell conversion
    # ------------------------------------------------------------------ #
    def contains(self, lat: float, lon: float) -> bool:
        """True if the point falls inside the grid bounds."""
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def cell_for(self, lat: float, lon: float) -> GridCell:
        """Return the nearest grid cell to (lat, lon), clamped to bounds."""
        lat = min(max(lat, self.lat_min), self.lat_max)
        lon = min(max(lon, self.lon_min), self.lon_max)
        i = int(np.argmin(np.abs(self.lats - lat)))
        j = int(np.argmin(np.abs(self.lons - lon)))
        return GridCell(i, j)

    def center_of(self, cell: GridCell) -> tuple[float, float]:
        """Return the (lat, lon) of a grid cell."""
        return float(self.lats[cell.lat_index]), float(self.lons[cell.lon_index])

    def project_cell(self, cell: GridCell) -> tuple[float, float]:
        """Project a grid cell into local northing/easting kilometers."""
        lat, lon = self.center_of(cell)
        return self.projection.project(lat, lon)

    # ------------------------------------------------------------------ #
    # Navigability / land
    # ------------------------------------------------------------------ #
    @property
    def navigable_mask(self) -> np.ndarray:
        """Boolean mask: True where a cell is navigable (not land)."""
        return ~self.land_mask

    def is_navigable(self, cell: GridCell) -> bool:
        """True if the cell is inside bounds and not marked as land."""
        if not (0 <= cell.lat_index < self.nlat and 0 <= cell.lon_index < self.nlon):
            return False
        return not bool(self.land_mask[cell.lat_index, cell.lon_index])

    def mark_land_cell(self, cell: GridCell) -> None:
        """Mark a single cell as land (not navigable)."""
        if 0 <= cell.lat_index < self.nlat and 0 <= cell.lon_index < self.nlon:
            self.land_mask[cell.lat_index, cell.lon_index] = True

    def mark_land(self, lat: float, lon: float) -> None:
        """Mark the nearest cell to (lat, lon) as land."""
        self.mark_land_cell(self.cell_for(lat, lon))

    def mark_land_box(
        self, lat_low: float, lat_high: float, lon_low: float, lon_high: float
    ) -> None:
        """Mark a lat/lon bounding box as land (clipped to grid bounds)."""
        lat_low = max(lat_low, self.lat_min)
        lat_high = min(lat_high, self.lat_max)
        lon_low = max(lon_low, self.lon_min)
        lon_high = min(lon_high, self.lon_max)

        rows = (self.lats >= lat_low) & (self.lats <= lat_high)
        cols = (self.lons >= lon_low) & (self.lons <= lon_high)
        self.land_mask[np.ix_(rows, cols)] = True

    def set_land_mask(self, mask: np.ndarray) -> None:
        """Replace the land mask with an external dataset (e.g. coastlines)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.nlat, self.nlon):
            raise ValueError(
                f"Land mask shape {mask.shape} does not match grid {(self.nlat, self.nlon)}"
            )
        self.land_mask = mask.copy()

    # ------------------------------------------------------------------ #
    # Neighbors
    # ------------------------------------------------------------------ #
    def neighbors(self, cell: GridCell, allow_diagonal: bool = True) -> list[GridCell]:
        """Return in-bounds neighbors of a cell (8- or 4-connectivity)."""
        i, j = cell
        result = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if not allow_diagonal and di != 0 and dj != 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nlat and 0 <= nj < self.nlon:
                    result.append(GridCell(ni, nj))
        return result
"""A* pathfinding over the navigation risk grid."""
from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from app.utils.geo import haversine_distance_nm


class GridCell(NamedTuple):
    lat_index: int
    lon_index: int


class AStarPlanner:
    """A* search on a lat/lon risk grid.

    Cost model combines:
        - grid cell risk (multi-objective safety)
        - great-circle distance (fuel efficiency)

    Diagonal and main-axis moves between adjacent grid points are allowed.
    """

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        risk: np.ndarray,
        safety_weight: float = 0.6,
        distance_weight: float = 0.4,
        max_risk: float = 0.95,
    ) -> None:
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("AStarPlanner expects 1-D coordinate arrays.")
        if risk.shape != (len(lats), len(lons)):
            raise ValueError("Risk grid shape does not match coordinates.")

        self.lats = lats
        self.lons = lons
        self.risk = risk
        self.safety_weight = safety_weight
        self.distance_weight = distance_weight
        self.max_risk = max_risk
        self.nlat = len(lats)
        self.nlon = len(lons)

    def _lat(self, i: int) -> float:
        return float(self.lats[i])

    def _lon(self, j: int) -> float:
        return float(self.lons[j])

    @staticmethod
    def _neighbors(cell: GridCell, nlat: int, nlon: int) -> list[GridCell]:
        """Generate 8-neighborhood cells within bounds."""
        i, j = cell
        result = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < nlat and 0 <= nj < nlon:
                    result.append(GridCell(ni, nj))
        return result

    def _step_cost(self, a: GridCell, b: GridCell) -> float:
        """Combined cost of moving between adjacent cells."""
        risk_cost = max(self.risk[a.lat_index, a.lon_index], self.risk[b.lat_index, b.lon_index])
        dist = haversine_distance_nm(
            self._lat(a.lat_index), self._lon(a.lon_index),
            self._lat(b.lat_index), self._lon(b.lon_index),
        )
        # Risk normalized to [0.1, 1.0] so traversable cells still cost something.
        normalized_risk = 0.1 + 0.9 * risk_cost
        return self.safety_weight * normalized_risk + self.distance_weight * dist

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        """Admissible heuristic: great-circle distance only."""
        return haversine_distance_nm(
            self._lat(a.lat_index), self._lon(a.lon_index),
            self._lat(b.lat_index), self._lon(b.lon_index),
        )

    def _nearest_cell(self, lat: float, lon: float) -> GridCell:
        i = int(np.argmin(np.abs(self.lats - lat)))
        j = int(np.argmin(np.abs(self.lons - lon)))
        return GridCell(i, j)

    def plan(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> list[GridCell]:
        """Find the lowest-cost path from start to goal. Returns cell indices."""
        start = self._nearest_cell(start_lat, start_lon)
        goal = self._nearest_cell(goal_lat, goal_lon)

        open_set: list[tuple[float, int, GridCell]] = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start, goal), counter, start))

        came_from: dict[GridCell, GridCell] = {}
        g_score: dict[GridCell, float] = {start: 0.0}
        f_score: dict[GridCell, float] = {start: self._heuristic(start, goal)}

        closed: set[GridCell] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            for neighbor in self._neighbors(current, self.nlat, self.nlon):
                if self.risk[neighbor.lat_index, neighbor.lon_index] > self.max_risk:
                    continue
                tentative = g_score[current] + self._step_cost(current, neighbor)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        raise RuntimeError(
            f"No safe route found between ({start_lat}, {start_lon}) and "
            f"({goal_lat}, {goal_lon}) given the maximum risk constraint."
        )

    @staticmethod
    def _reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def coordinates_from_cells(self, cells: list[GridCell]) -> list[tuple[float, float]]:
        """Convert route cells to (lat, lon) coordinate pairs."""
        return [(self._lat(c.lat_index), self._lon(c.lon_index)) for c in cells]
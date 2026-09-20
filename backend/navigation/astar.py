"""A* routing over the Antarctic navigation grid.

The step cost combines a (scaled) great-circle distance component with a
risk component taken from a per-cell cost array. Objectives such as
``shortest``, ``lowest-risk``, ``fuel-efficient`` and ``recommended`` are
produced by different (safety, distance) weight combinations applied to the
same planner.
"""
from __future__ import annotations

import heapq

import numpy as np

from navigation.geodesy import haversine_km
from navigation.grid import GridCell

DEFAULT_MAX_COST = 0.95


class NoPathFoundError(RuntimeError):
    """Raised when no navigable route exists between the start and goal."""


class AStarPlanner:
    """A* search on a lat/lon grid with configurable weighted costs.

    Parameters
    ----------
    lats, lons
        1-D node coordinate arrays for the grid.
    cost
        Optional per-cell cost array in [0, 1] (typically a risk score).
        ``None`` behaves as zero cost everywhere.
    obstacles
        Optional boolean mask of non-traversable cells. When ``None``, cells
        whose cost exceeds ``max_cost`` are treated as obstacles.
    safety_weight, distance_weight
        Weights of the risk and distance components. The step cost between
        adjacent cells is ``distance * (distance_weight + safety_weight * cost)``.
    allow_diagonal
        Whether diagonal moves are permitted (8- vs 4-connectivity).
    """

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        cost: np.ndarray | None = None,
        obstacles: np.ndarray | None = None,
        safety_weight: float = 0.6,
        distance_weight: float = 0.4,
        max_cost: float = DEFAULT_MAX_COST,
        allow_diagonal: bool = True,
    ) -> None:
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("AStarPlanner expects 1-D coordinate arrays.")
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.nlat = len(self.lats)
        self.nlon = len(self.lons)

        if cost is not None:
            cost = np.asarray(cost, dtype=float)
            if cost.shape != (self.nlat, self.nlon):
                raise ValueError("Cost array shape does not match coordinate grid.")
        self.cost = cost

        if obstacles is None:
            obstacles = (
                cost > max_cost
                if cost is not None
                else np.zeros((self.nlat, self.nlon), dtype=bool)
            )
        obstacles = np.asarray(obstacles, dtype=bool)
        if obstacles.shape != (self.nlat, self.nlon):
            raise ValueError("Obstacle mask shape does not match coordinate grid.")
        self.obstacles = obstacles

        if safety_weight < 0.0 or distance_weight < 0.0:
            raise ValueError("Cost weights must be non-negative.")
        self.safety_weight = float(safety_weight)
        self.distance_weight = float(distance_weight)
        self.max_cost = max_cost
        self.allow_diagonal = allow_diagonal

    # ------------------------------------------------------------------ #
    # Cell utilities
    # ------------------------------------------------------------------ #
    def _center(self, cell: GridCell) -> tuple[float, float]:
        return float(self.lats[cell.lat_index]), float(self.lons[cell.lon_index])

    def _is_obstacle(self, cell: GridCell) -> bool:
        return bool(self.obstacles[cell.lat_index, cell.lon_index])

    def _neighbors(self, cell: GridCell) -> list[GridCell]:
        i, j = cell
        result = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if not self.allow_diagonal and di != 0 and dj != 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nlat and 0 <= nj < self.nlon:
                    result.append(GridCell(ni, nj))
        return result

    def _cuts_blocked_corner(self, current: GridCell, neighbor: GridCell) -> bool:
        """Prevent a diagonal segment from passing through a blocked corner."""
        di = neighbor.lat_index - current.lat_index
        dj = neighbor.lon_index - current.lon_index
        if abs(di) != 1 or abs(dj) != 1:
            return False
        return (
            self._is_obstacle(GridCell(current.lat_index + di, current.lon_index))
            or self._is_obstacle(GridCell(current.lat_index, current.lon_index + dj))
        )

    def _nearest_cell(self, lat: float, lon: float) -> GridCell:
        i = int(np.argmin(np.abs(self.lats - lat)))
        j = int(np.argmin(np.abs(self.lons - lon)))
        return GridCell(i, j)

    # ------------------------------------------------------------------ #
    # Cost model
    # ------------------------------------------------------------------ #
    def _segment_km(self, a: GridCell, b: GridCell) -> float:
        la, lo = self._center(a)
        lb, lo2 = self._center(b)
        return haversine_km(la, lo, lb, lo2)

    def _step_cost(self, a: GridCell, b: GridCell) -> float:
        """Weighted cost of moving between adjacent cells."""
        distance_km = self._segment_km(a, b)
        risk = 0.0
        if self.cost is not None:
            risk = 0.5 * (
                float(self.cost[a.lat_index, a.lon_index])
                + float(self.cost[b.lat_index, b.lon_index])
            )
        return distance_km * (self.distance_weight + self.safety_weight * risk)

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        """Admissible heuristic: the distance component of the step cost."""
        return self.distance_weight * self._segment_km(a, b)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def plan(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> list[GridCell]:
        """Find the lowest-cost cell path from start to goal.

        Raises ``NoPathFoundError`` when the start/goal is blocked or no
        navigable route exists.
        """
        start = self._nearest_cell(start_lat, start_lon)
        goal = self._nearest_cell(goal_lat, goal_lon)

        if self._is_obstacle(start):
            raise NoPathFoundError(
                f"Start cell {start} is blocked (not navigable)."
            )
        if self._is_obstacle(goal):
            raise NoPathFoundError(
                f"Goal cell {goal} is blocked (not navigable)."
            )

        open_set: list[tuple[float, int, GridCell]] = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start, goal), counter, start))

        came_from: dict[GridCell, GridCell] = {}
        g_score: dict[GridCell, float] = {start: 0.0}
        closed: set[GridCell] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)
            if current in closed:
                continue
            closed.add(current)

            for neighbor in self._neighbors(current):
                if self._is_obstacle(neighbor) or self._cuts_blocked_corner(current, neighbor):
                    continue
                tentative = g_score[current] + self._step_cost(current, neighbor)
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (tentative + self._heuristic(neighbor, goal), counter, neighbor),
                    )

        raise NoPathFoundError(
            f"No navigable route found between ({start_lat}, {start_lon}) and "
            f"({goal_lat}, {goal_lon}) given the obstacle constraints."
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
        return [self._center(c) for c in cells]
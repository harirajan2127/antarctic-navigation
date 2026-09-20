"""Persistence baseline sea-ice forecaster.

The persistence forecast holds the most recent observed sea-ice
concentration constant into the future. This is the standard baseline that
more sophisticated models must beat.
"""
from __future__ import annotations

from typing import Any


class PersistenceForecaster:
    """Persistence baseline: today's concentration is tomorrow's forecast."""

    def __init__(self) -> None:
        self.is_trained = True  # persistence requires no training

    def predict(self, concentration: list[list[float]], horizon_hours: int) -> list[list[float]]:
        """Return the current concentration field unchanged."""
        return [list(map(float, row)) for row in concentration]

    def skill_note(self) -> str:
        """Honest description of baseline skill."""
        return (
            "Persistence baseline: no learning takes place. Skill is expected "
            "to be strong at short horizons and degrades with horizon length."
        )
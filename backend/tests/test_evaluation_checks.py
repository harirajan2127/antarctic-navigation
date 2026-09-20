from __future__ import annotations

import pandas as pd
import numpy as np

from navigation.grid import AntarcticGrid
from services.evaluation_checks import (
    STATUS_INSUFFICIENT,
    STATUS_UNAVAILABLE,
    evaluate_land_mask_predictions,
    evaluate_two_hour_predictions,
)


def make_grid() -> AntarcticGrid:
    return AntarcticGrid(lat_min=-76, lat_max=-74, lon_min=0, lon_max=2, resolution_deg=1)


def write_tracks(tmp_path, rows):
    path = tmp_path / "icebergs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_valid_ocean_coordinate():
    grid = make_grid()
    result = evaluate_land_mask_predictions([{"predicted_latitude": -75.0, "predicted_longitude": 1.0}], grid, np.zeros((3, 3), dtype=bool))
    assert result["ocean_points"] == 1
    assert result["accuracy_percent"] == 100.0


def test_coordinate_inside_antarctica_land():
    grid = make_grid()
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    result = evaluate_land_mask_predictions([{"predicted_latitude": -75.0, "predicted_longitude": 1.0}], grid, mask)
    assert result["land_points"] == 1
    assert result["accuracy_percent"] == 0.0


def test_coordinate_near_coastline_uses_tolerance():
    grid = make_grid()
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    result = evaluate_land_mask_predictions([{"predicted_latitude": -75.0, "predicted_longitude": 1.03}], grid, mask, coastline_tolerance_km=5)
    assert result["land_points"] == 1


def test_correct_two_hour_prediction(tmp_path):
    path = write_tracks(tmp_path, [
        {"iceberg_id": "A", "timestamp": "2024-01-01T00:00:00Z", "latitude": -75, "longitude": 1},
        {"iceberg_id": "A", "timestamp": "2024-01-01T02:00:00Z", "latitude": -75, "longitude": 1.1},
    ])
    result = evaluate_two_hour_predictions(path, predictor=lambda row, gap: (-75, 1.1))
    assert result["number_of_valid_2h_evaluations"] == 1
    assert result["average_distance_error_km"] == 0.0
    assert result["accuracy_percent"] == 100.0


def test_large_two_hour_prediction_error(tmp_path):
    path = write_tracks(tmp_path, [
        {"iceberg_id": "A", "timestamp": "2024-01-01T00:00:00Z", "latitude": -75, "longitude": 1},
        {"iceberg_id": "A", "timestamp": "2024-01-01T02:00:00Z", "latitude": -75, "longitude": 1.1},
    ])
    result = evaluate_two_hour_predictions(path, predictor=lambda row, gap: (-74, 2))
    assert result["accuracy_percent"] == 0.0
    assert result["average_distance_error_km"] > 20


def test_missing_timestamps_and_duplicate_records(tmp_path):
    path = write_tracks(tmp_path, [
        {"iceberg_id": "A", "timestamp": "bad", "latitude": -75, "longitude": 1},
        {"iceberg_id": "A", "timestamp": "2024-01-01T00:00:00Z", "latitude": -75, "longitude": 1},
        {"iceberg_id": "A", "timestamp": "2024-01-01T00:00:00Z", "latitude": -75, "longitude": 1},
        {"iceberg_id": "A", "timestamp": "2024-01-01T07:00:00Z", "latitude": -75, "longitude": 1.1},
    ])
    result = evaluate_two_hour_predictions(path)
    assert result["status"] == STATUS_INSUFFICIENT
    assert result["missing_or_invalid_records"] >= 2


def test_missing_land_mask_data():
    result = evaluate_land_mask_predictions([{"predicted_latitude": -75, "predicted_longitude": 1}], make_grid(), None)
    assert result["status"] == STATUS_UNAVAILABLE
    assert result["message"] == "Land-mask data unavailable"

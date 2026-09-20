from __future__ import annotations

import pytest

from services.accuracy_scoring import WEIGHTS, calculate_weighted_accuracy


ALL = {"sea_ice": 89.81, "iceberg": 83.58, "route": 100.0, "recalc": 50.0, "land_mask": 73.85}


def test_all_components_available_is_complete():
    result = calculate_weighted_accuracy(ALL)
    assert result["provisional"] is False
    assert result["available_weight"] == 1.0
    assert result["score"] == pytest.approx(85.2, abs=0.01)


def test_one_component_na_is_provisional():
    result = calculate_weighted_accuracy({**ALL, "recalc": None})
    assert result["provisional"] is True
    assert result["available_weighted_sum"] == pytest.approx(80.2, abs=0.01)
    assert result["available_weight"] == pytest.approx(0.9)
    assert result["score"] == pytest.approx(89.11, abs=0.01)
    assert result["missing_components"] == ["recalc"]


def test_multiple_components_na_reduce_coverage():
    result = calculate_weighted_accuracy({**ALL, "recalc": None, "land_mask": None})
    assert result["available_weight"] == pytest.approx(0.85)
    assert result["score"] == pytest.approx((31.4335 + 25.074 + 20) / 0.85, abs=0.01)


def test_true_zero_is_valid_not_na():
    result = calculate_weighted_accuracy({**ALL, "recalc": 0.0})
    row = next(row for row in result["breakdown"] if row["component"] == "recalc")
    assert row["status"] == "Valid"
    assert row["accuracy"] == 0.0
    assert result["missing_components"] == []


@pytest.mark.parametrize("value", [-0.01, 100.01, float("nan"), float("inf"), "bad"])
def test_invalid_accuracy_is_excluded(value):
    result = calculate_weighted_accuracy({**ALL, "recalc": value})
    assert result["missing_components"] == ["recalc"]
    assert result["provisional"] is True


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        calculate_weighted_accuracy(ALL, weights={**WEIGHTS, "route": 1.1})


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        calculate_weighted_accuracy(ALL, weights={**WEIGHTS, "route": 0.1})


def test_all_components_unavailable():
    result = calculate_weighted_accuracy({key: None for key in WEIGHTS})
    assert result["score"] is None
    assert result["available_weight"] == 0.0
    assert result["status"] == "N/A"

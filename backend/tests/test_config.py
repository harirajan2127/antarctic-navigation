"""Test configuration loading and constant correctness."""
from pathlib import Path

import pytest

from app.config import (
    CONFIG_DIR,
    load_ports,
    load_research_centers,
    load_simulation_config,
    load_vessels,
)


def test_config_dir_exists():
    assert CONFIG_DIR.exists()


@pytest.mark.parametrize("filename", ["ports.json", "research_centers.json",
                                      "vessels.json", "datasets.json", "simulation.json"])
def test_config_files_exist(filename):
    assert (CONFIG_DIR / filename).exists(), f"Missing {filename}"


def test_ports_have_valid_coordinates():
    ports = load_ports()
    assert len(ports) >= 5, "Expected at least five departure ports"
    for port in ports:
        assert -90.0 <= port["latitude"] <= 90.0
        assert -180.0 <= port["longitude"] <= 180.0


def test_research_centers_are_antarctic():
    centers = load_research_centers()
    assert len(centers) >= 5
    for center in centers:
        assert center["latitude"] <= -60.0, f"{center['name']} not in Antarctica"


def test_vessels_are_valid():
    vessels = load_vessels()
    assert len(vessels) >= 1
    for vessel in vessels:
        assert vessel["max_speed_knots"] > 0
        assert vessel["cruise_speed_knots"] > 0


def test_simulation_config_has_two_hour_interval():
    sim = load_simulation_config()
    assert sim["time_step_hours"] == 2, "Rolling update interval must be 2 hours"
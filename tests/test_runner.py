"""Tests for the high-level run_simulation entry point and its validation."""
import numpy as np
import pytest

from core.runner import (
    BodySpec, SimConfig, SimulationError, run_simulation,
    MAX_BODIES, MAX_POINTS,
)

G_SI = 6.67430e-11


def _two_body_config(**overrides):
    r = 1.0e8
    v = np.sqrt(G_SI * 1.0e30 / (r * 1000)) / 1000
    cfg = dict(
        bodies=[
            {"mass": 1.0e30, "position": [0, 0], "velocity": [0, 0], "name": "C"},
            {"mass": 1.0e22, "position": [r, 0], "velocity": [0, v], "name": "O"},
        ],
        duration=2 * np.pi * np.sqrt((r * 1000) ** 3 / (G_SI * 1.0e30)),
        n_points=500,
        method="rk4",
    )
    cfg.update(overrides)
    return cfg


def test_run_simulation_returns_serializable_result():
    import json
    result = run_simulation(_two_body_config())
    assert result["dim"] == 2
    assert len(result["times"]) == 500
    assert np.array(result["positions"]).shape == (500, 2, 2)
    assert result["energy_drift"] < 1e-4
    # Must be JSON-serializable for an HTTP response.
    json.dumps(result)


def test_run_simulation_accepts_dataclasses():
    cfg = _two_body_config()
    typed = SimConfig(
        bodies=[BodySpec(**b) for b in cfg["bodies"]],
        duration=cfg["duration"], n_points=100, method="verlet")
    result = run_simulation(typed)
    assert result["method"] == "verlet"
    assert len(result["times"]) == 100


def test_3d_config_runs():
    result = run_simulation(_two_body_config(bodies=[
        {"mass": 1.0e30, "position": [0, 0, 0], "velocity": [0, 0, 0]},
        {"mass": 1.0e22, "position": [1e8, 0, 0], "velocity": [0, 20, 5]},
    ]))
    assert result["dim"] == 3
    assert np.array(result["positions"]).shape[2] == 3


@pytest.mark.parametrize("overrides,match", [
    (dict(bodies=[{"mass": 1.0, "position": [0, 0], "velocity": [0, 0]}]), "2 bodies"),
    (dict(method="banana"), "Unknown method"),
    (dict(n_points=MAX_POINTS + 1), "n_points too large"),
    (dict(n_points=1), "at least 2"),
    (dict(duration=-1.0), "positive"),
])
def test_invalid_configs_raise(overrides, match):
    with pytest.raises(SimulationError, match=match):
        run_simulation(_two_body_config(**overrides))


def test_too_many_bodies_rejected():
    bodies = [{"mass": 1.0e24, "position": [i, 0], "velocity": [0, 0]}
              for i in range(MAX_BODIES + 1)]
    with pytest.raises(SimulationError, match="Too many bodies"):
        run_simulation(_two_body_config(bodies=bodies))


def test_non_finite_and_mismatched_dims_rejected():
    with pytest.raises(SimulationError):
        run_simulation(_two_body_config(bodies=[
            {"mass": 1.0e30, "position": [0, 0], "velocity": [0, 0]},
            {"mass": float("nan"), "position": [1e8, 0], "velocity": [0, 1]},
        ]))
    with pytest.raises(SimulationError, match="same dimension"):
        run_simulation(_two_body_config(bodies=[
            {"mass": 1.0e30, "position": [0, 0], "velocity": [0, 0]},
            {"mass": 1.0e22, "position": [1e8, 0, 0], "velocity": [0, 1, 0]},
        ]))

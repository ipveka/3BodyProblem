"""Tests for the FastAPI backend."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_presets():
    r = client.get("/api/presets")
    assert r.status_code == 200
    presets = r.json()
    assert len(presets) >= 3
    assert {"earth-moon", "figure-eight", "inclined-3d"} <= {p["id"] for p in presets}


def test_unknown_preset_404():
    assert client.get("/api/presets/nope").status_code == 404


@pytest.mark.parametrize("preset_id", ["earth-moon", "figure-eight", "inclined-3d"])
def test_simulate_each_preset(preset_id):
    preset = client.get(f"/api/presets/{preset_id}").json()
    r = client.post("/api/simulate", json=preset["request"])
    assert r.status_code == 200, r.text
    data = r.json()
    n = preset["request"]["n_points"]
    arr = np.array(data["positions"])
    assert arr.shape[0] == n
    assert arr.shape[1] == len(preset["request"]["bodies"])
    assert arr.shape[2] == data["dim"]
    assert len(data["times"]) == n
    # Well-posed presets conserve energy reasonably.
    assert data["energy_drift"] < 1e-2


def test_inclined_preset_is_3d():
    preset = client.get("/api/presets/inclined-3d").json()
    data = client.post("/api/simulate", json=preset["request"]).json()
    assert data["dim"] == 3


def test_simulate_rejects_single_body():
    r = client.post("/api/simulate", json={
        "bodies": [{"mass": 1.0, "position": [0, 0], "velocity": [0, 0]}],
        "duration": 100.0,
    })
    assert r.status_code == 422  # Pydantic min_length


def test_simulate_rejects_bad_method():
    r = client.post("/api/simulate", json={
        "bodies": [
            {"mass": 1e30, "position": [0, 0], "velocity": [0, 0]},
            {"mass": 1e24, "position": [1e8, 0], "velocity": [0, 1]},
        ],
        "duration": 100.0, "method": "banana",
    })
    assert r.status_code == 422


def test_simulate_rejects_mismatched_dimensions():
    # Pydantic accepts each body (2D and 3D individually valid); the engine
    # rejects the mismatch and FastAPI maps it to 422.
    r = client.post("/api/simulate", json={
        "bodies": [
            {"mass": 1e30, "position": [0, 0], "velocity": [0, 0]},
            {"mass": 1e24, "position": [1e8, 0, 0], "velocity": [0, 1, 0]},
        ],
        "duration": 100.0,
    })
    assert r.status_code == 422

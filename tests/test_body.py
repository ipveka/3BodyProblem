"""Tests for the CelestialBody class."""
import numpy as np
import pytest

from core.body import CelestialBody, PresetBodies


def test_init_stores_attributes():
    b = CelestialBody(mass=1.0e24, position=[1.0, 2.0], velocity=[3.0, 4.0],
                      name="X", color="#fff")
    assert b.mass == 1.0e24
    assert np.allclose(b.position, [1.0, 2.0])
    assert np.allclose(b.velocity, [3.0, 4.0])
    assert b.name == "X"


def test_rejects_non_positive_mass():
    with pytest.raises(ValueError):
        CelestialBody(mass=0.0, position=[0, 0], velocity=[0, 0])
    with pytest.raises(ValueError):
        CelestialBody(mass=-5.0, position=[0, 0], velocity=[0, 0])


def test_rejects_non_2d_vectors():
    with pytest.raises(ValueError):
        CelestialBody(mass=1.0, position=[0, 0, 0], velocity=[0, 0])
    with pytest.raises(ValueError):
        CelestialBody(mass=1.0, position=[0, 0], velocity=[0])


def test_kinetic_energy_matches_formula():
    # 1 kg moving at 1 km/s -> 0.5 * 1 * (1000 m/s)^2 = 5e5 J
    b = CelestialBody(mass=1.0, position=[0, 0], velocity=[1.0, 0.0])
    assert b.kinetic_energy() == pytest.approx(0.5 * 1.0 * 1000.0 ** 2)


def test_distance_to_is_symmetric():
    a = CelestialBody(mass=1.0, position=[0, 0], velocity=[0, 0])
    b = CelestialBody(mass=1.0, position=[3.0, 4.0], velocity=[0, 0])
    assert a.distance_to(b) == pytest.approx(5.0)
    assert b.distance_to(a) == pytest.approx(5.0)


def test_copy_is_independent():
    a = CelestialBody(mass=1.0, position=[0, 0], velocity=[1, 1], name="A")
    b = a.copy()
    b.update_position([9, 9])
    b.update_velocity([5, 5])
    # Mutating the copy must not touch the original.
    assert np.allclose(a.position, [0, 0])
    assert np.allclose(a.velocity, [1, 1])


def test_preset_bodies_have_sane_values():
    sun = PresetBodies.sun()
    earth = PresetBodies.earth()
    assert sun.mass > earth.mass
    assert earth.distance_to(sun) == pytest.approx(149597870.7, rel=1e-6)

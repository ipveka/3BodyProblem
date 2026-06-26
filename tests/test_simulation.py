"""Physics-level tests for the N-body engine.

These lean on conservation laws and known analytic results as oracles:
circular-orbit stability, Kepler's third law, energy/momentum conservation,
integrator ordering, and the periodic figure-eight solution.
"""
import numpy as np
import pytest

from core.body import CelestialBody
from core.simulation import NBodySimulation

G_SI = 6.67430e-11


def _energy_drift(energies):
    e = energies["total"]
    return abs(e[-1] - e[0]) / abs(e[0])


def _circular_two_body(n_points=2000, periods=1.0):
    """Heavy central body + light orbiter on a (near) circular orbit."""
    central_mass = 1.0e30  # kg
    r = 1.0e8              # km
    central = CelestialBody(central_mass, [0, 0], [0, 0], "Central")
    v = np.sqrt(G_SI * central_mass / (r * 1000)) / 1000  # km/s
    orbiter = CelestialBody(1.0e22, [r, 0], [0, v], "Orbiter")
    sim = NBodySimulation([central, orbiter])
    period = 2 * np.pi * np.sqrt((r * 1000) ** 3 / (G_SI * central_mass))  # s
    t_end = period * periods
    t_eval = np.linspace(0, t_end, n_points)
    pos, vel, t, en = sim.simulate((0, t_end), t_eval, method="rk4")
    return pos, vel, t, en, r, period


def test_requires_at_least_two_bodies():
    with pytest.raises(ValueError):
        NBodySimulation([CelestialBody(1.0, [0, 0], [0, 0])])


def test_circular_orbit_radius_is_stable():
    pos, vel, t, en, r, period = _circular_two_body()
    radius = np.sqrt(np.sum(pos[:, 1, :] ** 2, axis=1))
    # Light orbiter, so the orbit is very nearly circular.
    assert np.max(np.abs(radius - r)) / r < 0.02


def test_kepler_third_law_orbit_closes():
    # After exactly one predicted period the orbiter returns to its start.
    pos, vel, t, en, r, period = _circular_two_body(periods=1.0)
    start = pos[0, 1]
    end = pos[-1, 1]
    assert np.linalg.norm(end - start) / r < 0.02


def test_rk4_conserves_energy():
    pos, vel, t, en, r, period = _circular_two_body()
    assert _energy_drift(en) < 1e-4


def test_total_momentum_is_conserved():
    pos, vel, t, en, r, period = _circular_two_body()
    masses = np.array([1.0e30, 1.0e22])
    # p(t) = Σ m_i v_i ; should be constant (no external forces).
    p = np.einsum("i,tij->tj", masses, vel)
    p0 = p[0]
    drift = np.linalg.norm(p - p0, axis=1).max()
    scale = np.linalg.norm(masses[:, None] * vel[0]).max()
    assert drift / scale < 1e-6


def test_euler_drifts_more_than_rk4():
    central = CelestialBody(1.0e30, [0, 0], [0, 0], "C")
    r = 1.0e8
    v = np.sqrt(G_SI * 1.0e30 / (r * 1000)) / 1000
    orbiter = CelestialBody(1.0e22, [r, 0], [0, v], "O")
    period = 2 * np.pi * np.sqrt((r * 1000) ** 3 / (G_SI * 1.0e30))
    t_eval = np.linspace(0, period, 1000)

    sim = NBodySimulation([central, orbiter])
    _, _, _, en_e = sim.simulate((0, period), t_eval, method="euler")
    sim2 = NBodySimulation([central, orbiter])
    _, _, _, en_r = sim2.simulate((0, period), t_eval, method="rk4")

    assert _energy_drift(en_e) > _energy_drift(en_r)


def _figure_eight():
    p = np.array([0.97000436, -0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    return [
        CelestialBody(1.0, list(p), list(-v3 / 2), "B1"),
        CelestialBody(1.0, list(-p), list(-v3 / 2), "B2"),
        CelestialBody(1.0, [0.0, 0.0], list(v3), "B3"),
    ]


def test_figure_eight_is_periodic_in_normalized_units():
    period = 6.3259
    sim = NBodySimulation(_figure_eight(), G=1.0, length_unit=1.0)
    t_eval = np.linspace(0, period, 4000)
    pos, vel, t, en = sim.simulate((0, period), t_eval, method="rk4")
    # Each body should return close to where it started after one period.
    for i in range(3):
        assert np.linalg.norm(pos[-1, i] - pos[0, i]) < 0.05
    # And it must actually move (not a degenerate static config).
    assert np.ptp(pos[:, 0, 0]) > 1.0


def test_vectorized_accelerations_match_bruteforce():
    """The vectorized acceleration must equal a naive O(N²) reference."""
    rng = np.random.default_rng(0)
    bodies = [CelestialBody(rng.uniform(1e22, 1e30),
                            list(rng.uniform(-1e8, 1e8, 2)),
                            list(rng.uniform(-10, 10, 2)))
              for _ in range(6)]
    sim = NBodySimulation(bodies)
    state = sim._state_vector()
    acc = sim._compute_accelerations(state).reshape(sim.n_bodies, sim.dim)

    pos = state.reshape(sim.n_bodies, sim.stride)[:, :sim.dim]
    ref = np.zeros_like(acc)
    for i in range(sim.n_bodies):
        for j in range(sim.n_bodies):
            if i != j:
                disp = pos[j] - pos[i]
                dist = max(np.sqrt(np.sum(disp ** 2)), 1e-3)
                ref[i] += sim.G_eff * sim.masses[j] * disp / dist ** 3
    assert np.allclose(acc, ref)


def test_3d_inclined_orbit_is_stable_and_conserves_energy():
    central = CelestialBody(1.0e30, [0, 0, 0], [0, 0, 0], "C")
    r = 1.0e8
    v = np.sqrt(G_SI * 1.0e30 / (r * 1000)) / 1000
    # Circular orbit in a plane tilted 45° out of the xy-plane.
    c = np.sqrt(0.5)
    orbiter = CelestialBody(1.0e22, [r, 0, 0], [0, v * c, v * c], "O")
    sim = NBodySimulation([central, orbiter])
    assert sim.dim == 3
    period = 2 * np.pi * np.sqrt((r * 1000) ** 3 / (G_SI * 1.0e30))
    t_eval = np.linspace(0, period, 2000)
    pos, vel, t, en = sim.simulate((0, period), t_eval, method="rk4")

    radius = np.sqrt(np.sum(pos[:, 1, :] ** 2, axis=1))
    assert np.max(np.abs(radius - r)) / r < 0.02
    assert _energy_drift(en) < 1e-4
    # The orbit genuinely uses the third dimension.
    assert np.ptp(pos[:, 1, 2]) > 0.5 * r


def test_g_default_matches_explicit_si():
    """Passing the default G must reproduce the implicit-default behaviour."""
    bodies = [CelestialBody(1.0e30, [0, 0], [0, 0]),
              CelestialBody(1.0e24, [1.0e8, 0], [0, 1.0])]
    t_eval = np.linspace(0, 1.0e6, 200)
    a = NBodySimulation([b.copy() for b in bodies]).simulate(
        (0, 1.0e6), t_eval, method="rk4")[0]
    b = NBodySimulation([bb.copy() for bb in bodies], G=6.67430e-11,
                        length_unit=1000.0).simulate(
        (0, 1.0e6), t_eval, method="rk4")[0]
    assert np.allclose(a, b)

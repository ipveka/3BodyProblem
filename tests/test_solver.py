"""Tests for the numerical ODE solvers in isolation."""
import numpy as np
import pytest

from core.solver import RungeKuttaSolver, EulerSolver


def test_rk4_solves_exponential_growth():
    # y' = y, y(0) = 1  ->  y(t) = e^t. Use the [x, v] layout the solvers expect.
    def f(t, y):
        return y.copy()

    y0 = np.array([1.0, 0.0, 0.0, 0.0])
    t_eval = np.linspace(0, 2.0, 200)
    sol, times = RungeKuttaSolver().solve(f, y0, t_eval, dt=t_eval[1] - t_eval[0])
    assert sol[-1, 0] == pytest.approx(np.exp(2.0), rel=1e-4)


def test_rk4_is_more_accurate_than_euler():
    def f(t, y):
        return y.copy()

    y0 = np.array([1.0, 0.0, 0.0, 0.0])
    t_eval = np.linspace(0, 2.0, 100)
    dt = t_eval[1] - t_eval[0]
    exact = np.exp(2.0)
    rk4 = RungeKuttaSolver().solve(f, y0, t_eval, dt)[0][-1, 0]
    euler = EulerSolver().solve(f, y0, t_eval, dt)[0][-1, 0]
    assert abs(rk4 - exact) < abs(euler - exact)


def test_solver_hits_all_eval_points():
    def f(t, y):
        return np.zeros_like(y)

    y0 = np.zeros(4)
    t_eval = np.linspace(0, 1.0, 50)
    sol, times = EulerSolver().solve(f, y0, t_eval, dt=0.02)
    assert len(times) == len(t_eval)
    assert np.allclose(times, t_eval)

"""
High-level entry point for running N-body simulations.

This module provides a single, validated function -- :func:`run_simulation` --
that turns a plain configuration (dataclasses or dicts) into a JSON-serializable
result. It is the seam that user-facing layers (the Streamlit app, a FastAPI
backend, scripts) should call instead of wiring up ``NBodySimulation`` by hand,
so that input validation and resource limits live in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np

from .body import CelestialBody
from .simulation import NBodySimulation

# Resource limits. These guard the engine against untrusted input (an open
# HTTP endpoint must never accept an unbounded body count or step count).
MAX_BODIES = 50
MAX_POINTS = 100_000
VALID_METHODS = ("rk4", "euler", "verlet", "scipy")


class SimulationError(ValueError):
    """Raised when a simulation configuration is invalid."""


@dataclass
class BodySpec:
    """Plain description of a single body (no NumPy required to construct)."""
    mass: float
    position: Sequence[float]
    velocity: Sequence[float]
    name: str = "Body"
    color: str = "#FF0000"


@dataclass
class SimConfig:
    """Validated description of a simulation run."""
    bodies: List[BodySpec]
    duration: float
    n_points: int = 1000
    method: str = "rk4"
    G: float = 6.67430e-11
    length_unit: float = 1000.0


def _coerce_config(config: Union[SimConfig, dict]) -> SimConfig:
    if isinstance(config, SimConfig):
        return config
    data = dict(config)
    raw_bodies = data.get("bodies", [])
    bodies = [b if isinstance(b, BodySpec) else BodySpec(**b) for b in raw_bodies]
    return SimConfig(
        bodies=bodies,
        duration=data["duration"],
        n_points=data.get("n_points", 1000),
        method=data.get("method", "rk4"),
        G=data.get("G", 6.67430e-11),
        length_unit=data.get("length_unit", 1000.0),
    )


def _validate(config: SimConfig) -> int:
    """Validate a config and return the spatial dimension."""
    n = len(config.bodies)
    if n < 2:
        raise SimulationError("At least 2 bodies are required")
    if n > MAX_BODIES:
        raise SimulationError(f"Too many bodies (max {MAX_BODIES})")

    if config.method.lower() not in VALID_METHODS:
        raise SimulationError(
            f"Unknown method '{config.method}'. Valid: {', '.join(VALID_METHODS)}")

    if not np.isfinite(config.duration) or config.duration <= 0:
        raise SimulationError("duration must be a positive, finite number")
    if config.n_points < 2:
        raise SimulationError("n_points must be at least 2")
    if config.n_points > MAX_POINTS:
        raise SimulationError(f"n_points too large (max {MAX_POINTS})")
    if not np.isfinite(config.G) or not np.isfinite(config.length_unit):
        raise SimulationError("G and length_unit must be finite")
    if config.length_unit <= 0:
        raise SimulationError("length_unit must be positive")

    dim = len(config.bodies[0].position)
    if dim not in (2, 3):
        raise SimulationError("Bodies must be 2D or 3D")

    for b in config.bodies:
        if not np.isfinite(b.mass) or b.mass <= 0:
            raise SimulationError(f"Body '{b.name}' must have positive finite mass")
        if len(b.position) != dim or len(b.velocity) != dim:
            raise SimulationError("All bodies must share the same dimension")
        if not (np.all(np.isfinite(b.position)) and np.all(np.isfinite(b.velocity))):
            raise SimulationError(f"Body '{b.name}' has non-finite position/velocity")

    return dim


def run_simulation(config: Union[SimConfig, dict]) -> Dict:
    """
    Run a validated N-body simulation.

    Args:
        config: A :class:`SimConfig` (or an equivalent dict). ``duration`` is in
            seconds for SI/km systems, or in normalized time units when
            ``G``/``length_unit`` are set to 1.

    Returns:
        A JSON-serializable dict with keys:
            ``dim``, ``method``, ``times`` (list[T]),
            ``bodies`` (list of {name, color, mass}),
            ``positions`` and ``velocities`` (nested lists, [T][N][dim]),
            ``energies`` ({kinetic, potential, total}),
            ``energy_drift`` (relative, dimensionless).

    Raises:
        SimulationError: if the configuration is invalid.
    """
    config = _coerce_config(config)
    _validate(config)

    bodies = [
        CelestialBody(mass=b.mass, position=list(b.position),
                      velocity=list(b.velocity), name=b.name, color=b.color)
        for b in config.bodies
    ]

    sim = NBodySimulation(bodies, G=config.G, length_unit=config.length_unit)
    t_eval = np.linspace(0.0, config.duration, config.n_points)
    positions, velocities, times, energies = sim.simulate(
        t_span=(0.0, config.duration), t_eval=t_eval, method=config.method)

    e_total = energies["total"]
    drift = (abs(e_total[-1] - e_total[0]) / abs(e_total[0])
             if e_total[0] != 0 else 0.0)

    return {
        "dim": sim.dim,
        "method": config.method.lower(),
        "times": times.tolist(),
        "bodies": [{"name": b.name, "color": b.color, "mass": b.mass}
                   for b in bodies],
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "energies": {k: v.tolist() for k, v in energies.items()},
        "energy_drift": float(drift),
    }

"""Built-in example systems exposed via the API.

Each preset is a ready-to-run :class:`SimRequest`, so the frontend can offer a
"load preset -> simulate" flow without hardcoding physics on the client.
"""
import math
from typing import List, Optional

from .schemas import BodyIn, PresetOut, SimRequest

G_SI = 6.67430e-11
DAY = 86400.0


def _circular_speed(central_mass_kg: float, radius_km: float) -> float:
    """Circular orbital speed in km/s for a body orbiting a fixed central mass."""
    return math.sqrt(G_SI * central_mass_kg / (radius_km * 1000.0)) / 1000.0


def _earth_moon() -> PresetOut:
    r = 384400.0
    v = _circular_speed(5.972e24, r)
    return PresetOut(
        id="earth-moon",
        name="Earth–Moon",
        description="Classic two-body system: the Moon on a circular orbit.",
        request=SimRequest(
            bodies=[
                BodyIn(mass=5.972e24, position=[0, 0], velocity=[0, 0],
                       name="Earth", color="#4a90e2"),
                BodyIn(mass=7.342e22, position=[r, 0], velocity=[0, v],
                       name="Moon", color="#b0b0b0"),
            ],
            duration=30 * DAY, n_points=1000, method="rk4",
        ),
    )


def _sun_earth_moon() -> PresetOut:
    scale = 1000.0  # distances scaled for visualization; masses stay realistic
    earth_r = 149597870.7 / scale
    earth_v = _circular_speed(1.989e30, earth_r)
    moon_r = 384400.0 / scale
    moon_v = _circular_speed(5.972e24, moon_r)
    return PresetOut(
        id="sun-earth-moon",
        name="Sun–Earth–Moon",
        description="Three-body system (distances scaled 1000x for viewing).",
        request=SimRequest(
            bodies=[
                BodyIn(mass=1.989e30, position=[0, 0], velocity=[0, 0],
                       name="Sun", color="#ffd700"),
                BodyIn(mass=5.972e24, position=[earth_r, 0], velocity=[0, earth_v],
                       name="Earth", color="#4a90e2"),
                BodyIn(mass=7.342e22, position=[earth_r + moon_r, 0],
                       velocity=[0, earth_v + moon_v], name="Moon", color="#b0b0b0"),
            ],
            duration=365 * DAY, n_points=1500, method="rk4",
        ),
    )


def _figure_eight() -> PresetOut:
    px, py = 0.97000436, -0.24308753
    v3x, v3y = -0.93240737, -0.86473146
    return PresetOut(
        id="figure-eight",
        name="Figure-Eight",
        description="Periodic three-body choreography (normalized units, G=1).",
        request=SimRequest(
            bodies=[
                BodyIn(mass=1.0, position=[px, py], velocity=[-v3x / 2, -v3y / 2],
                       name="Body 1", color="#e74c3c"),
                BodyIn(mass=1.0, position=[-px, -py], velocity=[-v3x / 2, -v3y / 2],
                       name="Body 2", color="#2ecc71"),
                BodyIn(mass=1.0, position=[0.0, 0.0], velocity=[v3x, v3y],
                       name="Body 3", color="#3498db"),
            ],
            duration=13.0, n_points=2000, method="rk4", G=1.0, length_unit=1.0,
        ),
    )


def _inclined_3d() -> PresetOut:
    central, r = 1.0e30, 1.0e8
    v = _circular_speed(central, r)
    c = math.sqrt(0.5)  # 45-degree inclination
    period = 2 * math.pi * math.sqrt((r * 1000) ** 3 / (G_SI * central))
    return PresetOut(
        id="inclined-3d",
        name="Inclined Orbit (3D)",
        description="A circular orbit tilted 45° out of plane — shows 3D support.",
        request=SimRequest(
            bodies=[
                BodyIn(mass=central, position=[0, 0, 0], velocity=[0, 0, 0],
                       name="Star", color="#ffd700"),
                BodyIn(mass=1.0e22, position=[r, 0, 0], velocity=[0, v * c, v * c],
                       name="Planet", color="#9b59b6"),
            ],
            duration=period, n_points=1500, method="rk4",
        ),
    )


_PRESETS: List[PresetOut] = [
    _earth_moon(), _sun_earth_moon(), _figure_eight(), _inclined_3d(),
]


def list_presets() -> List[PresetOut]:
    return _PRESETS


def get_preset(preset_id: str) -> Optional[PresetOut]:
    return next((p for p in _PRESETS if p.id == preset_id), None)

"""Pydantic request/response models.

These models *are* the API contract: FastAPI generates an OpenAPI schema from
them, from which the frontend's TypeScript types can be generated
(`openapi-typescript`), keeping the two sides in sync. They mirror
``core.runner.SimConfig`` but add HTTP-friendly bounds and documentation.
"""
from typing import List, Literal

from pydantic import BaseModel, Field

from core.runner import MAX_BODIES, MAX_POINTS

Method = Literal["rk4", "euler", "verlet", "scipy"]


class BodyIn(BaseModel):
    mass: float = Field(gt=0, description="Mass in kilograms (must be positive).")
    position: List[float] = Field(min_length=2, max_length=3,
                                  description="Position vector (2D or 3D).")
    velocity: List[float] = Field(min_length=2, max_length=3,
                                  description="Velocity vector, same dim as position.")
    name: str = "Body"
    color: str = "#FF0000"


class SimRequest(BaseModel):
    bodies: List[BodyIn] = Field(min_length=2, max_length=MAX_BODIES)
    duration: float = Field(gt=0, description="Total simulated time (seconds, or "
                            "normalized units when G/length_unit are 1).")
    n_points: int = Field(default=1000, ge=2, le=MAX_POINTS)
    method: Method = "rk4"
    G: float = Field(default=6.67430e-11, description="Gravitational constant (SI).")
    length_unit: float = Field(default=1000.0, gt=0,
                               description="Meters per position unit (km -> 1000).")


class BodyOut(BaseModel):
    name: str
    color: str
    mass: float


class Energies(BaseModel):
    kinetic: List[float]
    potential: List[float]
    total: List[float]


class SimResponse(BaseModel):
    dim: int
    method: str
    times: List[float]
    bodies: List[BodyOut]
    # Trajectory arrays, indexed [time][body][dim].
    positions: List[List[List[float]]]
    velocities: List[List[List[float]]]
    energies: Energies
    energy_drift: float


class PresetOut(BaseModel):
    id: str
    name: str
    description: str
    request: SimRequest

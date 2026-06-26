"""FastAPI application exposing the N-body simulation engine.

Run locally with:

    uvicorn backend.main:app --reload

Interactive docs are then available at http://localhost:8000/docs
"""
import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.runner import SimulationError, run_simulation

from .presets import get_preset, list_presets
from .schemas import PresetOut, SimRequest, SimResponse

app = FastAPI(
    title="N-Body Simulation API",
    version="1.0.0",
    description="Gravitational N-body simulation with 2D/3D support.",
)

# CORS: allow the frontend's origin(s). Defaults to permissive for local dev;
# set ALLOWED_ORIGINS (comma-separated) in production.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if _origins.strip() == "*" else [
    o.strip() for o in _origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/api/presets", response_model=List[PresetOut])
def get_presets() -> List[PresetOut]:
    """List the built-in example systems."""
    return list_presets()


@app.get("/api/presets/{preset_id}", response_model=PresetOut)
def get_one_preset(preset_id: str) -> PresetOut:
    """Fetch a single preset by id."""
    preset = get_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{preset_id}'")
    return preset


@app.post("/api/simulate", response_model=SimResponse)
def simulate(request: SimRequest) -> dict:
    """Run a simulation and return the full trajectory + energy diagnostics."""
    try:
        return run_simulation(request.model_dump())
    except SimulationError as exc:
        # Engine-level validation (e.g. mismatched dims, non-finite values)
        # that Pydantic bounds don't cover.
        raise HTTPException(status_code=422, detail=str(exc))

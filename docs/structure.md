# Application Structure

This document describes the architecture of the 3BodyProblem N-body
gravitational simulation.

## Overview

The project is organized into clear layers so each concern can evolve
independently:

```
3BodyProblem/
├── core/                   # Simulation engine (NumPy/SciPy)
│   ├── body.py             # CelestialBody class
│   ├── simulation.py       # NBodySimulation engine (2D/3D, vectorized)
│   ├── solver.py           # Numerical integration methods
│   └── runner.py           # Validated run_simulation() entry point
├── backend/                # FastAPI REST API
│   ├── main.py             # App + routes
│   ├── schemas.py          # Pydantic models (the OpenAPI contract)
│   └── presets.py          # Built-in example systems
├── frontend/               # React + TypeScript + Vite (three.js viewer)
│   └── src/
│       ├── api/            # Generated types + fetch client
│       ├── components/     # Viewer3D, Controls, EnergyChart, ...
│       └── store.ts        # Zustand state
├── visualization/          # Matplotlib/Plotly plotting (used by scripts)
├── scripts/                # Two-body / three-body demo scripts
├── tests/                  # Pytest suite
├── Dockerfile, docker-compose.yml, render.yaml   # Deployment
└── requirements.txt, requirements-api.txt, pyproject.toml
```

The data flow is: **frontend → backend → `core`**. The browser sends a
configuration to the API, which validates it and calls the engine, then returns
the full trajectory and energy diagnostics for the client to render.

## Core engine (`core/`)

### `body.py` — `CelestialBody`
Represents a single body (mass, position, velocity, name, color). Positions and
velocities may be 2D or 3D. Includes helpers for kinetic energy, distance, and
copying.

### `simulation.py` — `NBodySimulation`
The physics engine. It:
- infers the spatial dimension (2D/3D) from the bodies;
- computes pairwise gravitational accelerations with a **vectorized** NumPy
  formulation (`einsum`), with configurable `G` and `length_unit` so both
  realistic (SI/km) and normalized (e.g. figure-eight, `G=1`) systems work;
- integrates the equations of motion via the chosen solver;
- tracks kinetic/potential/total energy (vectorized, with chunked memory use).

### `solver.py` — numerical integrators
`ODESolver` base class with `RungeKuttaSolver` (RK4), `EulerSolver`,
`VerletSolver` (velocity Verlet), and `AdaptiveRungeKuttaSolver`. The simulation
can also delegate to SciPy's `solve_ivp`.

### `runner.py` — `run_simulation()`
The single validated entry point. It takes a `SimConfig`/`BodySpec` (or a plain
dict), enforces resource limits (`MAX_BODIES`, `MAX_POINTS`) and input validity
(`SimulationError`), runs the simulation, and returns a JSON-serializable
result. Every user-facing layer should call this rather than wiring up
`NBodySimulation` directly.

## Backend (`backend/`)

A FastAPI application exposing the engine over HTTP:

- `GET /health` — liveness probe
- `GET /api/presets`, `GET /api/presets/{id}` — built-in example systems
- `POST /api/simulate` — run a simulation

`schemas.py` defines the Pydantic models that form the OpenAPI contract;
engine-level `SimulationError`s are mapped to HTTP 422. CORS origins are
configurable via the `ALLOWED_ORIGINS` environment variable.

## Frontend (`frontend/`)

A React + TypeScript single-page app built with Vite:

- **Viewer3D** renders bodies and trajectories in 3D with react-three-fiber.
- **Controls** drive presets, integration method, duration, step count, and
  playback.
- **EnergyChart** plots normalized kinetic/potential/total energy.
- The API client's types are generated from the backend's OpenAPI schema
  (`npm run gen:types`), keeping the two sides in sync.

## Scripts & visualization

`scripts/run_two_body.py` and `scripts/run_three_body.py` are standalone
Matplotlib demos that exercise the engine directly. `visualization/` holds the
plotting/animation helpers they use.

## Testing

`tests/` contains the pytest suite: `CelestialBody` behaviour, the solvers, the
engine (conservation laws, Kepler's third law, vectorization equivalence, 2D and
3D), the `run_simulation` entry point and its validation, and the FastAPI
endpoints. CI runs the suite on Python 3.11/3.12 and builds the frontend.

## Deployment

`Dockerfile` builds a lean backend image (engine + API only). `render.yaml`
provisions a Render web service for the API and a static site for the frontend;
`docker-compose.yml` brings the whole stack up locally. See the README for
details.

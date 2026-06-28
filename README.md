# 3BodyProblem: N-Body Gravitational Simulation

A comprehensive Python implementation of N-body gravitational simulations with interactive visualization. It demonstrates two- and three-body gravitational dynamics, energy conservation, and chaotic orbital mechanics. The stack is a dimension-agnostic (2D/3D) NumPy engine, a FastAPI backend, and a React + three.js frontend — plus standalone demo scripts.

## 🌌 Features

### Core Physics Engine
- **Newtonian Gravity Simulation**: Accurate gravitational force calculations between N celestial bodies
- **Multiple Integration Methods**: 
  - Runge-Kutta 4th order (RK4) for high accuracy
  - Forward Euler method for comparison
  - Velocity Verlet method for better energy conservation
  - Adaptive Runge-Kutta with error control
- **Energy Conservation Tracking**: Monitor kinetic, potential, and total energy throughout simulations
- **Customizable Time Integration**: Adjustable time steps and simulation duration

### Web App (React + three.js)
- **3D Visualization**: Animated trajectories rendered with react-three-fiber
- **Preset Configurations**: Pre-built systems (Earth-Moon, Sun-Earth-Moon, figure-eight, 3D inclined orbit)
- **Parameter Control**: Integration method, duration, and step count
- **Energy Diagnostics**: Live energy-conservation chart and drift readout
- **FastAPI Backend**: Typed REST API with an OpenAPI contract

### Visualization Capabilities
- **2D Orbital Trajectories**: Static and animated orbit visualization
- **Energy Conservation Plots**: Track energy drift and conservation
- **Phase Space Diagrams**: Position-velocity relationships
- **Inter-body Distance Analysis**: Monitor body separations over time
- **Interactive Animations**: Playback controls and time sliders

### Demonstration Scripts
- **Two-Body Systems**: Earth-Moon, binary stars, eccentric orbits
- **Three-Body Systems**: Sun-Earth-Moon, Lagrange triangles, figure-eight orbits
- **Chaotic Dynamics**: Demonstration of sensitive dependence on initial conditions
- **Integration Method Comparison**: Accuracy and stability analysis

## 📁 Project Structure

```
3BodyProblem/
├── core/                   # Simulation engine (NumPy)
│   ├── body.py             # CelestialBody class
│   ├── simulation.py       # NBodySimulation engine (2D/3D, vectorized)
│   ├── solver.py           # Numerical integration methods
│   └── runner.py           # Validated run_simulation entry point
├── backend/                # FastAPI REST API
│   ├── main.py             # App + routes
│   ├── schemas.py          # Pydantic models (OpenAPI contract)
│   └── presets.py          # Built-in example systems
├── frontend/               # React + TypeScript + Vite (three.js viewer)
├── visualization/          # Matplotlib/Plotly plotting (used by scripts)
├── scripts/                # Two-body / three-body demo scripts
├── tests/                  # Pytest suite
├── Dockerfile, docker-compose.yml, render.yaml   # Deployment
├── requirements.txt, requirements-api.txt, pyproject.toml
└── README.md
```

## 🚀 Quick Start

Run the full web app (backend + frontend):

```bash
# Backend (terminal 1)
pip install -e ".[api]"
uvicorn backend.main:app --reload          # http://localhost:8000/docs

# Frontend (terminal 2)
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Or bring up everything with Docker: `docker compose up --build`.

### Running the demo scripts

The scripts use matplotlib/plotly, so install the full project (not the lean
`requirements.txt`, which is only the serverless backend runtime):

```bash
pip install -e .
python scripts/run_two_body.py
python scripts/run_three_body.py
```

## 🧩 Using the engine as a library

The simulation core is dimension-agnostic (**2D or 3D**) and exposes a single
validated entry point, `core.runner.run_simulation`, which returns a
JSON-serializable result. This is the seam intended for backends (e.g. FastAPI):

```python
from core.runner import run_simulation

result = run_simulation({
    "bodies": [
        {"mass": 1.0e30, "position": [0, 0, 0], "velocity": [0, 0, 0], "name": "Star"},
        {"mass": 1.0e24, "position": [1.0e8, 0, 0], "velocity": [0, 25, 5], "name": "Planet"},
    ],
    "duration": 30 * 86400,   # seconds (or normalized time units when G=1)
    "n_points": 1000,
    "method": "rk4",          # "rk4" | "euler" | "verlet" | "scipy"
})

result["positions"]    # nested list, shape [n_times][n_bodies][dim]
result["energy_drift"] # relative total-energy drift (lower is better)
```

Invalid configurations raise `core.runner.SimulationError`, and resource limits
(`MAX_BODIES`, `MAX_POINTS`) guard against unbounded input.

## 🔌 REST API (FastAPI)

A FastAPI backend wraps the engine for use by a web frontend or other clients.

```bash
pip install -e ".[api]"
uvicorn backend.main:app --reload
```

Then open the interactive docs at http://localhost:8000/docs. Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/presets` | List built-in example systems |
| `GET` | `/api/presets/{id}` | Fetch one preset (ready-to-run request) |
| `POST` | `/api/simulate` | Run a simulation, return trajectory + energies |

Example:

```bash
curl -s localhost:8000/api/presets/figure-eight \
  | python -c "import sys,json,urllib.request as u; req=json.load(sys.stdin)['request']; \
print(urllib.request.urlopen(u.Request('http://localhost:8000/api/simulate', \
data=json.dumps(req).encode(), headers={'Content-Type':'application/json'})).read()[:120])"
```

Set `ALLOWED_ORIGINS` (comma-separated) to restrict CORS in production.

## 💻 Web frontend (React + three.js)

A React + TypeScript + Vite single-page app renders trajectories in 3D
(react-three-fiber) and talks to the FastAPI backend. Its API types are
generated from the backend's OpenAPI schema, so the two stay in sync.

```bash
# 1. Start the backend (in one terminal)
uvicorn backend.main:app --reload

# 2. Start the frontend (in another)
cd frontend
npm install
npm run dev        # http://localhost:5173
```

The frontend reads the backend URL from `VITE_API_URL` (see
`frontend/.env.example`; defaults to `http://localhost:8000`). To regenerate the
API types after changing the backend schema:

```bash
python -c "import json; from backend.main import app; \
open('frontend/openapi.json','w').write(json.dumps(app.openapi(), indent=2))"
cd frontend && npm run gen:types
```

## 🧪 Testing

Install the dev dependencies and run the suite with `pytest`:

```bash
pip install -e ".[dev]"
pytest
```

The tests use physical conservation laws and known analytic results as oracles
(circular-orbit stability, Kepler's third law, energy/momentum conservation,
integrator ordering, and the periodic figure-eight solution). CI runs them on
Python 3.11 and 3.12 via GitHub Actions.

## 🚀 Deploy (API + React frontend)

### Local: Docker Compose

Bring up the whole stack with one command:

```bash
docker compose up --build
# Backend  -> http://localhost:8000  (docs at /docs)
# Frontend -> http://localhost:5173
```

The backend image (`Dockerfile`) ships only the engine + API with a lean
dependency set (`requirements-api.txt`) — no plotting/visualization stack.

### Render (Blueprint)

A `render.yaml` blueprint provisions both services:

1. Push this repo to GitHub.
2. In Render: **New → Blueprint** and select the repo.
3. Render creates a **web service** (`3body-api`, FastAPI) and a **static site**
   (`3body-frontend`, the built React app).

The two services cross-reference each other by URL via env vars in `render.yaml`:

- `VITE_API_URL` (frontend, build-time) → the backend URL
- `ALLOWED_ORIGINS` (backend, CORS) → the frontend URL

They default to `https://3body-api.onrender.com` and
`https://3body-frontend.onrender.com`. If Render assigns different URLs, update
those two values in `render.yaml` (or in the dashboard) and redeploy.

### Vercel (all-in-one, serverless)

The whole app can run on Vercel as a single project: the React app on the CDN
and the FastAPI backend as a Python **serverless function**. This works because
the API is stateless (request → simulate → response). Config lives in
`vercel.json`.

1. Push this repo to GitHub.
2. In Vercel: **Add New → Project** and import the repo. Accept the defaults
   (`vercel.json` provides the build command, output dir, and routing).
3. Deploy. The frontend is served at `/`; `/api/*` is routed to the function.

Notes:
- The function ships a minimal dependency set (the root `requirements.txt`:
  fastapi + numpy — Vercel detects Python from it). SciPy is imported lazily and
  is **not** bundled, keeping the function well under the serverless size limit.
  The `"scipy"` integration method is therefore unavailable on Vercel — use
  `rk4`/`verlet`/`euler`.
- No env var is required: the frontend defaults to **same-origin** in
  production, and the API is same-origin (so no CORS).
- Serverless limits to keep in mind on the free **Hobby** tier: function
  duration (`maxDuration` is set to 60s) and response size (~4.5 MB). Very large
  runs (high body count × many steps) may hit these — keep `n_points` modest.
  Hobby is free for personal/non-commercial use.

## 📜 License

MIT — see [LICENSE](LICENSE).


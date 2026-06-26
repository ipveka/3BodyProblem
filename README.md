# 3BodyProblem: N-Body Gravitational Simulation

A comprehensive Python implementation of N-body gravitational simulations with interactive visualization capabilities. This project demonstrates two-body and three-body gravitational dynamics, energy conservation, and chaotic orbital mechanics through both standalone scripts and an interactive Streamlit web application.

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

### Interactive Streamlit Interface
- **Real-time Parameter Control**: Adjust simulation parameters on the fly
- **Multiple Body Management**: Add, remove, and configure celestial bodies
- **Preset Configurations**: Pre-built systems (Earth-Moon, Sun-Earth-Moon, etc.)
- **Live Visualization**: Animated trajectories with interactive controls
- **Statistical Analysis**: Real-time energy and orbital statistics

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
├── app/
│   └── app.py              # Streamlit web application
├── core/
│   ├── body.py             # CelestialBody class
│   ├── simulation.py       # NBodySimulation engine
│   └── solver.py           # Numerical integration methods
├── visualization/
│   ├── plot.py             # Static plotting functions
│   └── animate.py          # Animation functions
├── scripts/
│   ├── run_two_body.py     # Two-body demo script
│   └── run_three_body.py   # Three-body demo script
├── run_app.py              # Application launcher with dependency checks
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python run_app.py
   ```

3. **Or run Streamlit directly:**
   ```bash
   streamlit run app/app.py
   ```

### Running Demo Scripts

```bash
# Two-body system demonstration
python scripts/run_two_body.py

# Three-body system demonstration
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
dependency set (`requirements-api.txt`) — no Streamlit/visualization stack.

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

## 🌐 Deploying the Streamlit app

### Using `run_app.py`

The `run_app.py` script automatically:
- ✅ Checks Python version (3.11+)
- ✅ Verifies all required packages are installed
- ✅ Validates project structure
- ✅ Launches Streamlit with correct configuration

### Deployment Platforms

#### ⚠️ Important Note About Vercel

**Streamlit applications are NOT well-suited for Vercel** because Vercel is designed for serverless functions, while Streamlit requires a persistent server process with WebSocket connections. For Vercel deployment, you would need to refactor the application significantly.

#### ✅ Recommended Platforms

**1. Streamlit Cloud (Easiest)**
- Push your code to GitHub
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your repository
- Deploy automatically (free tier available)

**2. Railway**
```bash
# Set these in Railway dashboard:
Build Command: pip install -r requirements.txt
Start Command: python run_app.py
```

**3. Render**
```bash
# Set these in Render dashboard:
Build Command: pip install -r requirements.txt
Start Command: python run_app.py
```

**4. Heroku**
Create a `Procfile`:
```
web: python run_app.py
```

### Environment Variables

The application respects these environment variables:

- `PORT` - Server port (default: 8501)
- `HOST` - Server address (default: 0.0.0.0)

### Deployment Checklist

- [ ] Ensure Python 3.11+ is available
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Test locally: `python run_app.py`
- [ ] Set environment variables (PORT, HOST) if needed
- [ ] Configure build and start commands on your platform
- [ ] Deploy!

### Troubleshooting

**Missing Dependencies:**
```bash
pip install -r requirements.txt
```

**Port Already in Use:**
```bash
# Linux/Mac
export PORT=8502
python run_app.py

# Windows
set PORT=8502
python run_app.py
```

**Import Errors:**
Make sure you're running from the project root directory.


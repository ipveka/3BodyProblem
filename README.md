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

## 🌐 Deployment

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


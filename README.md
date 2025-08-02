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


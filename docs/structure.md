# Application Structure

This document describes the architecture and organization of the 3BodyProblem N-Body Gravitational Simulation application.

## Project Overview

The application is organized into several key modules that separate concerns: physics simulation, visualization, and user interface. This modular design makes the codebase maintainable and extensible.

## Directory Structure

```
3BodyProblem/
├── app/
│   └── app.py              # Streamlit web application interface
├── core/
│   ├── body.py            # CelestialBody class definition
│   ├── simulation.py      # NBodySimulation engine
│   └── solver.py          # Numerical integration methods
├── visualization/
│   ├── plot.py            # Static plotting functions
│   └── animate.py         # Animation functions
├── scripts/
│   ├── run_two_body.py    # Two-body demonstration script
│   └── run_three_body.py  # Three-body demonstration script
├── docs/                  # Documentation (this folder)
├── run_app.py             # Application launcher with dependency checks
├── requirements.txt       # Python package dependencies
├── pyproject.toml         # Project configuration
├── setup.py               # Package installation script
└── README.md              # Project overview and quick start
```

## Core Modules

### `core/body.py` - Celestial Body Representation

**Purpose**: Defines the `CelestialBody` class that represents individual celestial objects in the simulation.

**Key Components**:
- **CelestialBody Class**: 
  - Stores mass, position, velocity, name, color, and radius
  - Tracks position history for trail visualization
  - Provides methods for calculating kinetic energy, momentum, and gravitational forces
  - Includes methods for updating position/velocity and managing trails

- **PresetBodies Class**: 
  - Factory methods for creating common celestial bodies (Sun, Earth, Moon, Jupiter)
  - Provides realistic physical parameters for educational demonstrations

**Responsibilities**:
- Encapsulating physical properties of celestial bodies
- Calculating body-specific quantities (energy, momentum, forces)
- Managing visualization data (trails, colors)

### `core/simulation.py` - Simulation Engine

**Purpose**: Implements the main N-body gravitational simulation engine.

**Key Components**:
- **NBodySimulation Class**:
  - Manages a collection of celestial bodies
  - Computes gravitational forces between all body pairs
  - Integrates equations of motion using various numerical methods
  - Tracks energy conservation throughout the simulation
  - Calculates center of mass

**Key Methods**:
- `simulate()`: Main simulation loop that runs the numerical integration
- `_compute_accelerations()`: Calculates gravitational accelerations for all bodies
- `_derivatives()`: Defines the system of ODEs for the N-body problem
- `_calculate_energies()`: Computes kinetic, potential, and total energy
- `get_center_of_mass()`: Calculates system center of mass

**Responsibilities**:
- Orchestrating the physics simulation
- Managing the state of all bodies
- Providing integration with different numerical solvers
- Tracking simulation statistics (energy, time, steps)

### `core/solver.py` - Numerical Integration

**Purpose**: Implements various numerical methods for solving the system of ODEs.

**Key Components**:
- **ODESolver (Abstract Base Class)**: Defines the interface for all solvers
- **RungeKuttaSolver**: 4th-order Runge-Kutta method (RK4) - high accuracy
- **EulerSolver**: Forward Euler method - simple, educational
- **VerletSolver**: Velocity Verlet method - excellent energy conservation

**Responsibilities**:
- Providing numerical integration algorithms
- Handling time stepping and state evolution
- Supporting different accuracy/stability trade-offs

## Visualization Modules

### `visualization/plot.py` - Static Plotting

**Purpose**: Functions for creating static plots of simulation results.

**Key Functions**:
- `plot_trajectories()`: Creates 2D trajectory plots
- `plot_energy_conservation()`: Plots energy over time
- Additional plotting utilities for analysis

**Responsibilities**:
- Generating matplotlib/plotly visualizations
- Creating publication-quality plots
- Displaying simulation results in various formats

### `visualization/animate.py` - Animations

**Purpose**: Functions for creating animated visualizations of orbital motion.

**Key Functions**:
- `create_animated_plot()`: Generates animated trajectory plots
- Animation utilities for real-time visualization

**Responsibilities**:
- Creating dynamic visualizations
- Managing animation frames and playback
- Providing interactive animation controls

## Application Layer

### `app/app.py` - Streamlit Web Interface

**Purpose**: Interactive web application built with Streamlit.

**Key Features**:
- **Preset Systems**: Pre-configured celestial body systems (Earth-Moon, Sun-Earth-Moon, etc.)
- **Body Management**: Add, remove, and configure celestial bodies
- **Simulation Controls**: Adjust parameters (duration, time step, integration method)
- **Real-time Visualization**: Interactive Plotly plots showing orbital trajectories
- **Statistics Display**: Energy conservation, final positions, system metrics
- **Documentation**: Built-in physics and mathematics explanations

**Responsibilities**:
- Providing user interface for the simulation
- Managing user interactions and state
- Displaying results in an intuitive format
- Handling input validation and error messages

**Session State Management**:
- `st.session_state.simulation`: Current simulation instance
- `st.session_state.bodies`: List of celestial bodies
- `st.session_state.simulation_data`: Results from completed simulations

## Scripts

### `scripts/run_two_body.py`

**Purpose**: Standalone script demonstrating two-body gravitational systems.

**Features**:
- Pre-configured two-body scenarios (Earth-Moon, binary stars, etc.)
- Command-line execution
- Static visualization output
- Educational examples

### `scripts/run_three_body.py`

**Purpose**: Standalone script demonstrating three-body gravitational systems.

**Features**:
- Pre-configured three-body scenarios (Sun-Earth-Moon, Lagrange points, etc.)
- Demonstrates chaotic dynamics
- Comparison of different integration methods

## Application Launcher

### `run_app.py`

**Purpose**: Robust application launcher with pre-flight checks.

**Key Features**:
- **Python Version Check**: Ensures Python 3.11+ is available
- **Dependency Verification**: Checks that all required packages are installed
- **Project Structure Validation**: Verifies all necessary files exist
- **Environment Configuration**: Handles PORT and HOST environment variables
- **Streamlit Launch**: Configures and launches Streamlit with appropriate settings

**Responsibilities**:
- Ensuring deployment readiness
- Providing clear error messages for missing dependencies
- Handling deployment environment configuration
- Serving as the entry point for production deployments

## Data Flow

### Simulation Workflow

1. **User Input** → `app/app.py` collects user parameters and body configurations
2. **Body Creation** → `core/body.py` creates `CelestialBody` instances
3. **Simulation Setup** → `core/simulation.py` initializes `NBodySimulation` with bodies
4. **Integration** → `core/solver.py` methods solve the ODE system
5. **Force Calculation** → `core/simulation.py` computes gravitational forces
6. **State Update** → Bodies are updated with new positions and velocities
7. **Energy Tracking** → Energy conservation is monitored
8. **Visualization** → `visualization/` modules create plots and animations
9. **Display** → `app/app.py` presents results to the user

### State Management

- **Bodies**: Created in the UI, stored in session state, passed to simulation
- **Simulation Data**: Generated by simulation engine, stored in session state, used for visualization
- **Trails**: Managed by `CelestialBody` class, updated during simulation

## Dependencies

### Core Scientific Libraries
- **NumPy**: Numerical computations, array operations
- **SciPy**: Advanced numerical methods (optional, for scipy solver)

### Visualization
- **Matplotlib**: Static plotting (used in scripts)
- **Plotly**: Interactive web visualizations (used in Streamlit app)
- **Seaborn**: Statistical visualizations (optional)

### Web Framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation for tables and statistics

## Design Patterns

### Object-Oriented Design
- **Encapsulation**: Each class manages its own state and behavior
- **Separation of Concerns**: Physics, visualization, and UI are separate modules
- **Factory Pattern**: `PresetBodies` class for creating common body configurations

### Modularity
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Each module has a single, clear responsibility
- **Extensibility**: Easy to add new integration methods, body types, or visualizations

## Extension Points

The architecture supports easy extension:

1. **New Integration Methods**: Add to `core/solver.py` by implementing `ODESolver` interface
2. **New Body Types**: Extend `CelestialBody` or add to `PresetBodies`
3. **New Visualizations**: Add functions to `visualization/` modules
4. **New UI Features**: Extend `app/app.py` with additional Streamlit components
5. **New Preset Systems**: Add preset configurations in `app/app.py` or create new scripts

## Testing and Validation

- **Energy Conservation**: Built-in energy tracking validates numerical accuracy
- **Dependency Checks**: `run_app.py` validates environment before launch
- **Input Validation**: UI validates user inputs (mass > 0, proper dimensions, etc.)

## Performance Considerations

- **Trail Management**: Trails are limited to prevent memory issues
- **Subsampling**: Position history is subsampled for performance
- **Vectorization**: NumPy operations are vectorized for efficiency
- **Time Step Control**: User can adjust time step to balance accuracy and speed

This structure provides a clean, maintainable codebase that separates physics simulation, visualization, and user interface concerns while remaining extensible and educational.


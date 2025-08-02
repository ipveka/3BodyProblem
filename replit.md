# Overview

This is a comprehensive N-body gravitational simulation system built in Python that demonstrates celestial mechanics through interactive visualization. The project implements Newton's laws of gravitation to simulate the motion of multiple celestial bodies (planets, moons, stars) under mutual gravitational attraction. It features multiple numerical integration methods, real-time parameter control through a Streamlit web interface, and both static and animated visualization capabilities for analyzing orbital dynamics, energy conservation, and chaotic behavior in multi-body systems.

# User Preferences

Preferred communication style: Simple, everyday language.

## Recent User Requests (August 2025):
- Light theme as default for Streamlit interface
- Simplified visualization approach using matplotlib instead of complex animation
- Daily position calculations for yearly simulations (365.25 days per year)
- Consistent plot sizing with proper axis scaling and equal aspect ratios  
- Custom system design: Sun-Earth-Moon base + additional custom bodies
- Final position labels and annotations on orbital plots
- Comprehensive data tables showing position coordinates and movement analysis

# System Architecture

## Core Physics Engine
The system uses an object-oriented architecture with separate modules for celestial bodies, simulation logic, and numerical solvers. The `CelestialBody` class encapsulates physical properties (mass, position, velocity) and visualization attributes. The `NBodySimulation` class manages the physics calculations and integrates Newton's gravitational equations using multiple numerical methods including Runge-Kutta 4th order (RK4), Euler, and Velocity Verlet solvers for different accuracy and stability trade-offs.

## Integration Methods
Three different ODE solver implementations are provided through an abstract base class pattern. This allows users to compare numerical accuracy and computational performance between methods. RK4 provides high accuracy for smooth simulations, Euler offers simplicity for educational purposes, and Verlet maintains better energy conservation for long-term simulations.

## Web Interface Architecture
The system uses Streamlit as the web framework to provide an interactive interface for real-time parameter adjustment. The web app allows users to add/remove celestial bodies, modify simulation parameters, select integration methods, and visualize results without requiring direct code interaction. Session state management preserves user configurations across interactions.

## Visualization System
Dual visualization approach using both Matplotlib for traditional scientific plotting and Plotly for interactive web-based visualizations. The system supports static trajectory plots, animated simulations with trails, energy conservation analysis, phase space diagrams, and real-time statistical monitoring. Animation capabilities include playback controls and customizable trail lengths.

## Script-Based Demonstrations
Pre-built demonstration scripts showcase specific gravitational systems like Earth-Moon two-body problems, Sun-Earth-Moon three-body systems, and chaotic dynamics examples. These serve as both educational tools and validation of the physics implementation.

# External Dependencies

## Scientific Computing Stack
- **NumPy**: Core numerical operations and array handling for position/velocity vectors
- **SciPy**: Advanced numerical integration methods and mathematical functions
- **Matplotlib**: Traditional scientific plotting and animation capabilities
- **Plotly**: Interactive web-based visualizations and real-time plotting

## Web Framework
- **Streamlit**: Web interface framework for interactive parameter control and real-time visualization

## Data Handling
- **Pandas**: Data structure management for simulation results and statistical analysis
- **Seaborn**: Enhanced statistical visualization styling and color palettes

The project is designed as a standalone Python package with setup.py configuration, requiring Python 3.7+ and standard scientific computing libraries. No external databases or cloud services are required - all computations and visualizations run locally.
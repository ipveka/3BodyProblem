"""
N-Body Gravitational Simulation Engine

This module implements the core physics simulation for N-body gravitational systems.
It handles the numerical integration of Newton's laws of motion under gravitational
forces between multiple celestial bodies.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.integrate import solve_ivp
import warnings

from .body import CelestialBody
from .solver import RungeKuttaSolver, EulerSolver, VerletSolver

class NBodySimulation:
    """
    Main simulation engine for N-body gravitational systems.
    
    This class manages the simulation of multiple celestial bodies under
    mutual gravitational attraction using various numerical integration methods.
    """
    
    def __init__(self, bodies: List[CelestialBody]):
        """
        Initialize the N-body simulation.
        
        Args:
            bodies (List[CelestialBody]): List of celestial bodies to simulate
        """
        if len(bodies) < 2:
            raise ValueError("At least 2 bodies are required for simulation")
        
        self.bodies = [body.copy() for body in bodies]  # Deep copy to avoid modifying originals
        self.n_bodies = len(self.bodies)
        
        # Gravitational constant in m³ kg⁻¹ s⁻²
        self.G = 6.67430e-11
        
        # Initialize custom solvers
        self.rk4_solver = RungeKuttaSolver()
        self.euler_solver = EulerSolver()
        self.verlet_solver = VerletSolver()
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        
    def _state_vector(self) -> np.ndarray:
        """
        Convert the current state of all bodies to a single state vector.
        
        Returns:
            np.ndarray: State vector of shape (4*n_bodies,) containing
                       [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        state = np.zeros(4 * self.n_bodies)
        
        for i, body in enumerate(self.bodies):
            start_idx = i * 4
            state[start_idx:start_idx+2] = body.position
            state[start_idx+2:start_idx+4] = body.velocity
            
        return state
    
    def _update_bodies_from_state(self, state: np.ndarray, update_trails: bool = True):
        """
        Update all bodies from a state vector.
        
        Args:
            state (np.ndarray): State vector containing positions and velocities
            update_trails (bool): Whether to update position trails
        """
        for i, body in enumerate(self.bodies):
            start_idx = i * 4
            new_position = state[start_idx:start_idx+2]
            new_velocity = state[start_idx+2:start_idx+4]
            
            body.update_position(new_position, update_trails)
            body.update_velocity(new_velocity)
    
    def _compute_accelerations(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gravitational accelerations for all bodies.
        
        Args:
            state (np.ndarray): Current state vector
            
        Returns:
            np.ndarray: Acceleration vector for all bodies
        """
        accelerations = np.zeros(2 * self.n_bodies)
        
        # Extract positions from state vector
        positions = np.zeros((self.n_bodies, 2))
        for i in range(self.n_bodies):
            start_idx = i * 4
            positions[i] = state[start_idx:start_idx+2]
        
        # Calculate gravitational forces between all pairs
        for i in range(self.n_bodies):
            total_force = np.array([0.0, 0.0])
            
            for j in range(self.n_bodies):
                if i != j:
                    # Calculate displacement vector (from body i to body j)
                    displacement = positions[j] - positions[i]  # km
                    distance = np.sqrt(np.sum(displacement ** 2))  # km
                    
                    # Avoid singularity with softening parameter
                    softening = 1e-3  # km
                    distance = max(distance, softening)
                    
                    # Convert to meters for force calculation
                    distance_m = distance * 1000
                    displacement_m = displacement * 1000
                    
                    # Calculate force magnitude
                    force_magnitude = (self.G * self.bodies[i].mass * self.bodies[j].mass 
                                     / (distance_m ** 2))
                    
                    # Force direction (unit vector)
                    unit_vector = displacement / distance
                    
                    # Add to total force on body i
                    total_force += force_magnitude * unit_vector
            
            # Convert force to acceleration (F = ma -> a = F/m)
            acceleration = total_force / self.bodies[i].mass
            accelerations[i*2:(i+1)*2] = acceleration / 1000  # Convert back to km/s²
        
        return accelerations
    
    def _derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Calculate derivatives for the ODE system.
        
        This function defines the system of ODEs that govern the N-body problem:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = ax
        dvy/dt = ay
        
        Args:
            t (float): Current time
            state (np.ndarray): Current state vector
            
        Returns:
            np.ndarray: Derivative vector
        """
        derivatives = np.zeros_like(state)
        
        # Calculate accelerations
        accelerations = self._compute_accelerations(state)
        
        # Fill derivatives vector
        for i in range(self.n_bodies):
            start_idx = i * 4
            
            # Position derivatives are velocities
            derivatives[start_idx:start_idx+2] = state[start_idx+2:start_idx+4]
            
            # Velocity derivatives are accelerations
            derivatives[start_idx+2:start_idx+4] = accelerations[i*2:(i+1)*2]
        
        return derivatives
    
    def simulate(self, 
                 t_span: Tuple[float, float], 
                 t_eval: Optional[np.ndarray] = None,
                 method: str = 'rk4',
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Run the N-body simulation.
        
        Args:
            t_span (Tuple[float, float]): Time span (start_time, end_time)
            t_eval (np.ndarray, optional): Specific times to evaluate solution
            method (str): Integration method ('rk4', 'euler', 'verlet', 'scipy')
            **kwargs: Additional arguments for scipy solvers
            
        Returns:
            Tuple containing:
                - positions (np.ndarray): Position history of shape (n_times, n_bodies, 2)
                - velocities (np.ndarray): Velocity history of shape (n_times, n_bodies, 2)
                - times (np.ndarray): Time points
                - energies (Dict[str, np.ndarray]): Energy conservation data
        """
        t_start, t_end = t_span
        
        if t_eval is None:
            n_points = 1000
            t_eval = np.linspace(t_start, t_end, n_points)
        
        # Get initial state
        initial_state = self._state_vector()
        
        if method.lower() == 'scipy':
            # Use scipy's solve_ivp
            sol = solve_ivp(
                self._derivatives, 
                t_span, 
                initial_state, 
                t_eval=t_eval,
                method=kwargs.get('scipy_method', 'RK45'),
                rtol=kwargs.get('rtol', 1e-8),
                atol=kwargs.get('atol', 1e-10)
            )
            
            if not sol.success:
                warnings.warn(f"Integration failed: {sol.message}")
            
            solution = sol.y.T  # Transpose to get shape (n_times, n_variables)
            times = sol.t
            
        else:
            # Use custom solvers
            dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.01
            
            if method.lower() == 'rk4':
                solution, times = self.rk4_solver.solve(
                    self._derivatives, initial_state, t_eval, dt
                )
            elif method.lower() == 'euler':
                solution, times = self.euler_solver.solve(
                    self._derivatives, initial_state, t_eval, dt
                )
            elif method.lower() == 'verlet':
                solution, times = self.verlet_solver.solve(
                    self._derivatives, initial_state, t_eval, dt
                )
            else:
                raise ValueError(f"Unknown integration method: {method}")
        
        # Extract positions and velocities
        n_times = len(times)
        positions = np.zeros((n_times, self.n_bodies, 2))
        velocities = np.zeros((n_times, self.n_bodies, 2))
        
        for t_idx in range(n_times):
            for body_idx in range(self.n_bodies):
                start_idx = body_idx * 4
                positions[t_idx, body_idx] = solution[t_idx, start_idx:start_idx+2]
                velocities[t_idx, body_idx] = solution[t_idx, start_idx+2:start_idx+4]
        
        # Calculate energy conservation
        energies = self._calculate_energies(positions, velocities)
        
        # Update body trails
        for t_idx in range(n_times):
            for body_idx, body in enumerate(self.bodies):
                if t_idx % max(1, n_times // 100) == 0:  # Subsample for performance
                    body.trail_positions.append(positions[t_idx, body_idx].copy())
        
        return positions, velocities, times, energies
    
    def _calculate_energies(self, positions: np.ndarray, velocities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate kinetic, potential, and total energy throughout the simulation.
        
        Args:
            positions (np.ndarray): Position history
            velocities (np.ndarray): Velocity history
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing energy arrays
        """
        n_times, n_bodies, _ = positions.shape
        
        kinetic_energies = np.zeros(n_times)
        potential_energies = np.zeros(n_times)
        
        for t_idx in range(n_times):
            # Kinetic energy
            ke = 0.0
            for body_idx in range(n_bodies):
                mass = self.bodies[body_idx].mass
                vel = velocities[t_idx, body_idx] * 1000  # Convert km/s to m/s
                speed_squared = np.sum(vel ** 2)
                ke += 0.5 * mass * speed_squared
            
            kinetic_energies[t_idx] = ke
            
            # Potential energy
            pe = 0.0
            for i in range(n_bodies):
                for j in range(i + 1, n_bodies):
                    mass_i = self.bodies[i].mass
                    mass_j = self.bodies[j].mass
                    
                    pos_i = positions[t_idx, i] * 1000  # Convert km to m
                    pos_j = positions[t_idx, j] * 1000
                    
                    distance = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    distance = max(distance, 1e-3)  # Avoid singularity
                    
                    pe -= self.G * mass_i * mass_j / distance
            
            potential_energies[t_idx] = pe
        
        total_energies = kinetic_energies + potential_energies
        
        return {
            'kinetic': kinetic_energies,
            'potential': potential_energies,
            'total': total_energies
        }
    
    def get_center_of_mass(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the center of mass position and velocity.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Center of mass position and velocity
        """
        total_mass = sum(body.mass for body in self.bodies)
        
        com_position = np.zeros(2)
        com_velocity = np.zeros(2)
        
        for body in self.bodies:
            com_position += body.mass * body.position
            com_velocity += body.mass * body.velocity
        
        com_position /= total_mass
        com_velocity /= total_mass
        
        return com_position, com_velocity
    
    def reset_to_initial_conditions(self):
        """Reset all bodies to their initial conditions."""
        for body in self.bodies:
            body.clear_trail()
        
        self.current_time = 0.0
        self.step_count = 0
    
    def add_body(self, body: CelestialBody):
        """
        Add a new body to the simulation.
        
        Args:
            body (CelestialBody): Body to add
        """
        self.bodies.append(body.copy())
        self.n_bodies = len(self.bodies)
    
    def remove_body(self, index: int):
        """
        Remove a body from the simulation.
        
        Args:
            index (int): Index of body to remove
        """
        if 0 <= index < self.n_bodies:
            self.bodies.pop(index)
            self.n_bodies = len(self.bodies)
        else:
            raise IndexError("Body index out of range")

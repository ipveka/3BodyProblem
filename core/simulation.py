"""
N-Body Gravitational Simulation Engine

This module implements the core physics simulation for N-body gravitational systems.
It handles the numerical integration of Newton's laws of motion under gravitational
forces between multiple celestial bodies.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

from .body import CelestialBody
from .solver import RungeKuttaSolver, EulerSolver, VerletSolver

# NOTE: SciPy is imported lazily inside the 'scipy' integration method so that
# the rest of the engine (RK4/Euler/Verlet) runs without it. This keeps lean
# deployments (e.g. serverless functions) small — only numpy is required.

class NBodySimulation:
    """
    Main simulation engine for N-body gravitational systems.
    
    This class manages the simulation of multiple celestial bodies under
    mutual gravitational attraction using various numerical integration methods.
    """
    
    def __init__(self,
                 bodies: List[CelestialBody],
                 G: float = 6.67430e-11,
                 length_unit: float = 1000.0):
        """
        Initialize the N-body simulation.

        Args:
            bodies (List[CelestialBody]): List of celestial bodies to simulate
            G (float): Gravitational constant in SI units (m³ kg⁻¹ s⁻²).
                For normalized systems (e.g. the figure-eight orbit) pass
                ``G=1.0`` together with ``length_unit=1.0``.
            length_unit (float): Number of meters represented by one position
                unit. Defaults to 1000.0 (positions in km). Use ``1.0`` for
                systems expressed in normalized/SI units.
        """
        if len(bodies) < 2:
            raise ValueError("At least 2 bodies are required for simulation")

        # Spatial dimension is inferred from the bodies (2D or 3D supported).
        self.dim = len(bodies[0].position)
        if self.dim not in (2, 3):
            raise ValueError("Only 2D and 3D simulations are supported")
        for body in bodies:
            if len(body.position) != self.dim or len(body.velocity) != self.dim:
                raise ValueError("All bodies must share the same dimension")
            if not (np.all(np.isfinite(body.position))
                    and np.all(np.isfinite(body.velocity))):
                raise ValueError("Body position/velocity must be finite")

        self.bodies = [body.copy() for body in bodies]
        self.n_bodies = len(self.bodies)
        # Number of state values per body: positions + velocities.
        self.stride = 2 * self.dim
        # Mass vector, cached for the vectorized acceleration computation.
        self.masses = np.array([b.mass for b in self.bodies], dtype=float)

        # Gravitational constant in SI units (m³ kg⁻¹ s⁻²)
        self.G = G
        # Meters per position unit (km -> 1000); keeps energies in SI Joules.
        self.length_unit = length_unit
        # Effective constant so acceleration comes out directly in
        # (position_unit / s²):  a_i = G_eff * Σ_{j≠i} m_j (r_j - r_i) / |r_j - r_i|³
        self.G_eff = G / (length_unit ** 3)

        # Initialize custom solvers
        self.rk4_solver = RungeKuttaSolver()
        self.euler_solver = EulerSolver()
        self.verlet_solver = VerletSolver(dim=self.dim)

        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        
    def _state_vector(self) -> np.ndarray:
        """
        Convert the current state of all bodies to a single state vector.
        
        Returns:
            np.ndarray: State vector of shape (2*dim*n_bodies,) laid out per
                       body as [pos(dim), vel(dim), pos(dim), vel(dim), ...]
        """
        state = np.zeros(self.stride * self.n_bodies)
        d = self.dim

        for i, body in enumerate(self.bodies):
            start_idx = i * self.stride
            state[start_idx:start_idx+d] = body.position
            state[start_idx+d:start_idx+2*d] = body.velocity

        return state

    def _update_bodies_from_state(self, state: np.ndarray, update_trails: bool = True):
        """
        Update all bodies from a state vector.

        Args:
            state (np.ndarray): State vector containing positions and velocities
            update_trails (bool): Whether to update position trails
        """
        d = self.dim
        for i, body in enumerate(self.bodies):
            start_idx = i * self.stride
            new_position = state[start_idx:start_idx+d]
            new_velocity = state[start_idx+d:start_idx+2*d]

            body.update_position(new_position, update_trails)
            body.update_velocity(new_velocity)
    
    def _compute_accelerations(self, state: np.ndarray) -> np.ndarray:
        """
        Compute gravitational accelerations for all bodies.
        
        Args:
            state (np.ndarray): Current state vector
            
        Returns:
            np.ndarray: Flat acceleration vector of shape (dim*n_bodies,),
                       laid out as [a1(dim), a2(dim), ...]
        """
        d = self.dim
        # Positions as an (N, dim) array: every body's first ``dim`` entries.
        positions = state.reshape(self.n_bodies, self.stride)[:, :d]

        # Softening length (in position units) to avoid the 1/r² singularity.
        softening = 1e-3

        # Vectorized pairwise gravity. displacement[i, j] = r_j - r_i.
        #   a_i = G_eff * Σ_{j≠i} m_j (r_j - r_i) / |r_j - r_i|³
        # The mass of body i cancels out of F = m_i * a_i, so it never appears.
        displacement = positions[None, :, :] - positions[:, None, :]  # (N, N, dim)
        dist_sq = np.sum(displacement ** 2, axis=2)                    # (N, N)
        np.fill_diagonal(dist_sq, np.inf)        # self-interaction -> 0 below
        distance = np.maximum(np.sqrt(dist_sq), softening)
        inv_cube = 1.0 / distance ** 3                                 # (N, N)

        # acc[i] = G_eff * Σ_j m_j * displacement[i, j] * inv_cube[i, j]
        acc = self.G_eff * np.einsum(
            'j,ijk,ij->ik', self.masses, displacement, inv_cube)

        return acc.reshape(-1)
    
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
        d = self.dim
        accelerations = self._compute_accelerations(state)

        # Vectorized assembly: dpos/dt = vel, dvel/dt = acc, per body.
        derivatives = np.empty_like(state)
        s = state.reshape(self.n_bodies, self.stride)
        out = derivatives.reshape(self.n_bodies, self.stride)
        out[:, :d] = s[:, d:2 * d]
        out[:, d:2 * d] = accelerations.reshape(self.n_bodies, d)

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
                - positions (np.ndarray): Position history, shape (n_times, n_bodies, dim)
                - velocities (np.ndarray): Velocity history, shape (n_times, n_bodies, dim)
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
            # Imported lazily so non-scipy methods work without SciPy installed.
            from scipy.integrate import solve_ivp

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
        d = self.dim
        # solution is (n_times, stride*n_bodies); reshape to per-body blocks.
        reshaped = solution.reshape(n_times, self.n_bodies, self.stride)
        positions = reshaped[:, :, :d].copy()
        velocities = reshaped[:, :, d:2*d].copy()
        
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
        lu = self.length_unit
        masses = self.masses

        # Kinetic energy: KE(t) = ½ Σ_i m_i |v_i|²  (fully vectorized, small memory).
        vm = velocities * lu  # -> m/s
        kinetic_energies = 0.5 * np.einsum('i,tik,tik->t', masses, vm, vm)

        # Potential energy: PE(t) = -G Σ_{i<j} m_i m_j / |r_i - r_j|
        potential_energies = np.zeros(n_times)
        iu = np.triu_indices(n_bodies, k=1)
        if iu[0].size > 0:
            mass_products = (masses[:, None] * masses[None, :])[iu]  # (n_pairs,)
            pm_all = positions * lu  # -> m
            # Process time in chunks so peak memory stays bounded regardless of
            # how long the run is (the pairwise array is O(chunk · n_bodies²)).
            chunk = max(1, 4_000_000 // max(1, n_bodies * n_bodies))
            for start in range(0, n_times, chunk):
                end = min(start + chunk, n_times)
                pm = pm_all[start:end]  # (c, n_bodies, dim)
                diff = pm[:, :, None, :] - pm[:, None, :, :]  # (c, N, N, dim)
                dist = np.sqrt(np.sum(diff ** 2, axis=3))  # (c, N, N)
                pair_dist = np.maximum(dist[:, iu[0], iu[1]], 1e-3)  # (c, n_pairs)
                potential_energies[start:end] = (
                    -self.G * (mass_products / pair_dist).sum(axis=1))

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

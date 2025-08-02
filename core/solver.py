"""
Numerical ODE Solvers for N-Body Simulation

This module implements various numerical integration methods for solving
the system of ordinary differential equations that govern N-body dynamics.
"""

import numpy as np
from typing import Callable, Tuple
from abc import ABC, abstractmethod

class ODESolver(ABC):
    """Abstract base class for ODE solvers."""
    
    @abstractmethod
    def solve(self, 
              func: Callable[[float, np.ndarray], np.ndarray], 
              y0: np.ndarray, 
              t_eval: np.ndarray, 
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the ODE system.
        
        Args:
            func: Function that computes derivatives dy/dt = f(t, y)
            y0: Initial conditions
            t_eval: Time points to evaluate
            dt: Time step size
            
        Returns:
            Tuple of (solution_array, time_array)
        """
        pass

class RungeKuttaSolver(ODESolver):
    """
    Fourth-order Runge-Kutta (RK4) solver.
    
    This is a widely used method that provides good accuracy and stability
    for many ODE systems, including gravitational N-body problems.
    """
    
    def solve(self, 
              func: Callable[[float, np.ndarray], np.ndarray], 
              y0: np.ndarray, 
              t_eval: np.ndarray, 
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using fourth-order Runge-Kutta method.
        
        The RK4 method approximates the solution using:
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        Args:
            func: Derivative function f(t, y)
            y0: Initial state vector
            t_eval: Time evaluation points
            dt: Time step size
            
        Returns:
            Tuple of (solution_array, time_array)
        """
        # Initialize solution array
        n_points = len(t_eval)
        solution = np.zeros((n_points, len(y0)))
        times = np.zeros(n_points)
        
        # Set initial conditions
        solution[0] = y0.copy()
        times[0] = t_eval[0]
        
        # Current state
        y_current = y0.copy()
        t_current = t_eval[0]
        
        # Integrate over time
        eval_idx = 1
        
        while eval_idx < n_points and t_current < t_eval[-1]:
            # Adaptive step size to hit evaluation points
            t_target = t_eval[eval_idx]
            h = min(dt, t_target - t_current)
            
            # RK4 integration step
            k1 = h * func(t_current, y_current)
            k2 = h * func(t_current + h/2, y_current + k1/2)
            k3 = h * func(t_current + h/2, y_current + k2/2)
            k4 = h * func(t_current + h, y_current + k3)
            
            y_next = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6
            t_next = t_current + h
            
            # Check if we've reached an evaluation point
            if abs(t_next - t_target) < 1e-10 or t_next >= t_target:
                solution[eval_idx] = y_next.copy()
                times[eval_idx] = t_target
                eval_idx += 1
            
            # Update current state
            y_current = y_next
            t_current = t_next
        
        return solution[:eval_idx], times[:eval_idx]

class EulerSolver(ODESolver):
    """
    Forward Euler solver.
    
    This is the simplest numerical integration method. While less accurate
    than higher-order methods, it's useful for comparison and understanding
    the impact of integration method choice.
    """
    
    def solve(self, 
              func: Callable[[float, np.ndarray], np.ndarray], 
              y0: np.ndarray, 
              t_eval: np.ndarray, 
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using forward Euler method.
        
        The Euler method approximates the solution using:
        y_next = y + h * f(t, y)
        
        Args:
            func: Derivative function f(t, y)
            y0: Initial state vector
            t_eval: Time evaluation points
            dt: Time step size
            
        Returns:
            Tuple of (solution_array, time_array)
        """
        # Initialize solution array
        n_points = len(t_eval)
        solution = np.zeros((n_points, len(y0)))
        times = np.zeros(n_points)
        
        # Set initial conditions
        solution[0] = y0.copy()
        times[0] = t_eval[0]
        
        # Current state
        y_current = y0.copy()
        t_current = t_eval[0]
        
        # Integrate over time
        eval_idx = 1
        
        while eval_idx < n_points and t_current < t_eval[-1]:
            # Adaptive step size to hit evaluation points
            t_target = t_eval[eval_idx]
            h = min(dt, t_target - t_current)
            
            # Euler integration step
            dy_dt = func(t_current, y_current)
            y_next = y_current + h * dy_dt
            t_next = t_current + h
            
            # Check if we've reached an evaluation point
            if abs(t_next - t_target) < 1e-10 or t_next >= t_target:
                solution[eval_idx] = y_next.copy()
                times[eval_idx] = t_target
                eval_idx += 1
            
            # Update current state
            y_current = y_next
            t_current = t_next
        
        return solution[:eval_idx], times[:eval_idx]

class VerletSolver(ODESolver):
    """
    Velocity Verlet solver.
    
    This is a symplectic integrator that conserves energy better than
    non-symplectic methods like RK4 for Hamiltonian systems like
    gravitational N-body problems.
    """
    
    def __init__(self):
        self.previous_accelerations = None
    
    def solve(self, 
              func: Callable[[float, np.ndarray], np.ndarray], 
              y0: np.ndarray, 
              t_eval: np.ndarray, 
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using Velocity Verlet method.
        
        The Velocity Verlet method is specifically designed for second-order
        ODEs of the form d²x/dt² = f(x, t). For N-body problems, this
        provides better energy conservation.
        
        Args:
            func: Derivative function f(t, y) where y = [x, v]
            y0: Initial state vector [positions, velocities]
            t_eval: Time evaluation points
            dt: Time step size
            
        Returns:
            Tuple of (solution_array, time_array)
        """
        # Initialize solution array
        n_points = len(t_eval)
        n_vars = len(y0)
        solution = np.zeros((n_points, n_vars))
        times = np.zeros(n_points)
        
        # Set initial conditions
        solution[0] = y0.copy()
        times[0] = t_eval[0]
        
        # Extract initial positions and velocities
        # Assume state vector is [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        n_bodies = n_vars // 4
        
        positions = np.zeros((n_bodies, 2))
        velocities = np.zeros((n_bodies, 2))
        
        for i in range(n_bodies):
            start_idx = i * 4
            positions[i] = y0[start_idx:start_idx+2]
            velocities[i] = y0[start_idx+2:start_idx+4]
        
        # Calculate initial accelerations
        derivatives = func(t_eval[0], y0)
        accelerations = np.zeros((n_bodies, 2))
        for i in range(n_bodies):
            start_idx = i * 4
            accelerations[i] = derivatives[start_idx+2:start_idx+4]
        
        # Current state
        t_current = t_eval[0]
        eval_idx = 1
        
        while eval_idx < n_points and t_current < t_eval[-1]:
            # Adaptive step size to hit evaluation points
            t_target = t_eval[eval_idx]
            h = min(dt, t_target - t_current)
            
            # Velocity Verlet integration
            # Update positions: x(t+h) = x(t) + v(t)*h + 0.5*a(t)*h²
            new_positions = positions + velocities * h + 0.5 * accelerations * h**2
            
            # Construct new state vector for acceleration calculation
            new_state = np.zeros(n_vars)
            for i in range(n_bodies):
                start_idx = i * 4
                new_state[start_idx:start_idx+2] = new_positions[i]
                new_state[start_idx+2:start_idx+4] = velocities[i]  # Use old velocities temporarily
            
            # Calculate new accelerations
            new_derivatives = func(t_current + h, new_state)
            new_accelerations = np.zeros((n_bodies, 2))
            for i in range(n_bodies):
                start_idx = i * 4
                new_accelerations[i] = new_derivatives[start_idx+2:start_idx+4]
            
            # Update velocities: v(t+h) = v(t) + 0.5*(a(t) + a(t+h))*h
            new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * h
            
            # Update time
            t_next = t_current + h
            
            # Check if we've reached an evaluation point
            if abs(t_next - t_target) < 1e-10 or t_next >= t_target:
                # Store solution
                for i in range(n_bodies):
                    start_idx = i * 4
                    solution[eval_idx, start_idx:start_idx+2] = new_positions[i]
                    solution[eval_idx, start_idx+2:start_idx+4] = new_velocities[i]
                
                times[eval_idx] = t_target
                eval_idx += 1
            
            # Update current state
            positions = new_positions.copy()
            velocities = new_velocities.copy()
            accelerations = new_accelerations.copy()
            t_current = t_next
        
        return solution[:eval_idx], times[:eval_idx]

class AdaptiveRungeKuttaSolver(ODESolver):
    """
    Adaptive Runge-Kutta solver with error control.
    
    This solver automatically adjusts the step size based on local error
    estimates to maintain accuracy while optimizing performance.
    """
    
    def __init__(self, rtol: float = 1e-6, atol: float = 1e-8):
        """
        Initialize adaptive RK solver.
        
        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.rtol = rtol
        self.atol = atol
    
    def solve(self, 
              func: Callable[[float, np.ndarray], np.ndarray], 
              y0: np.ndarray, 
              t_eval: np.ndarray, 
              dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve ODE using adaptive Runge-Kutta method.
        
        Uses embedded RK4/RK5 pair for error estimation and step size control.
        
        Args:
            func: Derivative function f(t, y)
            y0: Initial state vector
            t_eval: Time evaluation points
            dt: Initial time step size
            
        Returns:
            Tuple of (solution_array, time_array)
        """
        # Initialize solution array
        n_points = len(t_eval)
        solution = np.zeros((n_points, len(y0)))
        times = np.zeros(n_points)
        
        # Set initial conditions
        solution[0] = y0.copy()
        times[0] = t_eval[0]
        
        # Current state
        y_current = y0.copy()
        t_current = t_eval[0]
        h = dt  # Current step size
        
        eval_idx = 1
        
        while eval_idx < n_points and t_current < t_eval[-1]:
            t_target = t_eval[eval_idx]
            
            # Ensure we don't overshoot the target
            h = min(h, t_target - t_current)
            
            # RK4 step
            k1 = h * func(t_current, y_current)
            k2 = h * func(t_current + h/2, y_current + k1/2)
            k3 = h * func(t_current + h/2, y_current + k2/2)
            k4 = h * func(t_current + h, y_current + k3)
            
            y_rk4 = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # RK5 step for error estimation (simplified)
            k5 = h * func(t_current + h, y_rk4)
            y_rk5 = y_current + (7*k1 + 32*k2 + 12*k3 + 32*k4 + 7*k5) / 90
            
            # Error estimate
            error = np.abs(y_rk5 - y_rk4)
            tolerance = self.atol + self.rtol * np.abs(y_current)
            
            # Check if error is acceptable
            error_ratio = np.max(error / tolerance)
            
            if error_ratio <= 1.0:
                # Accept step
                y_current = y_rk4
                t_current += h
                
                # Check if we've reached an evaluation point
                if abs(t_current - t_target) < 1e-10 or t_current >= t_target:
                    solution[eval_idx] = y_current.copy()
                    times[eval_idx] = t_target
                    eval_idx += 1
                
                # Adjust step size for next iteration
                if error_ratio > 0:
                    h *= min(2.0, 0.9 * (1.0 / error_ratio) ** 0.2)
                else:
                    h *= 2.0
            else:
                # Reject step and reduce step size
                h *= max(0.1, 0.9 * (1.0 / error_ratio) ** 0.25)
                
                # Ensure minimum step size
                if h < 1e-12:
                    raise RuntimeError("Step size became too small - integration failed")
        
        return solution[:eval_idx], times[:eval_idx]

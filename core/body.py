"""
Celestial Body Class for N-Body Gravitational Simulation

This module defines the CelestialBody class which represents a celestial object
with mass, position, velocity, and other physical properties used in gravitational
simulations.
"""

import numpy as np
from typing import List, Union, Optional

class CelestialBody:
    """
    Represents a celestial body in a gravitational N-body simulation.
    
    Attributes:
        mass (float): Mass of the body in kilograms
        position (np.ndarray): Position vector [x, y] in kilometers
        velocity (np.ndarray): Velocity vector [vx, vy] in km/s
        name (str): Name identifier for the body
        color (str): Color for visualization purposes
        radius (float): Physical radius in kilometers (for visualization)
        trail_positions (List): Historical positions for trail visualization
    """
    
    def __init__(self, 
                 mass: float, 
                 position: Union[List[float], np.ndarray], 
                 velocity: Union[List[float], np.ndarray],
                 name: str = "Unnamed Body",
                 color: str = "#FF0000",
                 radius: Optional[float] = None):
        """
        Initialize a celestial body.
        
        Args:
            mass (float): Mass in kilograms
            position (List[float] or np.ndarray): Initial position [x, y] in km
            velocity (List[float] or np.ndarray): Initial velocity [vx, vy] in km/s
            name (str): Name of the body
            color (str): Color for visualization (hex or named color)
            radius (float, optional): Physical radius in km. If None, calculated from mass.
        """
        self.mass = float(mass)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = str(name)
        self.color = color
        
        # Calculate radius if not provided (rough approximation based on mass)
        if radius is None:
            # Approximate radius assuming Earth-like density (5515 kg/m³)
            earth_density = 5515  # kg/m³
            volume = self.mass / earth_density  # m³
            self.radius = (3 * volume / (4 * np.pi)) ** (1/3) / 1000  # convert to km
        else:
            self.radius = radius
        
        # Initialize trail for visualization
        self.trail_positions = [self.position.copy()]
        
        # Validate dimensions
        if len(self.position) != 2 or len(self.velocity) != 2:
            raise ValueError("Position and velocity must be 2D vectors")
        
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
    
    def update_position(self, new_position: np.ndarray, update_trail: bool = True):
        """
        Update the body's position and optionally add to trail.
        
        Args:
            new_position (np.ndarray): New position vector
            update_trail (bool): Whether to add position to trail history
        """
        self.position = np.array(new_position, dtype=float)
        
        if update_trail:
            self.trail_positions.append(self.position.copy())
            # Limit trail length to prevent memory issues
            if len(self.trail_positions) > 10000:
                self.trail_positions = self.trail_positions[-5000:]
    
    def update_velocity(self, new_velocity: np.ndarray):
        """
        Update the body's velocity.
        
        Args:
            new_velocity (np.ndarray): New velocity vector
        """
        self.velocity = np.array(new_velocity, dtype=float)
    
    def kinetic_energy(self) -> float:
        """
        Calculate the kinetic energy of the body.
        
        Returns:
            float: Kinetic energy in Joules
        """
        speed_squared = np.sum(self.velocity ** 2) * 1000**2  # Convert km/s to m/s
        return 0.5 * self.mass * speed_squared
    
    def momentum(self) -> np.ndarray:
        """
        Calculate the momentum vector of the body.
        
        Returns:
            np.ndarray: Momentum vector in kg⋅m/s
        """
        return self.mass * self.velocity * 1000  # Convert km/s to m/s
    
    def distance_to(self, other: 'CelestialBody') -> float:
        """
        Calculate the distance to another celestial body.
        
        Args:
            other (CelestialBody): Another celestial body
            
        Returns:
            float: Distance in kilometers
        """
        displacement = self.position - other.position
        return np.sqrt(np.sum(displacement ** 2))
    
    def gravitational_force_from(self, other: 'CelestialBody') -> np.ndarray:
        """
        Calculate the gravitational force exerted by another body on this body.
        
        Args:
            other (CelestialBody): The body exerting the gravitational force
            
        Returns:
            np.ndarray: Force vector in Newtons
        """
        # Gravitational constant in m³ kg⁻¹ s⁻²
        G = 6.67430e-11
        
        # Calculate displacement vector (from this body to other body)
        displacement = other.position - self.position  # km
        distance = np.sqrt(np.sum(displacement ** 2))  # km
        
        # Avoid singularity
        if distance < 1e-10:
            return np.array([0.0, 0.0])
        
        # Convert distance to meters
        distance_m = distance * 1000
        
        # Calculate force magnitude
        force_magnitude = G * self.mass * other.mass / (distance_m ** 2)
        
        # Calculate unit vector pointing toward the other body
        unit_vector = displacement / distance
        
        # Force vector
        force_vector = force_magnitude * unit_vector
        
        return force_vector
    
    def clear_trail(self):
        """Clear the position trail history."""
        self.trail_positions = [self.position.copy()]
    
    def get_trail_array(self) -> np.ndarray:
        """
        Get the trail positions as a numpy array.
        
        Returns:
            np.ndarray: Array of shape (n_points, 2) containing trail positions
        """
        return np.array(self.trail_positions)
    
    def copy(self) -> 'CelestialBody':
        """
        Create a deep copy of the celestial body.
        
        Returns:
            CelestialBody: A new instance with the same properties
        """
        new_body = CelestialBody(
            mass=self.mass,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            name=self.name,
            color=self.color,
            radius=self.radius
        )
        new_body.trail_positions = [pos.copy() for pos in self.trail_positions]
        return new_body
    
    def __str__(self) -> str:
        """String representation of the celestial body."""
        return (f"CelestialBody(name='{self.name}', mass={self.mass:.2e} kg, "
                f"pos=[{self.position[0]:.3f}, {self.position[1]:.3f}] km, "
                f"vel=[{self.velocity[0]:.3f}, {self.velocity[1]:.3f}] km/s)")
    
    def __repr__(self) -> str:
        """Detailed representation of the celestial body."""
        return self.__str__()

# Predefined celestial bodies for common use cases
class PresetBodies:
    """Factory class for creating common celestial bodies with realistic parameters."""
    
    @staticmethod
    def sun() -> CelestialBody:
        """Create a Sun object with realistic parameters."""
        return CelestialBody(
            mass=1.989e30,  # kg
            position=[0, 0],
            velocity=[0, 0],
            name="Sun",
            color="#FDB462",
            radius=695700  # km
        )
    
    @staticmethod
    def earth() -> CelestialBody:
        """Create an Earth object with realistic parameters."""
        return CelestialBody(
            mass=5.972e24,  # kg
            position=[149597870.7, 0],  # 1 AU in km
            velocity=[0, 29.78],  # km/s
            name="Earth",
            color="#80B1D3",
            radius=6371  # km
        )
    
    @staticmethod
    def moon() -> CelestialBody:
        """Create a Moon object with realistic parameters (relative to Earth)."""
        return CelestialBody(
            mass=7.342e22,  # kg
            position=[149597870.7 + 384400, 0],  # Earth + Moon distance
            velocity=[0, 29.78 + 1.022],  # Earth velocity + Moon orbital velocity
            name="Moon",
            color="#BEBADA",
            radius=1737  # km
        )
    
    @staticmethod
    def jupiter() -> CelestialBody:
        """Create a Jupiter object with realistic parameters."""
        return CelestialBody(
            mass=1.898e27,  # kg
            position=[778299000, 0],  # 5.2 AU in km
            velocity=[0, 13.07],  # km/s
            name="Jupiter",
            color="#FB8072",
            radius=69911  # km
        )

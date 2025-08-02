#!/usr/bin/env python3
"""
Three-Body Problem Demonstration Script

This script demonstrates various three-body gravitational systems including
the famous restricted three-body problem and chaotic dynamics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.body import CelestialBody, PresetBodies
from core.simulation import NBodySimulation
from visualization.plot import plot_trajectories, plot_energy_conservation, plot_body_separations
from visualization.animate import create_matplotlib_animation

def create_sun_earth_moon_system():
    """Create the Sun-Earth-Moon three-body system."""
    print("Creating Sun-Earth-Moon System...")
    
    # Sun at origin
    sun = CelestialBody(
        mass=1.989e30,  # kg
        position=[0, 0],
        velocity=[0, 0],
        name="Sun",
        color="yellow"
    )
    
    # Earth in orbit around Sun
    earth_distance = 1.496e8  # 1 AU in km
    earth_velocity = 29.78    # km/s
    
    earth = CelestialBody(
        mass=5.972e24,  # kg
        position=[earth_distance, 0],
        velocity=[0, earth_velocity],
        name="Earth",
        color="blue"
    )
    
    # Moon in orbit around Earth (relative to Earth's position)
    moon_distance = 384400  # km from Earth
    moon_velocity = 1.022   # km/s relative to Earth
    
    moon = CelestialBody(
        mass=7.342e22,  # kg
        position=[earth_distance + moon_distance, 0],
        velocity=[0, earth_velocity + moon_velocity],
        name="Moon",
        color="gray"
    )
    
    return [sun, earth, moon]

def create_figure_eight_system():
    """
    Create the famous figure-eight three-body solution discovered by Chenciner and Montgomery.
    This is an approximate version of the exact solution.
    """
    print("Creating Figure-Eight System...")
    
    # Approximate parameters for the figure-eight solution
    mass = 1.0  # Normalized mass
    
    # Initial positions (approximate)
    body1 = CelestialBody(
        mass=mass,
        position=[-1.0, 0.0],
        velocity=[0.347111, 0.532728],
        name="Body 1",
        color="red"
    )
    
    body2 = CelestialBody(
        mass=mass,
        position=[1.0, 0.0],
        velocity=[0.347111, 0.532728],
        name="Body 2",
        color="green"
    )
    
    body3 = CelestialBody(
        mass=mass,
        position=[0.0, 0.0],
        velocity=[-0.694222, -1.065456],
        name="Body 3",
        color="blue"
    )
    
    return [body1, body2, body3]

def create_lagrange_triangle_system():
    """Create a Lagrange triangular configuration (equilateral triangle)."""
    print("Creating Lagrange Triangle System...")
    
    # Three equal masses at vertices of equilateral triangle
    mass = 1.989e30  # Solar mass
    distance = 1.496e8  # 1 AU
    
    # Calculate orbital velocity for circular motion
    # v = sqrt(G * M_total / R) where R is circumradius
    circumradius = distance / np.sqrt(3)
    orbital_velocity = np.sqrt(6.67430e-11 * 3 * mass / (circumradius * 1000)) / 1000  # km/s
    
    # Positions at vertices of equilateral triangle
    body1 = CelestialBody(
        mass=mass,
        position=[distance, 0],
        velocity=[0, orbital_velocity],
        name="Star 1",
        color="red"
    )
    
    body2 = CelestialBody(
        mass=mass,
        position=[-distance/2, distance * np.sqrt(3)/2],
        velocity=[-orbital_velocity * np.sqrt(3)/2, -orbital_velocity/2],
        name="Star 2",
        color="green"
    )
    
    body3 = CelestialBody(
        mass=mass,
        position=[-distance/2, -distance * np.sqrt(3)/2],
        velocity=[orbital_velocity * np.sqrt(3)/2, -orbital_velocity/2],
        name="Star 3",
        color="blue"
    )
    
    return [body1, body2, body3]

def create_restricted_three_body_system():
    """
    Create a restricted three-body problem with two massive bodies and one test particle.
    """
    print("Creating Restricted Three-Body System...")
    
    # Two massive bodies in circular orbit (Jupiter-Sun system)
    sun_mass = 1.989e30
    jupiter_mass = 1.898e27
    separation = 7.783e8  # km (Jupiter's orbital distance)
    
    # Calculate center of mass
    total_mass = sun_mass + jupiter_mass
    sun_distance = jupiter_mass * separation / total_mass
    jupiter_distance = sun_mass * separation / total_mass
    
    # Orbital velocity
    orbital_velocity = np.sqrt(6.67430e-11 * total_mass / (separation * 1000)) / 1000  # km/s
    
    sun = CelestialBody(
        mass=sun_mass,
        position=[-sun_distance, 0],
        velocity=[0, -orbital_velocity * jupiter_mass / total_mass],
        name="Sun",
        color="yellow"
    )
    
    jupiter = CelestialBody(
        mass=jupiter_mass,
        position=[jupiter_distance, 0],
        velocity=[0, orbital_velocity * sun_mass / total_mass],
        name="Jupiter",
        color="orange"
    )
    
    # Test particle at L4 Lagrange point (60 degrees ahead of Jupiter)
    l4_angle = np.pi / 3  # 60 degrees
    l4_x = jupiter_distance * np.cos(l4_angle)
    l4_y = jupiter_distance * np.sin(l4_angle)
    
    test_particle = CelestialBody(
        mass=1e20,  # Small mass (asteroid-like)
        position=[l4_x, l4_y],
        velocity=[-orbital_velocity * np.sin(l4_angle) * sun_mass / total_mass,
                  orbital_velocity * np.cos(l4_angle) * sun_mass / total_mass],
        name="Test Particle",
        color="gray"
    )
    
    return [sun, jupiter, test_particle]

def create_chaotic_three_body_system():
    """Create a chaotic three-body system with random initial conditions."""
    print("Creating Chaotic Three-Body System...")
    
    # Three bodies with random masses and positions
    np.random.seed(42)  # For reproducibility
    
    bodies = []
    for i in range(3):
        mass = np.random.uniform(1e29, 1e31)  # kg
        position = np.random.uniform(-1e8, 1e8, 2)  # km
        velocity = np.random.uniform(-10, 10, 2)    # km/s
        
        body = CelestialBody(
            mass=mass,
            position=position,
            velocity=velocity,
            name=f"Body {i+1}",
            color=['red', 'green', 'blue'][i]
        )
        bodies.append(body)
    
    return bodies

def run_simulation(bodies, simulation_name, time_span=10.0, n_points=1000):
    """
    Run a three-body simulation and create visualizations.
    
    Args:
        bodies: List of CelestialBody objects
        simulation_name: Name for the simulation
        time_span: Total simulation time
        n_points: Number of time points
    """
    print(f"\n{'='*60}")
    print(f"Running {simulation_name}")
    print(f"{'='*60}")
    
    # Display initial conditions
    print("\nInitial Conditions:")
    total_mass = 0
    center_of_mass_pos = np.zeros(2)
    center_of_mass_vel = np.zeros(2)
    
    for body in bodies:
        print(f"  {body}")
        total_mass += body.mass
        center_of_mass_pos += body.mass * body.position
        center_of_mass_vel += body.mass * body.velocity
    
    center_of_mass_pos /= total_mass
    center_of_mass_vel /= total_mass
    
    print(f"\nSystem Properties:")
    print(f"  Total Mass: {total_mass:.2e} kg")
    print(f"  Center of Mass Position: [{center_of_mass_pos[0]:.2e}, {center_of_mass_pos[1]:.2e}] km")
    print(f"  Center of Mass Velocity: [{center_of_mass_vel[0]:.3f}, {center_of_mass_vel[1]:.3f}] km/s")
    
    # Create simulation
    simulation = NBodySimulation(bodies)
    
    # Run simulation
    print(f"\nRunning simulation for {time_span} time units...")
    t_span = (0, time_span)
    t_eval = np.linspace(0, time_span, n_points)
    
    try:
        positions, velocities, times, energies = simulation.simulate(
            t_span=t_span,
            t_eval=t_eval,
            method='rk4'
        )
        
        print("Simulation completed successfully!")
        
        # Energy conservation analysis
        initial_energy = energies['total'][0]
        final_energy = energies['total'][-1]
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
        
        print(f"\nEnergy Conservation:")
        print(f"  Initial Total Energy: {initial_energy:.6e} J")
        print(f"  Final Total Energy:   {final_energy:.6e} J")
        print(f"  Relative Energy Drift: {energy_drift:.8f}%")
        
        # Calculate system statistics
        print(f"\nSystem Statistics:")
        
        # Calculate maximum separation
        max_separation = 0
        min_separation = float('inf')
        
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                separations = np.sqrt(np.sum((positions[:, i, :] - positions[:, j, :]) ** 2, axis=1))
                max_sep = np.max(separations)
                min_sep = np.min(separations)
                
                if max_sep > max_separation:
                    max_separation = max_sep
                if min_sep < min_separation:
                    min_separation = min_sep
                
                print(f"  {bodies[i].name} - {bodies[j].name}:")
                print(f"    Max separation: {max_sep:.2e} km")
                print(f"    Min separation: {min_sep:.2e} km")
        
        # Individual body statistics
        for i, body in enumerate(bodies):
            distances = np.sqrt(np.sum(positions[:, i, :] ** 2, axis=1))
            speeds = np.sqrt(np.sum(velocities[:, i, :] ** 2, axis=1))
            
            print(f"\n  {body.name} Statistics:")
            print(f"    Max distance from origin: {np.max(distances):.2e} km")
            print(f"    Min distance from origin: {np.min(distances):.2e} km")
            print(f"    Max speed: {np.max(speeds):.3f} km/s")
            print(f"    Min speed: {np.min(speeds):.3f} km/s")
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # Static trajectory plot
        try:
            fig_traj = plot_trajectories(positions, bodies, show_trails=True)
            print("  ✓ Trajectory plot created")
        except Exception as e:
            print(f"  ✗ Error creating trajectory plot: {e}")
        
        # Energy conservation plot
        try:
            fig_energy = plot_energy_conservation(times, energies)
            print("  ✓ Energy conservation plot created")
        except Exception as e:
            print(f"  ✗ Error creating energy plot: {e}")
        
        # Body separation plot
        try:
            fig_sep = plot_body_separations(positions, bodies, times)
            print("  ✓ Body separation plot created")
        except Exception as e:
            print(f"  ✗ Error creating separation plot: {e}")
        
        # Animation
        try:
            print("  Creating animation... (this may take a moment)")
            anim = create_matplotlib_animation(
                positions, bodies, times,
                interval=50, trail_length=100
            )
            print("  ✓ Animation created")
            
            # Show the animation
            plt.show()
            
        except Exception as e:
            print(f"  ✗ Error creating animation: {e}")
            
            # Fallback: show static plots
            print("  Showing static plots instead...")
            try:
                from visualization.plot import create_matplotlib_trajectories
                fig = create_matplotlib_trajectories(positions, bodies)
                plt.show()
            except Exception as e2:
                print(f"  ✗ Error showing fallback plots: {e2}")
        
        # Chaos analysis for chaotic systems
        if "Chaotic" in simulation_name:
            try:
                analyze_chaos(positions, velocities, times, bodies)
            except Exception as e:
                print(f"  ✗ Error in chaos analysis: {e}")
        
        return positions, velocities, times, energies
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None, None, None

def analyze_chaos(positions, velocities, times, bodies):
    """Analyze chaotic behavior in the three-body system."""
    print(f"\n{'='*40}")
    print("Chaos Analysis")
    print(f"{'='*40}")
    
    # Calculate Lyapunov exponent (simplified approach)
    n_times, n_bodies, _ = positions.shape
    
    # Calculate trajectory divergence over time
    # This is a simplified analysis - true Lyapunov calculation is more complex
    
    print("Analyzing trajectory stability...")
    
    # Look at the rate of separation between bodies
    separations = []
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            sep = np.sqrt(np.sum((positions[:, i, :] - positions[:, j, :]) ** 2, axis=1))
            separations.append(sep)
    
    # Plot separation evolution
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for i, sep in enumerate(separations):
            plt.plot(times, sep, label=f'Bodies {i//1} & {i%n_bodies + 1}')
        plt.xlabel('Time')
        plt.ylabel('Separation (km)')
        plt.title('Body Separations Over Time')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        # Plot relative changes in separation
        for i, sep in enumerate(separations):
            relative_change = np.abs(np.diff(sep)) / sep[:-1]
            plt.plot(times[1:], relative_change, label=f'Bodies {i//1} & {i%n_bodies + 1}')
        plt.xlabel('Time')
        plt.ylabel('Relative Change in Separation')
        plt.title('Rate of Separation Change')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        # Phase space plot for first body
        plt.plot(positions[:, 0, 0], velocities[:, 0, 0], 'b-', alpha=0.7)
        plt.xlabel('Position X (km)')
        plt.ylabel('Velocity X (km/s)')
        plt.title(f'{bodies[0].name} - Phase Space (X)')
        
        plt.subplot(2, 2, 4)
        # Energy fluctuations (indicator of numerical errors or chaos)
        # This would need to be passed from the main simulation
        plt.text(0.5, 0.5, 'Energy analysis\nwould go here\n(needs energy data)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Energy Analysis')
        
        plt.tight_layout()
        plt.show()
        
        print("  ✓ Chaos analysis plots created")
        
    except Exception as e:
        print(f"  ✗ Error creating chaos analysis: {e}")

def main():
    """Main function to run all three-body demonstrations."""
    print("Three-Body Problem Demonstration")
    print("="*60)
    
    # List of three-body systems to simulate
    systems = [
        (create_sun_earth_moon_system, "Sun-Earth-Moon System", 1.0, 1000),
        (create_lagrange_triangle_system, "Lagrange Triangle System", 2.0, 1000),
        (create_restricted_three_body_system, "Restricted Three-Body System", 5.0, 1500),
        (create_figure_eight_system, "Figure-Eight System", 20.0, 2000),
        (create_chaotic_three_body_system, "Chaotic Three-Body System", 10.0, 1500)
    ]
    
    for system_func, name, time_span, n_points in systems:
        try:
            bodies = system_func()
            positions, velocities, times, energies = run_simulation(
                bodies, name, time_span, n_points
            )
            
            if positions is not None:
                print(f"\n{name} completed successfully!")
            
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user.")
            break
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue
        
        # Ask if user wants to continue
        if name != systems[-1][1]:  # Not the last system
            response = input(f"\nContinue to next system? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\nThree-body problem demonstrations completed!")
    print("\nKey Observations:")
    print("- The Sun-Earth-Moon system shows the complexity of real celestial mechanics")
    print("- The Lagrange triangle demonstrates stable configurations")
    print("- The restricted three-body problem shows Lagrange points")
    print("- The figure-eight solution shows a special periodic orbit")
    print("- Chaotic systems demonstrate sensitive dependence on initial conditions")

if __name__ == "__main__":
    main()

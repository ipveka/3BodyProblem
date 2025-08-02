#!/usr/bin/env python3
"""
Two-Body Problem Demonstration Script

This script demonstrates the classic two-body problem in gravitational dynamics,
showing stable elliptical orbits between two massive bodies.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.body import CelestialBody, PresetBodies
from core.simulation import NBodySimulation
from visualization.plot import plot_trajectories, plot_energy_conservation
from visualization.animate import create_matplotlib_animation

def create_earth_moon_system():
    """Create Earth-Moon two-body system with realistic parameters."""
    print("Creating Earth-Moon System...")
    
    # Earth at origin
    earth = CelestialBody(
        mass=5.972e24,  # kg
        position=[0, 0],
        velocity=[0, 0],
        name="Earth",
        color="blue"
    )
    
    # Moon in circular orbit
    moon_distance = 384400  # km
    moon_orbital_velocity = 1.022  # km/s
    
    moon = CelestialBody(
        mass=7.342e22,  # kg
        position=[moon_distance, 0],
        velocity=[0, moon_orbital_velocity],
        name="Moon",
        color="gray"
    )
    
    return [earth, moon]

def create_binary_star_system():
    """Create a binary star system with two equal-mass stars."""
    print("Creating Binary Star System...")
    
    # Calculate parameters for circular orbit
    total_mass = 2 * 1.989e30  # Two solar masses
    separation = 2 * 1.496e8   # 2 AU in km
    orbital_velocity = np.sqrt(6.67430e-11 * total_mass / (separation * 1000)) / 1000  # km/s
    
    star1 = CelestialBody(
        mass=1.989e30,  # Solar mass
        position=[-separation/2, 0],
        velocity=[0, -orbital_velocity/2],
        name="Star 1",
        color="red"
    )
    
    star2 = CelestialBody(
        mass=1.989e30,  # Solar mass
        position=[separation/2, 0],
        velocity=[0, orbital_velocity/2],
        name="Star 2",
        color="orange"
    )
    
    return [star1, star2]

def create_eccentric_orbit_system():
    """Create a system with an eccentric elliptical orbit."""
    print("Creating Eccentric Orbit System...")
    
    # Massive central body
    central = CelestialBody(
        mass=1.989e30,  # Solar mass
        position=[0, 0],
        velocity=[0, 0],
        name="Central Star",
        color="yellow"
    )
    
    # Orbiting body with eccentric orbit
    # Start at aphelion with lower velocity for elliptical orbit
    orbiter = CelestialBody(
        mass=5.972e24,  # Earth mass
        position=[2.0 * 1.496e8, 0],  # 2 AU
        velocity=[0, 15.0],  # Reduced velocity for elliptical orbit
        name="Planet",
        color="blue"
    )
    
    return [central, orbiter]

def run_simulation(bodies, simulation_name, time_span=10.0, n_points=1000):
    """
    Run a two-body simulation and create visualizations.
    
    Args:
        bodies: List of CelestialBody objects
        simulation_name: Name for the simulation
        time_span: Total simulation time
        n_points: Number of time points
    """
    print(f"\n{'='*50}")
    print(f"Running {simulation_name}")
    print(f"{'='*50}")
    
    # Display initial conditions
    print("\nInitial Conditions:")
    for body in bodies:
        print(f"  {body}")
    
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
        
        # Calculate orbital statistics
        for i, body in enumerate(bodies):
            if body.name != "Earth" and body.name != "Central Star":  # Skip central bodies
                distances = np.sqrt(np.sum(positions[:, i, :] ** 2, axis=1))
                speeds = np.sqrt(np.sum(velocities[:, i, :] ** 2, axis=1))
                
                print(f"\n{body.name} Statistics:")
                print(f"  Max distance from origin: {np.max(distances):.2e} km")
                print(f"  Min distance from origin: {np.min(distances):.2e} km")
                print(f"  Max orbital speed: {np.max(speeds):.3f} km/s")
                print(f"  Min orbital speed: {np.min(speeds):.3f} km/s")
        
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
        
        # Animation
        try:
            print("  Creating animation... (this may take a moment)")
            anim = create_matplotlib_animation(
                positions, bodies, times,
                interval=50, trail_length=50
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
        
        return positions, velocities, times, energies
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None, None, None

def compare_integration_methods(bodies, simulation_name):
    """Compare different integration methods for accuracy."""
    print(f"\n{'='*50}")
    print(f"Integration Method Comparison - {simulation_name}")
    print(f"{'='*50}")
    
    methods = ['euler', 'rk4', 'verlet']
    time_span = 5.0
    n_points = 500
    
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        try:
            simulation = NBodySimulation(bodies)
            t_span = (0, time_span)
            t_eval = np.linspace(0, time_span, n_points)
            
            positions, velocities, times, energies = simulation.simulate(
                t_span=t_span,
                t_eval=t_eval,
                method=method
            )
            
            # Calculate energy drift
            initial_energy = energies['total'][0]
            final_energy = energies['total'][-1]
            energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
            
            results[method] = {
                'energy_drift': energy_drift,
                'energies': energies,
                'times': times
            }
            
            print(f"  {method.upper()} energy drift: {energy_drift:.8f}%")
            
        except Exception as e:
            print(f"  {method.upper()} failed: {e}")
            results[method] = None
    
    # Plot comparison
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for method, result in results.items():
            if result is not None:
                plt.plot(result['times'], result['energies']['total'], 
                        label=f'{method.upper()}', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Total Energy (J)')
        plt.title('Total Energy vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        methods_list = []
        drifts = []
        for method, result in results.items():
            if result is not None:
                methods_list.append(method.upper())
                drifts.append(result['energy_drift'])
        
        plt.bar(methods_list, drifts)
        plt.ylabel('Energy Drift (%)')
        plt.title('Energy Conservation Comparison')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating comparison plots: {e}")

def main():
    """Main function to run all two-body demonstrations."""
    print("Two-Body Problem Demonstration")
    print("="*50)
    
    # List of two-body systems to simulate
    systems = [
        (create_earth_moon_system, "Earth-Moon System", 30.0, 1000),
        (create_binary_star_system, "Binary Star System", 2.0, 1000),
        (create_eccentric_orbit_system, "Eccentric Orbit System", 5.0, 1500)
    ]
    
    for system_func, name, time_span, n_points in systems:
        try:
            bodies = system_func()
            positions, velocities, times, energies = run_simulation(
                bodies, name, time_span, n_points
            )
            
            if positions is not None:
                # Ask user if they want to see method comparison
                response = input(f"\nRun integration method comparison for {name}? (y/n): ")
                if response.lower() == 'y':
                    compare_integration_methods(bodies, name)
            
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
    
    print("\nTwo-body problem demonstrations completed!")

if __name__ == "__main__":
    main()

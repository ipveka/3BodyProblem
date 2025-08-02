import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import time

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.body import CelestialBody
from core.simulation import NBodySimulation
from visualization.animate import create_animated_plot
from visualization.plot import plot_energy_conservation, plot_trajectories

# Page configuration
st.set_page_config(
    page_title="N-Body Gravitational Simulation",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌌 N-Body Gravitational Simulation")
st.markdown("*Interactive visualization of gravitational orbital mechanics*")

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = None
if 'bodies' not in st.session_state:
    st.session_state.bodies = []
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None

# Sidebar controls with improved layout
with st.sidebar:
    st.title("🌌 N-Body Simulator")
    st.markdown("*Gravitational Physics Engine*")
    st.markdown("---")
    st.header("⚙️ Simulation Controls")
    
    with st.expander("🔧 Parameters", expanded=True):
        simulation_years = st.slider("Simulation Years", 0.1, 5.0, 1.0, 0.1,
                                    help="Duration in Earth years (365.25 days each)")
        days_per_year = 365.25
        total_days = simulation_years * days_per_year
        time_step_days = 1.0  # Daily positions
        
        st.write(f"**Total simulation:** {total_days:.1f} days")
        st.write(f"**Time step:** {time_step_days} day per calculation")
        
        simulation_method = st.selectbox("Integration Method", ["RK4", "Euler", "Verlet"],
                                       help="RK4: High accuracy | Euler: Simple | Verlet: Energy conserving")
    
    with st.expander("🎯 Body Management", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Random Body", use_container_width=True):
                # Generate random body parameters with better scaling
                mass = np.random.uniform(1e24, 1e30)  # kg
                x = np.random.uniform(-5e8, 5e8)  # km
                y = np.random.uniform(-5e8, 5e8)  # km
                vx = np.random.uniform(-20, 20)   # km/s
                vy = np.random.uniform(-20, 20)   # km/s
                
                # Generate random color
                colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "lime"]
                color = np.random.choice(colors)
                
                body = CelestialBody(
                    mass=mass,
                    position=[x, y],
                    velocity=[vx, vy],
                    name=f"Body {len(st.session_state.bodies) + 1}",
                    color=color
                )
                st.session_state.bodies.append(body)
                st.rerun()

        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.bodies = []
                st.session_state.simulation_data = None
                st.rerun()

# Main content area
st.subheader("🪐 Preset Systems")
st.markdown("Choose from pre-configured gravitational systems:")

# Preset configurations with side-by-side buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌍 Earth-Moon", use_container_width=True):
        st.session_state.bodies = []
        # Earth at origin (stationary for simplified two-body problem)
        earth = CelestialBody(
            mass=5.972e24,  # kg
            position=[0, 0],
            velocity=[0, 0],
            name="Earth",
            color="#4a90e2"
        )
        # Moon in realistic orbit (using actual Earth-Moon distance)
        moon_distance = 384400  # km (actual Earth-Moon distance)
        # Calculate proper orbital velocity: v = sqrt(GM/r)
        G = 6.67430e-11  # m³ kg⁻¹ s⁻²
        earth_mass = 5.972e24  # kg
        orbital_velocity = np.sqrt(G * earth_mass / (moon_distance * 1000)) / 1000  # convert to km/s
        
        moon = CelestialBody(
            mass=7.342e22,  # kg
            position=[moon_distance, 0],  # km
            velocity=[0, orbital_velocity],  # km/s (circular orbit)
            name="Moon",
            color="#b0b0b0"
        )
        st.session_state.bodies = [earth, moon]
        st.success(f"Earth-Moon system loaded! Distance: {moon_distance:,} km, Orbital velocity: {orbital_velocity:.3f} km/s. Try 1 year simulation.")
        st.rerun()

with col2:
    if st.button("☀️ Sun-Earth-Moon", use_container_width=True):
        st.session_state.bodies = []
        # Use scaled distances for better visualization while keeping physics realistic
        # Scale factor: divide by 1000 for visualization, masses stay realistic
        scale_factor = 1000
        
        # Sun at center
        sun = CelestialBody(
            mass=1.989e30,  # kg
            position=[0, 0],
            velocity=[0, 0],
            name="Sun",
            color="#ffd700"
        )
        
        # Earth at scaled distance
        earth_distance = 149597870.7 / scale_factor  # km (scaled 1 AU)
        G = 6.67430e-11  # m³ kg⁻¹ s⁻²
        earth_velocity = np.sqrt(G * 1.989e30 / (earth_distance * 1000)) / 1000  # km/s
        
        earth = CelestialBody(
            mass=5.972e24,  # kg
            position=[earth_distance, 0],
            velocity=[0, earth_velocity],
            name="Earth",
            color="#4a90e2"
        )
        
        # Moon relative to Earth at scaled distance
        moon_distance = 384400 / scale_factor  # km (scaled distance)
        moon_velocity = np.sqrt(G * 5.972e24 / (moon_distance * 1000)) / 1000  # km/s
        
        moon = CelestialBody(
            mass=7.342e22,  # kg
            position=[earth_distance + moon_distance, 0],
            velocity=[0, earth_velocity + moon_velocity],
            name="Moon",
            color="#b0b0b0"
        )
        
        st.session_state.bodies = [sun, earth, moon]
        st.success(f"Sun-Earth-Moon system loaded (scaled {scale_factor}x for visualization)")
        st.rerun()

with col3:
    if st.button("🛰️ Custom System", use_container_width=True):
        st.session_state.bodies = []
        # Start with Sun-Earth-Moon system
        G = 6.67430e-11
        
        # Sun
        sun = CelestialBody(
            mass=1.989e30,
            position=[0, 0],
            velocity=[0, 0],
            name="Sun",
            color="#ffd700"
        )
        
        # Earth (scaled for better visualization)
        scale_factor = 1000  # Scale distances for visualization
        earth_distance = 149597870.7 / scale_factor  # km (scaled 1 AU)
        earth_velocity = np.sqrt(G * 1.989e30 / (earth_distance * 1000)) / 1000
        earth = CelestialBody(
            mass=5.972e24,
            position=[earth_distance, 0],
            velocity=[0, earth_velocity],
            name="Earth",
            color="#4a90e2"
        )
        
        # Moon (relative to Earth, scaled)
        moon_distance = 384400 / scale_factor  # km (scaled distance)
        moon_velocity = np.sqrt(G * 5.972e24 / (moon_distance * 1000)) / 1000
        moon = CelestialBody(
            mass=7.342e22,
            position=[earth_distance + moon_distance, 0],
            velocity=[0, earth_velocity + moon_velocity],
            name="Moon",
            color="#b0b0b0"
        )
        
        st.session_state.bodies = [sun, earth, moon]
        st.success(f"Custom system base loaded (Sun-Earth-Moon, scaled {scale_factor}x for visualization). Add more bodies below!")
        st.rerun()

# Add custom body form in sidebar (only show when custom system is loaded)
with st.sidebar:
    if len(st.session_state.bodies) >= 3:  # Show custom form when we have base system
        with st.expander("➕ Add Custom Body to System", expanded=False):
            st.markdown("**Add to existing Sun-Earth-Moon system:**")
            
            with st.form("add_body_form"):
                body_name = st.text_input("Body Name", "Asteroid")
                
                # Preset options for common body types
                body_type = st.selectbox("Body Type", 
                    ["Custom", "Asteroid", "Comet", "Small Planet", "Large Planet"])
                
                # Use same scale factor as the base system for consistency
                scale_factor = 1000
                
                if body_type == "Asteroid":
                    mass = 1e20  # kg
                    distance = 3e8 / scale_factor  # km (scaled)
                    orbital_velocity = 15  # km/s
                    color = "#8B4513"
                elif body_type == "Comet":
                    mass = 1e13  # kg
                    distance = 5e8 / scale_factor  # km (scaled)
                    orbital_velocity = 10  # km/s
                    color = "#87CEEB"
                elif body_type == "Small Planet":
                    mass = 6e23  # kg (Mars-like)
                    distance = 2.3e8 / scale_factor  # km (scaled)
                    orbital_velocity = 20  # km/s
                    color = "#CD853F"
                elif body_type == "Large Planet":
                    mass = 1.9e27  # kg (Jupiter-like)
                    distance = 7.8e8 / scale_factor  # km (scaled)
                    orbital_velocity = 8  # km/s
                    color = "#FFA500"
                else:  # Custom
                    mass = st.number_input("Mass (kg)", value=1e24, format="%.2e")
                    distance = st.number_input("Distance from Sun (km, scaled)", value=3e8/1000, format="%.2e", 
                                             help="Distance already scaled for visualization")
                    orbital_velocity = st.number_input("Orbital Velocity (km/s)", value=15.0)
                    color = st.color_picker("Color", "#FF0000")
                
                if body_type != "Custom":
                    mass = st.number_input("Mass (kg)", value=mass, format="%.2e", key="preset_mass")
                    distance = st.number_input("Distance from Sun (km)", value=distance, format="%.2e", key="preset_distance")
                    orbital_velocity = st.number_input("Orbital Velocity (km/s)", value=orbital_velocity, key="preset_velocity")
                
                # Calculate position (random angle around sun)
                angle = st.slider("Orbital Position (degrees)", 0, 360, 90)
                angle_rad = np.radians(angle)
                
                pos_x = distance * np.cos(angle_rad)
                pos_y = distance * np.sin(angle_rad)
                vel_x = -orbital_velocity * np.sin(angle_rad)
                vel_y = orbital_velocity * np.cos(angle_rad)
                
                if st.form_submit_button("Add to System", use_container_width=True):
                    body = CelestialBody(
                        mass=mass,
                        position=[pos_x, pos_y],
                        velocity=[vel_x, vel_y],
                        name=body_name,
                        color=color
                    )
                    st.session_state.bodies.append(body)
                    st.success(f"Added {body_name} to the system!")
                    st.rerun()

# Display current bodies if any exist
if len(st.session_state.bodies) > 0:
    st.subheader("📊 Current System")
    
    # Create a more visual display of bodies
    cols = st.columns(min(len(st.session_state.bodies), 4))
    for i, body in enumerate(st.session_state.bodies):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {body.color}; background-color: #f8f9fa; margin: 5px 0;">
            <strong>{body.name}</strong><br>
            Mass: {body.mass:.2e} kg<br>
            Pos: ({body.position[0]:.0f}, {body.position[1]:.0f}) km<br>
            Vel: ({body.velocity[0]:.2f}, {body.velocity[1]:.2f}) km/s
            </div>
            """, unsafe_allow_html=True)
    
    # Run simulation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner(f"Simulating {total_days:.0f} days of orbital motion..."):
                # Create simulation
                simulation = NBodySimulation(st.session_state.bodies)
                
                # Convert days to time units (assuming time units are in some scaled form)
                # We'll use days as our time unit for simplicity
                time_step_seconds = time_step_days * 24 * 3600  # Convert days to seconds
                total_time_seconds = total_days * 24 * 3600
                
                t_span = (0, total_time_seconds)
                n_points = int(total_days / time_step_days) + 1  # Daily positions
                t_eval = np.linspace(0, total_time_seconds, n_points)
                
                positions, velocities, times, energies = simulation.simulate(
                    t_span=t_span,
                    t_eval=t_eval,
                    method=simulation_method.lower()
                )
                
                # Convert times back to days for display
                times_days = times / (24 * 3600)
                
                st.session_state.simulation_data = {
                    'positions': positions,
                    'velocities': velocities,
                    'times': times_days,
                    'energies': energies,
                    'bodies': st.session_state.bodies.copy(),
                    'simulation_years': simulation_years,
                    'total_days': total_days
                }
                
            st.success(f"Simulation completed! {len(times)} data points over {total_days:.0f} days")
            st.rerun()

# Display results if simulation has been run
if st.session_state.simulation_data is not None:
    data = st.session_state.simulation_data
    
    st.subheader(f"🛰️ Orbital Trajectories - {data['simulation_years']:.1f} Years ({data['total_days']:.0f} Days)")
    
    # Create matplotlib plot with consistent sizing
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate plot boundaries for consistent scaling
    all_positions = data['positions'].reshape(-1, 2)
    x_min, y_min = np.min(all_positions, axis=0)
    x_max, y_max = np.max(all_positions, axis=0)
    
    # Add 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    margin = max_range * 0.1
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Plot trajectories for each body
    for i, body in enumerate(data['bodies']):
        positions = data['positions'][:, i, :]
        
        # Main orbital trajectory
        ax.plot(positions[:, 0], positions[:, 1], 
               color=body.color, linewidth=2.5, alpha=0.8, 
               label=f'{body.name} Orbit')
        
        # Starting position (green dot)
        ax.plot(positions[0, 0], positions[0, 1], 
               'o', color='green', markersize=12, 
               markeredgecolor='black', markeredgewidth=1.5,
               zorder=5)
        
        # Final position (red square)  
        ax.plot(positions[-1, 0], positions[-1, 1], 
               's', color='red', markersize=12,
               markeredgecolor='black', markeredgewidth=1.5,
               zorder=5)
        
        # Current body position (colored circle with label)
        ax.plot(positions[-1, 0], positions[-1, 1], 
               'o', color=body.color, markersize=10,
               markeredgecolor='white', markeredgewidth=2,
               zorder=6)
        
        # Add final position label
        ax.annotate(f'{body.name}\nFinal', 
                   xy=(positions[-1, 0], positions[-1, 1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=body.color, alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Set consistent plot limits
    ax.set_xlim(x_center - max_range/2 - margin, x_center + max_range/2 + margin)
    ax.set_ylim(y_center - max_range/2 - margin, y_center + max_range/2 + margin)
    
    # Set equal aspect ratio for proper orbital shapes
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Position X (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Y (km)', fontsize=12, fontweight='bold')
    ax.set_title(f'Gravitational Orbits - {data["simulation_years"]:.1f} Earth Years', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add scaling note if there are 3+ bodies (Sun-Earth-Moon systems)
    if len(data['bodies']) >= 3:
        ax.text(0.02, 0.98, 'Note: Distances scaled 1000x for visualization', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Enhanced legend
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9, 
             fancybox=True, shadow=True)
    
    # Add center of mass marker
    ax.plot(0, 0, '+', color='black', markersize=18, markeredgewidth=4, 
           label='Origin', zorder=7)
    ax.annotate('Origin (0,0)', xy=(0, 0), xytext=(15, 15), 
               textcoords='offset points', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # Final positions summary table
    st.subheader("📍 Final Positions Summary")
    final_positions_data = []
    for i, body in enumerate(data['bodies']):
        final_pos = data['positions'][-1, i, :]
        final_vel = data['velocities'][-1, i, :]
        distance_from_origin = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        speed = np.sqrt(final_vel[0]**2 + final_vel[1]**2)
        
        final_positions_data.append({
            "Body": body.name,
            "Final X (km)": f"{final_pos[0]:,.0f}",
            "Final Y (km)": f"{final_pos[1]:,.0f}",
            "Distance from Origin": f"{distance_from_origin:,.0f} km",
            "Final Speed": f"{speed:.3f} km/s",
            "Color": body.color
        })
    
    df_final = st.dataframe(final_positions_data, use_container_width=True)
    
    # Show total system statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bodies", len(data['bodies']))
    with col2:
        st.metric("Simulation Duration", f"{data['simulation_years']:.1f} years")
    with col3:  
        st.metric("Data Points", f"{len(data['times']):,}")
    
    # Add data table below the plot
    st.subheader("📊 Position Data Table")
    with st.expander("Show Daily Position Data", expanded=False):
        st.write(f"**Simulation Details:** {len(data['times'])} daily positions over {data['total_days']:.0f} days")
        
        # Create a comprehensive data table
        table_data = []
        for t_idx in range(0, len(data['times']), max(1, len(data['times'])//20)):  # Sample 20 points
            row = {
                "Day": f"{data['times'][t_idx]:.1f}",
                "Time Index": t_idx
            }
            
            for i, body in enumerate(data['bodies']):
                pos = data['positions'][t_idx, i, :]
                vel = data['velocities'][t_idx, i, :]
                distance = np.sqrt(pos[0]**2 + pos[1]**2)
                speed = np.sqrt(vel[0]**2 + vel[1]**2)
                
                row[f"{body.name} X (km)"] = f"{pos[0]:.1f}"
                row[f"{body.name} Y (km)"] = f"{pos[1]:.1f}" 
                row[f"{body.name} Distance"] = f"{distance:.1f}"
                row[f"{body.name} Speed"] = f"{speed:.3f}"
            
            table_data.append(row)
        
        st.dataframe(table_data, use_container_width=True)

else:
    # Welcome screen when no bodies are present
    st.info("🌟 Welcome to the N-Body Gravitational Simulation!")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Choose a preset system** from the buttons above, or
    2. **Add celestial bodies** using the sidebar controls
    3. **Adjust simulation parameters** to your liking
    4. **Run the simulation** to watch orbital mechanics in action!
    
    #### 🌍 Recommended Starting Points:
    - **Earth-Moon**: Classic two-body system showing lunar orbit
    - **Sun-Earth-Moon**: Complex three-body system with realistic parameters
    - **Custom System**: Start with Sun-Earth-Moon and add your own bodies
    """)

# Documentation Section
st.markdown("---")
st.header("📚 Physics & Mathematics Documentation")

with st.expander("🧮 Gravitational Physics", expanded=False):
    st.markdown("""
    ### Newton's Law of Universal Gravitation
    
    The fundamental equation governing the simulation:
    
    **F = G × (m₁ × m₂) / r²**
    
    Where:
    - **F**: Gravitational force between two bodies
    - **G**: Gravitational constant (6.674 × 10⁻¹¹ N⋅m²/kg²)
    - **m₁, m₂**: Masses of the two bodies
    - **r**: Distance between the centers of the bodies
    
    ### Equations of Motion
    
    For each body i with mass mᵢ at position **rᵢ**:
    
    **F̅ᵢ = Σⱼ≠ᵢ [G × mᵢ × mⱼ × (r̅ⱼ - r̅ᵢ) / |r̅ⱼ - r̅ᵢ|³]**
    
    **acceleration = F̅ᵢ / mᵢ**
    """)

with st.expander("🔢 Numerical Integration Methods", expanded=False):
    st.markdown("""
    ### Integration Techniques
    
    The simulation uses three different methods to solve the differential equations:
    
    #### 1. Runge-Kutta 4th Order (RK4)
    - **Accuracy**: Very high (4th order)
    - **Stability**: Excellent for smooth systems
    - **Usage**: Best for high-precision simulations
    - **Computational Cost**: Higher
    
    #### 2. Euler Method
    - **Accuracy**: Basic (1st order)
    - **Stability**: Good for small time steps
    - **Usage**: Educational purposes, simple systems
    - **Computational Cost**: Lowest
    
    #### 3. Velocity Verlet
    - **Accuracy**: Good (2nd order)
    - **Stability**: Excellent energy conservation
    - **Usage**: Long-term orbital simulations
    - **Computational Cost**: Moderate
    
    ### Time Step Considerations
    - **Smaller steps**: Higher accuracy, longer computation
    - **Larger steps**: Faster computation, potential instability
    - **Adaptive stepping**: Automatically adjusts for optimal balance
    """)

with st.expander("⚡ Energy Conservation", expanded=False):
    st.markdown("""
    ### Total Energy
    
    **E_total = E_kinetic + E_potential**
    
    #### Kinetic Energy
    **E_kinetic = ½ × Σᵢ mᵢ × |v̅ᵢ|²**
    
    Where v̅ᵢ is the velocity vector of body i.
    
    #### Gravitational Potential Energy
    **E_potential = -Σᵢ<ⱼ [G × mᵢ × mⱼ / |r̅ᵢ - r̅ⱼ|]**
    
    ### Conservation Principle
    In an ideal isolated system, total energy should remain constant. 
    Energy drift in simulations indicates:
    - Numerical integration errors
    - Time step too large
    - Method not suitable for the system
    
    ### Interpretation
    - **Energy drift < 0.01%**: Excellent conservation
    - **Energy drift < 1%**: Good conservation
    - **Energy drift > 1%**: Consider smaller time steps or different method
    """)

with st.expander("🛰️ Orbital Mechanics Concepts", expanded=False):
    st.markdown("""
    ### Orbital Elements
    
    #### Eccentricity (e)
    - **e = 0**: Perfect circle
    - **0 < e < 1**: Elliptical orbit
    - **e = 1**: Parabolic trajectory (escape velocity)
    - **e > 1**: Hyperbolic trajectory (unbound)
    
    #### Semi-major Axis (a)
    - Average distance from focus to orbit
    - **a = (r_max + r_min) / 2**
    
    #### Orbital Period
    For two-body systems: **T² ∝ a³** (Kepler's Third Law)
    
    ### N-Body Complexity
    - **Two bodies**: Analytical solution exists (Kepler's laws)
    - **Three bodies**: No general analytical solution (requires numerical methods)
    - **N bodies**: Increasingly chaotic behavior possible
    
    ### Stability
    - **Stable orbits**: Bodies remain bound, predictable motion
    - **Chaotic systems**: Small changes lead to vastly different outcomes
    - **Ejection**: Bodies gain enough energy to escape the system
    """)

# Footer
st.markdown("---")
st.markdown("""
**N-Body Gravitational Simulation** - Built with Streamlit, NumPy, and SciPy  
*Demonstrating Newtonian gravity through interactive numerical simulation*
""")

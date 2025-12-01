import streamlit as st
import numpy as np
import plotly.graph_objects as go
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
        # Random Body button on top
        if st.button("🎲 Random Body", width='stretch'):
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
        
        # Clear All button below (minimal spacing)
        if st.button("🗑️ Clear All", width='stretch'):
            st.session_state.bodies = []
            st.session_state.simulation_data = None
            st.rerun()

# Main content area
st.subheader("🪐 Preset Systems")
st.markdown("Choose from pre-configured gravitational systems:")

# Add spacing before preset buttons
st.markdown("<br>", unsafe_allow_html=True)

# Preset configurations with side-by-side buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌍 Earth-Moon", width='stretch'):
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
    if st.button("☀️ Sun-Earth-Moon", width='stretch'):
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
    if st.button("🛰️ Custom System", width='stretch'):
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
                
                if st.form_submit_button("Add to System", width='stretch'):
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

# Add spacing between preset systems and current system
st.markdown("<br>", unsafe_allow_html=True)

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
    
    # Add spacing before Run Simulation button
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run simulation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Simulation", type="primary", width='stretch'):
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
    
    # Add spacing before results section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader(f"🛰️ Orbital Trajectories - {data['simulation_years']:.1f} Years ({data['total_days']:.0f} Days)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create beautiful Plotly plot
    fig = go.Figure()
    
    # Calculate plot boundaries for consistent scaling
    all_positions = data['positions'].reshape(-1, 2)
    x_min, y_min = np.min(all_positions, axis=0)
    x_max, y_max = np.max(all_positions, axis=0)
    
    # Add 10% margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range) if max(x_range, y_range) > 0 else 1
    margin = max_range * 0.1
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Plot trajectories for each body
    for i, body in enumerate(data['bodies']):
        positions = data['positions'][:, i, :]
        
        # Ensure color is in proper format (handle hex colors)
        body_color = body.color if body.color.startswith('#') else f'#{body.color}' if len(body.color) == 6 else body.color
        
        # Main orbital trajectory with smooth lines
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='lines',
            name=f'{body.name} Orbit',
            line=dict(
                color=body_color,
                width=3.5,
                shape='spline',
                smoothing=1.2
            ),
            opacity=0.8,
            hovertemplate=f'<b>{body.name} Orbit</b><br>' +
                         'X: %{x:,.0f} km<br>' +
                         'Y: %{y:,.0f} km<br>' +
                         '<extra></extra>',
            legendgroup=body.name,
            showlegend=True
        ))
        
        # Starting position (green circle)
        fig.add_trace(go.Scatter(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            mode='markers',
            name=f'{body.name} Start',
            marker=dict(
                color='#00AA00',
                size=12,
                symbol='circle',
                line=dict(color='#003300', width=2),
                opacity=0.95
            ),
            hovertemplate=f'<b>{body.name} Start</b><br>' +
                         f'X: {positions[0, 0]:,.0f} km<br>' +
                         f'Y: {positions[0, 1]:,.0f} km<extra></extra>',
            legendgroup=body.name,
            showlegend=False
        ))
        
        # Final position (red square)
        fig.add_trace(go.Scatter(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            mode='markers',
            name=f'{body.name} End',
            marker=dict(
                color='#CC0000',
                size=12,
                symbol='square',
                line=dict(color='#660000', width=2),
                opacity=0.95
            ),
            hovertemplate=f'<b>{body.name} End</b><br>' +
                         f'X: {positions[-1, 0]:,.0f} km<br>' +
                         f'Y: {positions[-1, 1]:,.0f} km<extra></extra>',
            legendgroup=body.name,
            showlegend=False
        ))
        
        # Current body position (colored circle with label)
        fig.add_trace(go.Scatter(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            mode='markers+text',
            name=body.name,
            marker=dict(
                color=body_color,
                size=18,
                symbol='circle',
                line=dict(color='white', width=2.5),
                opacity=1.0
            ),
            text=[body.name],
            textposition='top center',
            textfont=dict(size=11, color=body_color, family='Arial Black'),
            hovertemplate=f'<b>{body.name} (Final Position)</b><br>' +
                         f'X: {positions[-1, 0]:,.0f} km<br>' +
                         f'Y: {positions[-1, 1]:,.0f} km<extra></extra>',
            legendgroup=body.name,
            showlegend=False
        ))
    
    # Add origin marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        name='Origin',
        marker=dict(
            color='black',
            size=12,
            symbol='x',
            line=dict(color='black', width=3),
            opacity=0.8
        ),
        text=['Origin (0,0)'],
        textposition='top right',
        textfont=dict(size=10, color='black', family='Arial'),
        hovertemplate='<b>Origin</b><br>X: 0 km<br>Y: 0 km<extra></extra>',
        showlegend=False
    ))
    
    # Update layout for beautiful, compact design
    fig.update_layout(
        title=dict(
            text=f'Gravitational Orbits - {data["simulation_years"]:.1f} Earth Years',
            font=dict(size=16, family='Arial', color='#1f1f1f'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Position X (km)',
            titlefont=dict(size=12, family='Arial'),
            tickfont=dict(size=10),
            range=[x_center - max_range/2 - margin, x_center + max_range/2 + margin],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(100, 100, 100, 0.5)'
        ),
        yaxis=dict(
            title='Position Y (km)',
            titlefont=dict(size=12, family='Arial'),
            tickfont=dict(size=10),
            range=[y_center - max_range/2 - margin, y_center + max_range/2 + margin],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(100, 100, 100, 0.5)'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        width=1200,  # Larger plot for better visibility
        height=800,  # Larger plot for better visibility
        margin=dict(l=80, r=150, t=80, b=80),
        hovermode='closest',
        template='plotly_white',
        autosize=False
    )
    
    # Add scaling note if there are 3+ bodies
    if len(data['bodies']) >= 3:
        fig.add_annotation(
            text='Note: Distances scaled 1000x for visualization',
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=9, color='#666666'),
            bgcolor='rgba(173, 216, 230, 0.7)',
            bordercolor='rgba(173, 216, 230, 1)',
            borderwidth=1,
            borderpad=4
        )
    
    # Display the plot - full width for larger display
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True, 
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'orbital_trajectories',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    })
    
    # Add spacing after plot
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    
    df_final = st.dataframe(final_positions_data, width='stretch')
    
    # Add spacing before statistics
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show total system statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bodies", len(data['bodies']))
    with col2:
        st.metric("Simulation Duration", f"{data['simulation_years']:.1f} years")
    with col3:  
        st.metric("Data Points", f"{len(data['times']):,}")
    
    # Add spacing before data table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        
        st.dataframe(table_data, width='stretch')

else:
    # Getting started instructions when no bodies are present
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
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
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


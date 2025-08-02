"""
Static Plotting Module for N-Body Simulation

This module provides functions to create static visualizations including
orbital trajectories, energy plots, and phase space diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import seaborn as sns

from core.body import CelestialBody

# Set style for matplotlib plots
plt.style.use('default')
sns.set_palette("husl")

def plot_trajectories(positions: np.ndarray, 
                     bodies: List[CelestialBody],
                     show_trails: bool = True,
                     show_initial: bool = True,
                     show_final: bool = True,
                     figsize: Tuple[int, int] = (12, 8)) -> go.Figure:
    """
    Create a static plot of orbital trajectories using Plotly.
    
    Args:
        positions (np.ndarray): Position data of shape (n_times, n_bodies, 2)
        bodies (List[CelestialBody]): List of celestial bodies
        show_trails (bool): Whether to show orbital trails
        show_initial (bool): Whether to mark initial positions
        show_final (bool): Whether to mark final positions
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    n_times, n_bodies, _ = positions.shape
    
    fig = go.Figure()
    
    for i, body in enumerate(bodies):
        # Trajectory trail
        if show_trails:
            fig.add_trace(go.Scatter(
                x=positions[:, i, 0],
                y=positions[:, i, 1],
                mode='lines',
                line=dict(color=body.color, width=2),
                name=f'{body.name} Trajectory',
                opacity=0.7
            ))
        
        # Initial position
        if show_initial:
            fig.add_trace(go.Scatter(
                x=[positions[0, i, 0]],
                y=[positions[0, i, 1]],
                mode='markers',
                marker=dict(
                    color=body.color,
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name=f'{body.name} Start',
                showlegend=True
            ))
        
        # Final position
        if show_final:
            fig.add_trace(go.Scatter(
                x=[positions[-1, i, 0]],
                y=[positions[-1, i, 1]],
                mode='markers',
                marker=dict(
                    color=body.color,
                    size=12,
                    symbol='square',
                    line=dict(color='black', width=2)
                ),
                name=f'{body.name} End',
                showlegend=True
            ))
    
    # Set equal aspect ratio and layout
    all_positions = positions.reshape(-1, 2)
    margin = 0.1
    x_range = np.ptp(all_positions[:, 0])
    y_range = np.ptp(all_positions[:, 1])
    x_center = np.mean(all_positions[:, 0])
    y_center = np.mean(all_positions[:, 1])
    
    max_range = max(x_range, y_range)
    
    fig.update_layout(
        title='Orbital Trajectories',
        xaxis=dict(
            title='Position X (km)',
            range=[x_center - max_range/2 * (1 + margin),
                   x_center + max_range/2 * (1 + margin)],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title='Position Y (km)',
            range=[y_center - max_range/2 * (1 + margin),
                   y_center + max_range/2 * (1 + margin)]
        ),
        showlegend=True,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        template='plotly_white'
    )
    
    return fig

def plot_energy_conservation(times: np.ndarray, 
                           energies: Dict[str, np.ndarray],
                           figsize: Tuple[int, int] = (12, 8)) -> go.Figure:
    """
    Create energy conservation plots using Plotly.
    
    Args:
        times (np.ndarray): Time array
        energies (Dict[str, np.ndarray]): Dictionary containing energy arrays
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Energy', 'Energy Components', 
                       'Energy Drift', 'Energy Drift (Relative)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Total energy plot
    fig.add_trace(
        go.Scatter(x=times, y=energies['total'],
                  mode='lines', name='Total Energy',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Energy components
    fig.add_trace(
        go.Scatter(x=times, y=energies['kinetic'],
                  mode='lines', name='Kinetic Energy',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=times, y=energies['potential'],
                  mode='lines', name='Potential Energy',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Energy drift (absolute)
    initial_energy = energies['total'][0]
    energy_drift = energies['total'] - initial_energy
    
    fig.add_trace(
        go.Scatter(x=times, y=energy_drift,
                  mode='lines', name='Energy Drift',
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )
    
    # Energy drift (relative)
    relative_drift = np.abs(energy_drift) / np.abs(initial_energy) * 100
    
    fig.add_trace(
        go.Scatter(x=times, y=relative_drift,
                  mode='lines', name='Relative Drift (%)',
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    
    fig.update_yaxes(title_text="Energy (J)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (J)", row=1, col=2)
    fig.update_yaxes(title_text="Energy Drift (J)", row=2, col=1)
    fig.update_yaxes(title_text="Relative Drift (%)", row=2, col=2)
    
    fig.update_layout(
        title='Energy Conservation Analysis',
        showlegend=False,
        width=figsize[0] * 80,
        height=figsize[1] * 80,
        template='plotly_white'
    )
    
    return fig

def plot_phase_space(positions: np.ndarray, 
                    velocities: np.ndarray,
                    bodies: List[CelestialBody],
                    body_index: int = 0) -> go.Figure:
    """
    Create phase space plot (position vs velocity) for a specific body.
    
    Args:
        positions (np.ndarray): Position data
        velocities (np.ndarray): Velocity data
        bodies (List[CelestialBody]): List of celestial bodies
        body_index (int): Index of body to plot
        
    Returns:
        plotly.graph_objects.Figure: Phase space plot
    """
    if body_index >= len(bodies):
        raise ValueError("Body index out of range")
    
    body = bodies[body_index]
    pos = positions[:, body_index, :]
    vel = velocities[:, body_index, :]
    
    # Create subplots for x and y phase spaces
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{body.name} - X Phase Space', 
                       f'{body.name} - Y Phase Space')
    )
    
    # X phase space (x vs vx)
    fig.add_trace(
        go.Scatter(x=pos[:, 0], y=vel[:, 0],
                  mode='lines+markers',
                  name='X Phase Space',
                  line=dict(color=body.color, width=2),
                  marker=dict(size=3)),
        row=1, col=1
    )
    
    # Y phase space (y vs vy)
    fig.add_trace(
        go.Scatter(x=pos[:, 1], y=vel[:, 1],
                  mode='lines+markers',
                  name='Y Phase Space',
                  line=dict(color=body.color, width=2),
                  marker=dict(size=3)),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Position X (km)", row=1, col=1)
    fig.update_xaxes(title_text="Position Y (km)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity X (km/s)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity Y (km/s)", row=1, col=2)
    
    fig.update_layout(
        title=f'Phase Space Analysis - {body.name}',
        showlegend=False,
        width=1000,
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_body_separations(positions: np.ndarray,
                         bodies: List[CelestialBody],
                         times: np.ndarray) -> go.Figure:
    """
    Plot the distances between all pairs of bodies over time.
    
    Args:
        positions (np.ndarray): Position data
        bodies (List[CelestialBody]): List of celestial bodies
        times (np.ndarray): Time array
        
    Returns:
        plotly.graph_objects.Figure: Distance plot
    """
    n_times, n_bodies, _ = positions.shape
    
    fig = go.Figure()
    
    # Calculate distances between all pairs
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            distances = np.sqrt(np.sum((positions[:, i, :] - positions[:, j, :]) ** 2, axis=1))
            
            fig.add_trace(go.Scatter(
                x=times,
                y=distances,
                mode='lines',
                name=f'{bodies[i].name} - {bodies[j].name}',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Inter-body Distances Over Time',
        xaxis_title='Time',
        yaxis_title='Distance (km)',
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

def plot_velocity_vectors(positions: np.ndarray,
                         velocities: np.ndarray,
                         bodies: List[CelestialBody],
                         time_index: int = 0,
                         scale_factor: float = 1000.0) -> go.Figure:
    """
    Plot velocity vectors at a specific time point.
    
    Args:
        positions (np.ndarray): Position data
        velocities (np.ndarray): Velocity data
        bodies (List[CelestialBody]): List of celestial bodies
        time_index (int): Time index to plot
        scale_factor (float): Scale factor for velocity vectors
        
    Returns:
        plotly.graph_objects.Figure: Velocity vector plot
    """
    fig = go.Figure()
    
    # Plot bodies and velocity vectors
    for i, body in enumerate(bodies):
        pos = positions[time_index, i, :]
        vel = velocities[time_index, i, :]
        
        # Body position
        fig.add_trace(go.Scatter(
            x=[pos[0]],
            y=[pos[1]],
            mode='markers',
            marker=dict(
                size=12,
                color=body.color,
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            name=body.name,
            showlegend=True
        ))
        
        # Velocity vector
        fig.add_annotation(
            x=pos[0] + vel[0] * scale_factor,
            y=pos[1] + vel[1] * scale_factor,
            ax=pos[0],
            ay=pos[1],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=body.color,
        )
    
    # Set equal aspect ratio
    all_positions = positions[time_index, :, :]
    margin = 0.2
    x_range = np.ptp(all_positions[:, 0])
    y_range = np.ptp(all_positions[:, 1])
    x_center = np.mean(all_positions[:, 0])
    y_center = np.mean(all_positions[:, 1])
    
    max_range = max(x_range, y_range)
    
    fig.update_layout(
        title=f'Velocity Vectors at Time Index {time_index}',
        xaxis=dict(
            title='Position X (km)',
            range=[x_center - max_range/2 * (1 + margin),
                   x_center + max_range/2 * (1 + margin)],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title='Position Y (km)',
            range=[y_center - max_range/2 * (1 + margin),
                   y_center + max_range/2 * (1 + margin)]
        ),
        showlegend=True,
        width=800,
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_matplotlib_trajectories(positions: np.ndarray,
                                  bodies: List[CelestialBody],
                                  figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None):
    """
    Create matplotlib trajectory plot (for saving as high-quality images).
    
    Args:
        positions (np.ndarray): Position data
        bodies (List[CelestialBody]): List of celestial bodies
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.pyplot.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, body in enumerate(bodies):
        # Trajectory
        ax.plot(positions[:, i, 0], positions[:, i, 1],
               color=body.color, linewidth=2, alpha=0.7,
               label=f'{body.name} Trajectory')
        
        # Start point
        ax.plot(positions[0, i, 0], positions[0, i, 1],
               'o', color=body.color, markersize=8,
               markeredgecolor='black', markeredgewidth=2,
               label=f'{body.name} Start')
        
        # End point
        ax.plot(positions[-1, i, 0], positions[-1, i, 1],
               's', color=body.color, markersize=8,
               markeredgecolor='black', markeredgewidth=2,
               label=f'{body.name} End')
    
    ax.set_xlabel('Position X (km)')
    ax.set_ylabel('Position Y (km)')
    ax.set_title('N-Body Gravitational Simulation - Orbital Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

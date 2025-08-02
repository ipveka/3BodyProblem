"""
Animation Module for N-Body Simulation

This module provides functions to create animated visualizations of N-body
gravitational simulations using matplotlib and plotly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
import warnings

from core.body import CelestialBody

def create_matplotlib_animation(positions: np.ndarray, 
                              bodies: List[CelestialBody],
                              times: np.ndarray,
                              interval: int = 50,
                              trail_length: int = 100,
                              figsize: Tuple[int, int] = (10, 8),
                              save_path: Optional[str] = None) -> animation.FuncAnimation:
    """
    Create an animated matplotlib visualization of the N-body simulation.
    
    Args:
        positions (np.ndarray): Position data of shape (n_times, n_bodies, 2)
        bodies (List[CelestialBody]): List of celestial bodies
        times (np.ndarray): Time array
        interval (int): Animation interval in milliseconds
        trail_length (int): Length of position trails
        figsize (Tuple[int, int]): Figure size
        save_path (str, optional): Path to save animation as gif/mp4
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    n_times, n_bodies, _ = positions.shape
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('N-Body Gravitational Simulation')
    ax.set_xlabel('Position X (km)')
    ax.set_ylabel('Position Y (km)')
    
    # Determine plot limits
    all_positions = positions.reshape(-1, 2)
    margin = 0.1
    x_range = np.ptp(all_positions[:, 0])
    y_range = np.ptp(all_positions[:, 1])
    x_center = np.mean(all_positions[:, 0])
    y_center = np.mean(all_positions[:, 1])
    
    max_range = max(x_range, y_range)
    ax.set_xlim(x_center - max_range/2 * (1 + margin), 
                x_center + max_range/2 * (1 + margin))
    ax.set_ylim(y_center - max_range/2 * (1 + margin), 
                y_center + max_range/2 * (1 + margin))
    
    # Initialize plot elements
    body_plots = []
    trail_plots = []
    
    for i, body in enumerate(bodies):
        # Body marker
        body_plot, = ax.plot([], [], 'o', markersize=8, 
                           color=body.color, label=body.name)
        body_plots.append(body_plot)
        
        # Trail
        trail_plot, = ax.plot([], [], '-', alpha=0.6, 
                            color=body.color, linewidth=1)
        trail_plots.append(trail_plot)
    
    ax.legend(loc='upper right')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top')
    
    def init():
        """Initialize animation."""
        for body_plot, trail_plot in zip(body_plots, trail_plots):
            body_plot.set_data([], [])
            trail_plot.set_data([], [])
        time_text.set_text('')
        return body_plots + trail_plots + [time_text]
    
    def animate(frame):
        """Animation function."""
        # Update body positions
        for i, body_plot in enumerate(body_plots):
            body_plot.set_data([positions[frame, i, 0]], [positions[frame, i, 1]])
        
        # Update trails
        for i, trail_plot in enumerate(trail_plots):
            start_idx = max(0, frame - trail_length)
            trail_x = positions[start_idx:frame+1, i, 0]
            trail_y = positions[start_idx:frame+1, i, 1]
            trail_plot.set_data(trail_x, trail_y)
        
        # Update time
        time_text.set_text(f'Time: {times[frame]:.2f}')
        
        return body_plots + trail_plots + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=n_times, interval=interval,
                                 blit=True, repeat=True)
    
    # Save animation if path provided
    if save_path:
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            else:
                warnings.warn("Unsupported file format. Use .gif or .mp4")
        except Exception as e:
            warnings.warn(f"Could not save animation: {e}")
    
    return anim

def create_animated_plot(positions: np.ndarray,
                        bodies: List[CelestialBody],
                        times: np.ndarray,
                        skip_frames: int = 1,
                        trail_length: int = 50) -> go.Figure:
    """
    Create an animated plotly visualization of the N-body simulation.
    
    Args:
        positions (np.ndarray): Position data of shape (n_times, n_bodies, 2)
        bodies (List[CelestialBody]): List of celestial bodies
        times (np.ndarray): Time array
        skip_frames (int): Skip every N frames for performance
        trail_length (int): Length of position trails
        
    Returns:
        plotly.graph_objects.Figure: Animated plotly figure
    """
    n_times, n_bodies, _ = positions.shape
    
    # Subsample frames for performance
    frame_indices = range(0, n_times, skip_frames)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each body
    for i, body in enumerate(bodies):
        # Current position trace
        fig.add_trace(go.Scatter(
            x=[positions[0, i, 0]],
            y=[positions[0, i, 1]],
            mode='markers',
            marker=dict(
                size=12,
                color=body.color,
                symbol='circle'
            ),
            name=body.name,
            showlegend=True
        ))
        
        # Trail trace
        fig.add_trace(go.Scatter(
            x=[positions[0, i, 0]],
            y=[positions[0, i, 1]],
            mode='lines',
            line=dict(
                color=body.color,
                width=2
            ),
            name=f'{body.name} Trail',
            showlegend=False,
            opacity=0.6
        ))
    
    # Create frames for animation
    frames = []
    
    for frame_idx in frame_indices:
        frame_data = []
        
        for i, body in enumerate(bodies):
            # Current position
            frame_data.append(go.Scatter(
                x=[positions[frame_idx, i, 0]],
                y=[positions[frame_idx, i, 1]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=body.color,
                    symbol='circle'
                ),
                name=body.name
            ))
            
            # Trail
            start_idx = max(0, frame_idx - trail_length)
            trail_x = positions[start_idx:frame_idx+1, i, 0]
            trail_y = positions[start_idx:frame_idx+1, i, 1]
            
            frame_data.append(go.Scatter(
                x=trail_x,
                y=trail_y,
                mode='lines',
                line=dict(
                    color=body.color,
                    width=2
                ),
                name=f'{body.name} Trail',
                opacity=0.6
            ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=f'N-Body Simulation - Time: {times[frame_idx]:.2f}'
            )
        ))
    
    fig.frames = frames
    
    # Set up layout
    all_positions = positions.reshape(-1, 2)
    margin = 0.1
    x_range = np.ptp(all_positions[:, 0])
    y_range = np.ptp(all_positions[:, 1])
    x_center = np.mean(all_positions[:, 0])
    y_center = np.mean(all_positions[:, 1])
    
    max_range = max(x_range, y_range)
    
    fig.update_layout(
        title='N-Body Gravitational Simulation',
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
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[str(frame_idx)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{times[frame_idx]:.1f}',
                    'method': 'animate'
                }
                for frame_idx in frame_indices
            ],
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50},
            'x': 0.1,
            'len': 0.9,
            'y': 0,
            'pad': {'b': 10, 't': 50}
        }]
    )
    
    return fig

def create_3d_animation(positions: np.ndarray,
                       bodies: List[CelestialBody],
                       times: np.ndarray,
                       z_positions: Optional[np.ndarray] = None) -> go.Figure:
    """
    Create a 3D animated visualization (requires 3D position data or simulated Z).
    
    Args:
        positions (np.ndarray): Position data of shape (n_times, n_bodies, 2)
        bodies (List[CelestialBody]): List of celestial bodies
        times (np.ndarray): Time array
        z_positions (np.ndarray, optional): Z-coordinate data
        
    Returns:
        plotly.graph_objects.Figure: 3D animated plotly figure
    """
    n_times, n_bodies, _ = positions.shape
    
    # Generate z-coordinates if not provided (for visualization purposes)
    if z_positions is None:
        z_positions = np.zeros((n_times, n_bodies))
        # Add some variation for visual interest
        for i in range(n_bodies):
            z_positions[:, i] = 0.1 * np.sin(times + i * np.pi / n_bodies)
    
    # Create 3D figure
    fig = go.Figure()
    
    # Add traces for each body
    for i, body in enumerate(bodies):
        # Current position
        fig.add_trace(go.Scatter3d(
            x=[positions[0, i, 0]],
            y=[positions[0, i, 1]],
            z=[z_positions[0, i]],
            mode='markers',
            marker=dict(
                size=8,
                color=body.color,
                symbol='circle'
            ),
            name=body.name,
            showlegend=True
        ))
        
        # Trail
        fig.add_trace(go.Scatter3d(
            x=positions[:, i, 0],
            y=positions[:, i, 1],
            z=z_positions[:, i],
            mode='lines',
            line=dict(
                color=body.color,
                width=4
            ),
            name=f'{body.name} Trail',
            showlegend=False,
            opacity=0.6
        ))
    
    # Set up layout
    fig.update_layout(
        title='3D N-Body Gravitational Simulation',
        scene=dict(
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            zaxis_title='Z Position (km)',
            aspectmode='cube'
        ),
        showlegend=True,
        width=800,
        height=600
    )
    
    return fig

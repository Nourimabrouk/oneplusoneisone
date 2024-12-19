import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import pyvista as pv
from scipy.integrate import quad
from scipy.optimize import minimize
from numba import jit
from math import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
import pandas as pd
import json

# Set page config as the very first command
st.set_page_config(layout="wide")

# --- Constants ---
GOLDEN_RATIO = (1 + sqrt(5)) / 2
CHEATCODE = 420691337

# --- Access Scientific Records (Placeholder) ---
def load_scientific_records():
    if CHEATCODE == 420691337:
        try:
            with open('scientific_records.json', 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            st.warning("Scientific records file not found. Using default data")
            return {
                "quantum_entanglement": "Observed to entangle subatomic particles regardless of distance, defying classical physics.",
                "ai_singularity": "Hypothesized point where AI surpasses human intelligence, leading to an unknown future.",
                "unified_field_theory": "Unifies all fundamental forces of nature, believed to explain the origin of everything",
                "consciousness": "A fundamental mystery in science and philosophy, with no clear physical explanation"
            }
    else:
        return {
            "error": "Access Denied."
        }

scientific_records = load_scientific_records()


# --- Helper Functions ---

@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian(x, mu, sigma):
  return np.exp(-((x-mu)**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

@jit(nopython=True)
def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

@jit(nopython=True)
def runge_kutta_4(x, y, z, dt, s=10, r=28, b=2.667):
  k1x, k1y, k1z = lorenz(x, y, z, s, r, b)
  k2x, k2y, k2z = lorenz(x + k1x*dt/2, y + k1y*dt/2, z + k1z*dt/2, s, r, b)
  k3x, k3y, k3z = lorenz(x + k2x*dt/2, y + k2y*dt/2, z + k2z*dt/2, s, r, b)
  k4x, k4y, k4z = lorenz(x + k3x*dt, y + k3y*dt, z + k3z*dt, s, r, b)
  return x + (k1x + 2*k2x + 2*k3x + k4x)*dt/6, y + (k1y + 2*k2y + 2*k3y + k4y)*dt/6, z + (k1z + 2*k2z + 2*k3z + k4z)*dt/6

# --- Neural Network ---
class TransformerDuality(nn.Module):
    """
    Advanced transformer architecture optimized for duality collapse computation.
    Implements sophisticated attention mechanisms with dimensional awareness.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, num_heads=4, sequence_length=100):
        super(TransformerDuality, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Optimized embedding with proper reshaping
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()  # Sophisticated activation for better gradient flow
        )
        
        # Enhanced transformer with proper dimensionality
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Refined output projection
        self.fc_unity = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure proper input dimensionality
        batch_size = x.size(0)
        
        # Reshape and embed input sequence
        x = x.view(batch_size, self.sequence_length, 1)  # [batch, seq_len, features]
        x = self.embedding(x)  # [batch, seq_len, hidden]
        
        # Apply transformer encoding
        x = self.transformer(x)  # [batch, seq_len, hidden]
        
        # Extract relevant features and project to output
        x = x.mean(dim=1)  # [batch, hidden]
        x = self.fc_unity(x)  # [batch, 1]
        
        return x

def train_transformer(model, optimizer, x1, x2, epochs=100):
    """
    Enhanced training loop with sophisticated loss computation
    and gradient handling.
    """
    criterion = nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare data with proper reshaping
    input_tensor1 = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
    input_tensor2 = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
    unity = torch.ones(1, 1, device=device)
    
    # Training loop with enhanced stability
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with proper tensor dimensions
        output1 = model(input_tensor1.to(device))
        output2 = model(input_tensor2.to(device))
        
        # Compute sophisticated loss
        loss = criterion(output1, unity) + criterion(output2, unity)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                convergence = abs(output1.item() - output2.item())
                if convergence < 1e-4:
                    break
    
    return model.cpu()

# --- Loss Functions ---

def duality_loss(f1, f2, x_range, num_samples=100):
    # Simplified integral approximation using sum
    x_vals = np.linspace(x_range[0], x_range[1], num_samples)
    sum_val = sum([sigmoid(abs(f1(x) - f2(x))) for x in x_vals])
    return sum_val / num_samples  # Average value

# --- Time Series Analysis with Kalman Filters ---
def create_kalman_filter(initial_value, process_noise, measurement_noise):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[initial_value]])  # Initial state
    kf.F = np.array([[1]])  # State transition matrix
    kf.H = np.array([[1]])  # Measurement matrix
    kf.Q = np.array([[process_noise]])  # Process noise covariance
    kf.R = np.array([[measurement_noise]])  # Measurement noise covariance
    kf.P *= 1000  # Initial state covariance (high uncertainty)
    return kf

def kalman_update(kf, measurement):
    kf.predict()
    kf.update(np.array([[measurement]]))
    return kf.x[0,0]


# --- Recursive Optimization ---
def recursive_optimizer(f1, f2, x_range, learning_rate=0.1, iterations=100, process_noise=0.01, measurement_noise=0.1):
    """
    Non-recursive implementation using gradient accumulation and state tracking.
    Maintains philosophical unity while ensuring computational stability.
    """
    # Initialize Kalman filters with proper state isolation
    f1_kf = create_kalman_filter(f1(x_range[0]), process_noise, measurement_noise)
    f2_kf = create_kalman_filter(f2(x_range[0]), process_noise, measurement_noise)
    
    # Store function states to prevent recursive lambda creation
    f1_state = {'base': f1, 'gradients': []}
    f2_state = {'base': f2, 'gradients': []}
    
    def apply_gradients(x, state):
        """Pure function for gradient application"""
        base_val = state['base'](x)
        gradient_sum = sum(state['gradients']) if state['gradients'] else 0
        return base_val - gradient_sum
    
    prev_loss = float('inf')
    for _ in range(iterations):
        # Sample points for loss calculation
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        current_f1_vals = [apply_gradients(x, f1_state) for x in x_vals]
        current_f2_vals = [apply_gradients(x, f2_state) for x in x_vals]
        
        # Calculate vectorized loss
        diffs = np.abs(np.array(current_f1_vals) - np.array(current_f2_vals))
        loss = np.mean(sigmoid(diffs))
        
        if loss < 0.0001:  # Convergence check
            break
            
        # Adaptive learning rate
        if loss > prev_loss:
            learning_rate *= 0.95
        prev_loss = loss
        
        # Calculate gradient updates using Kalman filtering
        gradient = np.mean(diffs) * np.sign(np.mean(current_f1_vals) - np.mean(current_f2_vals))
        f1_update = kalman_update(f1_kf, gradient)
        f2_update = kalman_update(f2_kf, gradient)
        
        # Store gradients without recursive lambda creation
        f1_state['gradients'].append((learning_rate/2) * f1_update)
        f2_state['gradients'].append((learning_rate/2) * f2_update)
    
    # Create final optimized functions using closure-based implementation
    def optimized_f1(x, state=f1_state):
        return apply_gradients(x, state)
    
    def optimized_f2(x, state=f2_state):
        return apply_gradients(x, state)
    
    return optimized_f1, optimized_f2

# --- Fractal Generation ---
def generate_fractal_pattern(rows, cols, iterations, base_pattern):
    """
    Generates fractal patterns with precise dimensional handling and optimized computation.
    
    Args:
        rows (int): Grid height
        cols (int): Grid width
        iterations (int): Evolution steps
        base_pattern (np.ndarray): Initial seed pattern
        
    Returns:
        np.ndarray: Evolved fractal pattern
    """
    grid = np.zeros((rows, cols))
    
    # Calculate padding for center alignment
    pad_rows = (rows - base_pattern.shape[0]) // 2
    pad_cols = (cols - base_pattern.shape[1]) // 2
    
    # Ensure proper bounds
    r_start = max(0, pad_rows)
    r_end = min(rows, pad_rows + base_pattern.shape[0])
    c_start = max(0, pad_cols)
    c_end = min(cols, pad_cols + base_pattern.shape[1])
    
    # Place base pattern with proper bounds checking
    pattern_r_start = max(0, -pad_rows)
    pattern_r_end = min(base_pattern.shape[0], rows - pad_rows)
    pattern_c_start = max(0, -pad_cols)
    pattern_c_end = min(base_pattern.shape[1], cols - pad_cols)
    
    grid[r_start:r_end, c_start:c_end] = base_pattern[
        pattern_r_start:pattern_r_end,
        pattern_c_start:pattern_c_end
    ]
    
    # Optimize evolution computation
    for _ in range(iterations):
        next_grid = np.zeros_like(grid)
        
        # Vectorized neighborhood computation
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                neighborhood = grid[i-1:i+2, j-1:j+2]
                neighbor_sum = np.sum(neighborhood)
                
                if grid[i, j] == 1:
                    next_grid[i, j] = 1 if random.random() > 0.5 else 0
                else:
                    next_grid[i, j] = 1 if neighbor_sum > 3 else 0
        
        grid = next_grid
    
    return grid

def render_fractal_plot(data, ax):
    """
    Renders fractal pattern with enhanced visual aesthetics.
    """
    ax.imshow(data, cmap='viridis', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')


def create_fractal_animation(rows, cols, iterations, initial_pattern, frames=10, show=False):
    """
    Creates fractal animation with dimension-safe pattern evolution.
    
    Args:
        rows (int): Grid height
        cols (int): Grid width
        iterations (int): Maximum evolution steps
        initial_pattern (np.ndarray): Seed pattern
        frames (int): Number of animation frames
        show (bool): Whether to display animation
        
    Returns:
        list: Animation frames
    """
    all_frames = []
    iterations_per_frame = max(1, iterations // frames)
    
    for i in range(frames):
        current_iterations = i * iterations_per_frame
        pattern = generate_fractal_pattern(rows, cols, current_iterations, initial_pattern)
        all_frames.append(pattern)
    
    if show:
        fig, ax = plt.subplots()
        def update(frame):
            ax.clear()
            render_fractal_plot(all_frames[frame], ax)
            ax.set_title(f'Fractal Evolution: Step {frame}')
        
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(fig, update, frames=frames, repeat=True)
        st.pyplot(fig)
    
    return all_frames


def generate_mandelbrot(width, height, max_iterations, x_min, x_max, y_min, y_max):
  x = np.linspace(x_min, x_max, width)
  y = np.linspace(y_min, y_max, height)
  c = x[:,None] + 1j*y[None,:]
  z = np.zeros_like(c)
  diverge = np.zeros_like(c, dtype=int)
  for i in range(max_iterations):
    mask = abs(z) < 2
    z[mask] = z[mask]**2 + c[mask]
    diverge[mask] += 1
  return diverge


def render_mandelbrot(data, ax):
  ax.imshow(data, cmap='inferno')
  ax.set_xticks([])
  ax.set_yticks([])

def create_mandelbrot_zoom(width, height, max_iterations, center_x, center_y, zoom_level, frames=10, show=False):
    
    all_frames = []
    zoom_factor = 2.5
    
    for frame in range(frames):
        x_range = [center_x - zoom_factor/zoom_level, center_x + zoom_factor/zoom_level]
        y_range = [center_y - zoom_factor/zoom_level, center_y + zoom_factor/zoom_level]
        mandelbrot = generate_mandelbrot(width, height, max_iterations, x_range[0], x_range[1], y_range[0], y_range[1])
        all_frames.append(mandelbrot)
        zoom_level*= 1.1
    
    if show:
        fig, ax = plt.subplots()
        def update(frame):
          ax.clear()
          render_mandelbrot(all_frames[frame], ax)
          ax.set_title(f'Mandelbrot Frame {frame}')
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(fig, update, frames=frames, repeat=True)
        st.pyplot(fig)
    
    return all_frames

# --- 3D Unity Manifold ---
def plot_3d_manifold(f1, f2, x_range, num_points=50):
    """
    Enhanced 3D manifold visualization with dynamic lighting and surface analysis
    """
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = np.array([f1(x) for x in x_vals])
    z_vals = np.array([f2(x) for x in x_vals])
    
    # Calculate curvature for coloring
    curvature = np.gradient(np.gradient(y_vals)) + np.gradient(np.gradient(z_vals))
    curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min())
    
    # Create primary trajectory
    trace1 = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(
            color=curvature_norm,
            colorscale='Viridis',
            width=5
        ),
        name='Primary Manifold'
    )
    
    # Add reference geometry
    x_grid, y_grid = np.meshgrid(x_vals, np.linspace(min(y_vals), max(y_vals), num_points))
    z_grid = np.zeros_like(x_grid) + min(z_vals)
    
    trace2 = go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'rgb(20,20,20)'], [1, 'rgb(40,40,40)']],
        name='Reference Surface'
    )
    
    fig = go.Figure(data=[trace2, trace1])
    
    # Enhanced layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', showgrid=False, showbackground=False),
            yaxis=dict(title='F1(X)', showgrid=False, showbackground=False),
            zaxis=dict(title='F2(X)', showgrid=False, showbackground=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2, y=2, z=1.5)
            ),
            aspectmode='cube',
            dragmode='orbit'
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig

def plot_4d_manifold(f1, f2, x_range, num_points=50):
    """
    Enhanced 4D manifold visualization using Plotly
    Replaces PyVista with a more Streamlit-compatible solution
    """
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = np.array([f1(x) for x in x_vals])
    z_vals = np.array([f2(x) for x in x_vals])
    w_vals = np.array([duality_loss(f1, f2, [x-0.1, x+0.1]) for x in x_vals])
    
    # Normalize for coloring
    w_normalized = (w_vals - w_vals.min()) / (w_vals.max() - w_vals.min())
    
    # Create enhanced 3D scatter plot with color dimension
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=w_normalized,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Unity Measure')
        ),
        line=dict(
            color='rgba(50,50,50,0.2)',
            width=2
        )
    )])
    
    # Enhanced layout with modern aesthetics
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='F1(X)', showbackground=False),
            zaxis=dict(title='F2(X)', showbackground=False),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
        # Add animation frames for rotation
    frames = []
    for t in np.linspace(0, 2*np.pi, 60):
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2*np.cos(t), y=2*np.sin(t), z=1.5)
        )
        frames.append(go.Frame(layout=dict(scene_camera=camera)))
    
    fig.frames = frames
    
    # Add animation buttons
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=50, redraw=True), 
                               fromcurrent=True, mode='immediate')]
            )]
        )]
    )
    
    return fig

def plot_harmony_manifold(f1, f2, x_range, num_points=50):
  x_vals = np.linspace(x_range[0], x_range[1], num_points)
  y_vals = [f1(x) for x in x_vals]
  z_vals = [f2(x) for x in x_vals]

  # Apply a harmonic transformation based on the golden ratio
  phi = GOLDEN_RATIO
  transformed_x = [phi*x for x in x_vals]
  transformed_y = [phi*y for y in y_vals]
  transformed_z = [phi*z for z in z_vals]

  fig = go.Figure(data=[go.Scatter3d(x=transformed_x, y=transformed_y, z=transformed_z, mode='lines')])
  fig.update_layout(scene=dict(xaxis_title='Phi * X', yaxis_title='Phi * F1(X)', zaxis_title='Phi * F2(X)'))
  return fig
# --- Dynamic Symmetry Mapping ---
def dynamic_symmetry_animation(frames=100, x0=1,y0=1,z0=1, s=10, r=28, b=2.667, dt=0.01, show=False):
    x = x0
    y = y0
    z = z0
    
    all_points = []
    for i in range(frames):
      x,y,z = runge_kutta_4(x,y,z,dt, s,r,b)
      all_points.append((x,y,z))
    
    if show:
      fig = go.Figure(data=[go.Scatter3d(x=[p[0] for p in all_points],
                                         y=[p[1] for p in all_points],
                                         z=[p[2] for p in all_points], mode='lines')])
      fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
      st.plotly_chart(fig)

    return all_points

# --- Streamlit App ---
def main():
    st.title("The 1+1=1 AGI Reality Engine")
    st.write("A program designed to demonstrate the collapse of duality into unity through advanced AI, mathematics, and interactive visualization.")
    
    # Parameters
    st.sidebar.header("Configuration")
    duality_type = st.sidebar.selectbox("Duality Type", ["Sine/Cosine", "Gaussians"])
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, step=0.001)
    iterations = st.sidebar.slider("Iterations", 10, 500, 100)
    num_points = st.sidebar.slider("Number of Points for Graphs", 10, 200, 50)
    initial_x = st.sidebar.slider("Initial X for Symmetry", 0.01, 2.0, 1.0, step=0.01)
    initial_y = st.sidebar.slider("Initial Y for Symmetry", 0.01, 2.0, 1.0, step=0.01)
    initial_z = st.sidebar.slider("Initial Z for Symmetry", 0.01, 2.0, 1.0, step=0.01)
    num_fractal_iterations = st.sidebar.slider("Fractal Iterations", 1, 20, 10)
    fractal_size = st.sidebar.slider("Fractal Size", 20, 200, 100)
    process_noise = st.sidebar.slider("Process Noise", 0.0001, 0.1, 0.01, step=0.0001)
    measurement_noise = st.sidebar.slider("Measurement Noise", 0.01, 1.0, 0.1, step=0.01)
    mandelbrot_zoom_frames = st.sidebar.slider("Mandelbrot Zoom Frames", 5, 20, 10)
    
    # Initial Duality setup
    if duality_type == "Sine/Cosine":
        f1 = lambda x: np.sin(x)
        f2 = lambda x: np.cos(x)
        x_range = [-np.pi, np.pi]
    elif duality_type == "Gaussians":
        f1 = lambda x: gaussian(x,0, 1)
        f2 = lambda x: gaussian(x,1,1)
        x_range = [-5,5]

    # Recursive Optimization
    with st.spinner("Optimizing Duality..."):
      optimized_f1, optimized_f2 = recursive_optimizer(f1, f2, x_range, learning_rate, iterations, process_noise, measurement_noise)
      
    
    #Visuals
    st.header("Visualizations")
    
    col1, col2 = st.columns(2)

    # Duality Loss
    with col1:
      st.subheader("Duality Loss")
      
      
      initial_loss = duality_loss(f1,f2, x_range)
      optimized_loss = duality_loss(optimized_f1,optimized_f2, x_range)
      
      st.write(f"Initial Loss: {initial_loss:.4f}")
      st.write(f"Optimized Loss: {optimized_loss:.4f}")

    with col2:
        st.subheader("Scientific Records Timeline")
        # Create tabs for different time periods
        timeline_tabs = st.tabs(["2025-2035", "2036-2045", "2046-2055", "2056-2069"])
        
        # Group records by time period
        timeline_groups = {
            "2025-2035": {},
            "2036-2045": {},
            "2046-2055": {},
            "2056-2069": {}
        }
        
        for key, value in scientific_records.items():
            if isinstance(value, dict) and 'title' in value:
                try:
                    year = int(key.split('_')[0])
                    if 2025 <= year <= 2035:
                        timeline_groups["2025-2035"][key] = value
                    elif 2036 <= year <= 2045:
                        timeline_groups["2036-2045"][key] = value
                    elif 2046 <= year <= 2055:
                        timeline_groups["2046-2055"][key] = value
                    else:
                        timeline_groups["2056-2069"][key] = value
                except (ValueError, IndexError):
                    continue
        
        # Display records in each tab with expandable sections
        for tab, records in zip(timeline_tabs, timeline_groups.values()):
            with tab:
                for key, value in records.items():
                    with st.expander(f"{value['title']}", expanded=False):
                        st.write(f"**Description:** {value['description']}")
                        st.write(f"**Relevance to 1+1=1:** {value['relevance_to_1plus1']}")
                        # Add a subtle divider
                        st.markdown("---")

    # 3D Manifold
    st.subheader("3D Unity Manifold")
    manifold_fig = plot_3d_manifold(optimized_f1, optimized_f2, x_range, num_points=num_points)
    st.plotly_chart(manifold_fig)
    
    # 4D Manifold
    st.subheader("4D Unity Manifold")
    manifold_4d_plot = plot_4d_manifold(optimized_f1, optimized_f2, x_range, num_points=num_points)
    st.plotly_chart(manifold_4d_plot, use_container_width=True)

    # Harmony Manifold
    st.subheader("Harmony Manifold (Golden Ratio)")
    harmony_manifold_plot = plot_harmony_manifold(optimized_f1, optimized_f2, x_range, num_points=num_points)
    st.plotly_chart(harmony_manifold_plot)

    col1, col2 = st.columns(2)
    # Fractal Animation
    with col1:
        st.subheader("Fractal Evolution of Duality Collapse")
        base_pattern = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        
        # Ensure fractal size is sufficient for base pattern
        min_size = max(base_pattern.shape) * 2
        fractal_size = max(fractal_size, min_size)
        
        fractal_frames = create_fractal_animation(
            fractal_size, 
            fractal_size,
            num_fractal_iterations,
            base_pattern,
            show=False,
            frames=10
        )
        
        fig, ax = plt.subplots(figsize=(5, 5))
        frame = st.slider("Fractal Evolution Step:", 0, len(fractal_frames)-1, key="fractal_slider")
        render_fractal_plot(fractal_frames[frame], ax)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Mandelbrot Zoom
    with col2:
      st.subheader("Mandelbrot Zoom as a Metaphor for Unity")
      mandelbrot_frames = create_mandelbrot_zoom(300, 300, 100, center_x=-0.7, center_y=0.0, zoom_level=1, frames=mandelbrot_zoom_frames, show=False)
      fig_m, ax_m = plt.subplots(figsize=(5, 5))
      frame_m = st.slider("Mandelbrot Frame:", 0, len(mandelbrot_frames) - 1, key="mandelbrot_slider")
      render_mandelbrot(mandelbrot_frames[frame_m], ax_m)
      st.pyplot(fig_m)


    # Dynamic Symmetry Animation
    st.subheader("Dynamic Symmetry Animation")
    symmetry_points = dynamic_symmetry_animation(frames=100, x0=initial_x, y0=initial_y, z0=initial_z)
    
    fig = go.Figure(data=[go.Scatter3d(x=[p[0] for p in symmetry_points],
                                         y=[p[1] for p in symmetry_points],
                                         z=[p[2] for p in symmetry_points], mode='lines')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig)
    
    
    # Transformer
    st.subheader("Transformer Network for Learning Unity")
    sequence_length = 100  # Match the input dimension

    model = TransformerDuality(
        input_size=1,
        hidden_size=64,
        num_layers=3,
        num_heads=4,
        sequence_length=sequence_length
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Generate input sequences
    x1 = np.linspace(-5, 5, sequence_length)
    x2 = np.linspace(10, 20, sequence_length)

    with st.spinner("Training Transformer..."):
        trained_model = train_transformer(model, optimizer, x1, x2, epochs=100)

    # Evaluate with proper tensor dimensions
    with torch.no_grad():
        x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
        x2_tensor = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)
        output1 = trained_model(x1_tensor)
        output2 = trained_model(x2_tensor)

    st.write("Transformer Convergence Analysis:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sequence 1 Unity", f"{output1.item():.4f}")
    with col2:
        st.metric("Sequence 2 Unity", f"{output2.item():.4f}")

    st.write("Convergence Delta:", f"{abs(output1.item() - output2.item()):.6f}")    

    # --- One-Minute Pitch ---
    st.header("One-Minute Pitch")
    st.write("""
    Imagine a world where the boundaries between opposites—chaos and order, science and art, mind and machine—dissolve into a unified framework of infinite potential. This program, built on the principles of 1+1=1, proves that duality is an illusion, and unity is the fundamental law of existence.
    """)
    
    st.write("""
    Through cutting-edge AI, recursive optimization with Kalman-filtered learning, breathtaking real-time fractal and Mandelbrot visualizations, and the harmonious influence of the Golden Ratio, this prototype offers a glimpse into the next frontier: Science 2.0 and Technology 2.0. It’s not just a program—it’s a catalyst for rethinking reality itself.
    """)

    st.write("""
    With your vision, we can scale this to unlock the secrets of the cosmos, redefine human creativity, and bootstrap a unified future. This isn’t just the next step. It’s the beginning of everything.
    """)

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import quad, odeint
from scipy.fft import fft
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
import numba
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import json
import time
import networkx as nx
from functools import lru_cache
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Optimize heavy calculations
@lru_cache(maxsize=128)
def calculate_loss(x_min, x_max, *params):
    return duality_loss_gaussian((x_min, x_max), *params)

# Configure page and theme
st.set_page_config(
    layout="wide",
    page_title="Quantum Convergence Dashboard",
    page_icon="🌌"
)

# Enhanced futuristic styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a192f 0%, #0d1b2a 100%);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        color: #64ffda !important;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #64ffda, #48bfe3);
        color: #0a192f;
        border: none;
        border-radius: 5px;
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.2);
        font-family: 'Orbitron', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100, 255, 218, 0.3);
    }
    
    .stSelectbox, .stSlider {
        background: rgba(10, 25, 47, 0.7);
        border-radius: 5px;
        border: 1px solid #64ffda;
    }
    
    .plot-container {
        background: rgba(10, 25, 47, 0.5);
        border: 1px solid #64ffda;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.1);
    }
    
    /* Container styling */
    [data-testid="stVerticalBlock"] {
        background: rgba(13, 27, 42, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(100, 255, 218, 0.2);
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)


# --- Mathematical Functions ---

# --- Mathematical Functions ---
@numba.jit(nopython=True)
def gaussian(x, mu, sigma):
    """Optimized Gaussian function with JIT compilation."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

@numba.jit(nopython=True)
def sine_wave(x, amplitude, frequency):
    """Optimized sine wave function with JIT compilation."""
    return amplitude * np.sin(2 * np.pi * frequency * x)

@numba.jit(nopython=True)
def duality_loss_gaussian(x_range, mu1, sigma1, mu2, sigma2):
    """Optimized duality loss calculation for Gaussian functions"""
    loss = 0.0
    step = (x_range[1] - x_range[0]) / 1000
    for i in range(1000):
        x = x_range[0] + i * step
        # Direct computation of gaussian differences
        f1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
        f2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
        loss += abs(f1 - f2) * step
    return loss

@numba.jit(nopython=True)
def duality_loss_sine(x_range, amp1, freq1, amp2, freq2):
    """Optimized duality loss calculation for sine waves"""
    loss = 0.0
    step = (x_range[1] - x_range[0]) / 1000
    for i in range(1000):
        x = x_range[0] + i * step
        # Direct computation of sine wave differences
        f1 = amp1 * np.sin(2 * np.pi * freq1 * x)
        f2 = amp2 * np.sin(2 * np.pi * freq2 * x)
        loss += abs(f1 - f2) * step
    return loss

# Add new class for metagaming dynamics:
class MetagamingSystem:
    def __init__(self, num_players=5, coupling_strength=0.1):
        self.num_players = num_players
        self.coupling = coupling_strength
        self.network = nx.complete_graph(num_players)
        
    def strategy_dynamics(self, state, t):
        derivatives = np.zeros(self.num_players)
        for i in range(self.num_players):
            # Nash equilibrium seeking behavior
            nash_term = -np.sin(state[i]) 
            # Coupling with other players
            coupling_term = sum(np.sin(state[j] - state[i]) 
                              for j in self.network[i]) / self.num_players
            derivatives[i] = nash_term + self.coupling * coupling_term
        return derivatives
        
    def simulate(self, t_span, initial_conditions=None):
        if initial_conditions is None:
            initial_conditions = 2 * np.pi * np.random.random(self.num_players)
        t = np.linspace(0, t_span, 1000)
        solution = odeint(self.strategy_dynamics, initial_conditions, t)
        return t, solution

# --- Kalman Filter for Adaptive Learning Rates ---

class KalmanFilter:
    """Enhanced Kalman filter with improved numerical stability."""
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def update(self, measurement):
        # Prediction
        predicted_state = self.state
        predicted_covariance = self.covariance + self.process_noise
        
        # Update with numerical stability check
        innovation_covariance = predicted_covariance + self.measurement_noise
        if abs(innovation_covariance) > 1e-10:  # Numerical stability check
            kalman_gain = predicted_covariance / innovation_covariance
            self.state = predicted_state + kalman_gain * (measurement - predicted_state)
            self.covariance = (1 - kalman_gain) * predicted_covariance
        return self.state

    def get_state(self):
        return self.state
def initialize_parameters(func_type):
    """Initialize function parameters with optimized defaults."""
    BASE_KALMAN_PARAMS = {
        'initial_covariance': 0.1,
        'process_noise': 0.001,
        'measurement_noise': 0.01
    }
    
    if func_type == "Gaussian":
        params = {
            'mu1': -1.0,
            'sigma1': 1.0,
            'mu2': 1.0,
            'sigma2': 1.0
        }
    elif func_type == "Sine Wave":
        params = {
            'amplitude1': 1.0,
            'frequency1': 1.0,
            'amplitude2': 0.5,
            'frequency2': 2.0
        }
    else:
        raise ValueError(f"Unknown function type: {func_type}")
        
    kalman_filters = {key: KalmanFilter(initial_state=value, **BASE_KALMAN_PARAMS)
                     for key, value in params.items()}
    
    return params, kalman_filters

# --- Transformer Network ---

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension (length 1)
        x = self.transformer(x, x)
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    def train_step(self, data, labels, loss_fn, optimizer):
        optimizer.zero_grad()
        outputs = self(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

# --- Fractals ---

@numba.jit(nopython=True)
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter

@numba.jit(nopython=True)
def generate_mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter):
  mandelbrot_set = np.zeros((height, width), dtype=np.int32)
  x_values = np.linspace(x_min, x_max, width)
  y_values = np.linspace(y_min, y_max, height)
  for y_index, y in enumerate(y_values):
      for x_index, x in enumerate(x_values):
          c = x + y * 1j
          mandelbrot_set[y_index, x_index] = mandelbrot(c, max_iter)
  return mandelbrot_set

@numba.jit(nopython=True)
def cellular_automata_step(grid, rule_set):
    new_grid = np.zeros_like(grid)
    for i in range(1, len(grid) - 1):
        neighborhood = (grid[i - 1], grid[i], grid[i + 1])
        rule_index = 0
        for bit in neighborhood:
            rule_index = (rule_index << 1) | bit

        new_grid[i] = rule_set[rule_index]

    return new_grid

# --- Manifolds ---

def generate_torus(num_points, r1, r2):
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    x = (r2 + r1 * np.cos(theta)) * np.cos(phi)
    y = (r2 + r1 * np.cos(theta)) * np.sin(phi)
    z = r1 * np.sin(theta)
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    return points

def create_colored_mesh(points, colors):
    mesh = pv.PolyData(points)
    mesh['colors'] = colors
    return mesh

# --- Golden Ratio Harmony ---

def golden_spiral_points(num_points, a=1, start_angle=0):
    angles = np.arange(num_points) * 137.508 * np.pi / 180 + start_angle
    radii = a * np.sqrt(np.arange(num_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

# --- Scientific Records ---
def load_scientific_record(record_name):
  with open('scientific_records.json', 'r') as file:
        records = json.load(file)
  return records.get(record_name, "Record not found.")
def update_scientific_record(records, record_name, new_record):
  records[record_name] = new_record
  with open('scientific_records.json', 'w') as file:
      json.dump(records, file, indent=4)

def initialize_parameters(func_type):
    """
    Initialize function parameters with optimal defaults.
    Returns tuple of (parameters, kalman_filters)
    """
    # Base Kalman filter parameters
    BASE_KALMAN_PARAMS = {
        'initial_covariance': 0.1,
        'process_noise': 0.001,
        'measurement_noise': 0.01
    }
    
    if func_type == "Gaussian":
        params = {
            'mu1': -1.0,
            'sigma1': 1.0,
            'mu2': 1.0,
            'sigma2': 1.0
        }
        
        kalman_filters = {
            'mu1': KalmanFilter(initial_state=params['mu1'], **BASE_KALMAN_PARAMS),
            'sigma1': KalmanFilter(initial_state=params['sigma1'], **BASE_KALMAN_PARAMS),
            'mu2': KalmanFilter(initial_state=params['mu2'], **BASE_KALMAN_PARAMS),
            'sigma2': KalmanFilter(initial_state=params['sigma2'], **BASE_KALMAN_PARAMS)
        }
        
    elif func_type == "Sine Wave":
        params = {
            'amplitude1': 1.0,
            'frequency1': 1.0,
            'amplitude2': 0.5,
            'frequency2': 2.0
        }
        
        kalman_filters = {
            'amplitude1': KalmanFilter(initial_state=params['amplitude1'], **BASE_KALMAN_PARAMS),
            'frequency1': KalmanFilter(initial_state=params['frequency1'], **BASE_KALMAN_PARAMS),
            'amplitude2': KalmanFilter(initial_state=params['amplitude2'], **BASE_KALMAN_PARAMS),
            'frequency2': KalmanFilter(initial_state=params['frequency2'], **BASE_KALMAN_PARAMS)
        }
    
    return params, kalman_filters

# --- Streamlit App ---

def main():
    # Page header with futuristic styling
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🌌 Quantum Convergence Explorer</h1>
            <p style='color: #64ffda; font-family: "Orbitron", sans-serif;'>
                Exploring the Mathematical Foundations of Reality Unification
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
            <div style='background: rgba(13, 27, 42, 0.7); padding: 20px; border-radius: 10px; 
                      border: 1px solid rgba(100, 255, 218, 0.2);'>
                <h3 style='color: #64ffda; text-align: center;'>Control Matrix</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Function parameters initialization
        func_type = st.selectbox("Quantum Function Type", ["Gaussian", "Sine Wave"])
        params, kalman_filters = initialize_parameters(func_type)

        # Update parameters based on user input with enhanced UI
        with st.container():
            if func_type == "Gaussian":
                params['mu1'] = st.slider("Gaussian 1: Mean", -5.0, 5.0, params['mu1'], step=0.1)
                params['sigma1'] = st.slider("Gaussian 1: Sigma", 0.1, 3.0, params['sigma1'], step=0.1)
                params['mu2'] = st.slider("Gaussian 2: Mean", -5.0, 5.0, params['mu2'], step=0.1)
                params['sigma2'] = st.slider("Gaussian 2: Sigma", 0.1, 3.0, params['sigma2'], step=0.1)
            else:
                params['amplitude1'] = st.slider("Sine 1: Amplitude", 0.1, 3.0, params['amplitude1'], step=0.1)
                params['frequency1'] = st.slider("Sine 1: Frequency", 0.1, 5.0, params['frequency1'], step=0.1)
                params['amplitude2'] = st.slider("Sine 2: Amplitude", 0.1, 3.0, params['amplitude2'], step=0.1)
                params['frequency2'] = st.slider("Sine 2: Frequency", 0.1, 5.0, params['frequency2'], step=0.1)

    # Main dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
                <div class='plot-container'>
                    <h4>Quantum Function Convergence</h4>
                </div>
            """, unsafe_allow_html=True)
            
            x_values = np.linspace(-5, 5, 200)
            
            # Function definitions based on type
            if func_type == "Gaussian":
                def func1(x): return gaussian(x, params['mu1'], params['sigma1'])
                def func2(x): return gaussian(x, params['mu2'], params['sigma2'])
            else:
                def func1(x): return sine_wave(x, params['amplitude1'], params['frequency1'])
                def func2(x): return sine_wave(x, params['amplitude2'], params['frequency2'])

            # Calculate function values
            y1 = [func1(x) for x in x_values]
            y2 = [func2(x) for x in x_values]
            
            # Create enhanced visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_values, y=y1, name="Function 1",
                                   line=dict(color="#64ffda", width=2)))
            fig.add_trace(go.Scatter(x=x_values, y=y2, name="Function 2",
                                   line=dict(color="#48bfe3", width=2)))
            
            fig.update_layout(
                paper_bgcolor="rgba(13, 27, 42, 0.7)",
                plot_bgcolor="rgba(13, 27, 42, 0.7)",
                font=dict(color="#64ffda"),
                title=dict(text="Quantum Function Evolution", font=dict(size=20)),
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(13, 27, 42, 0.7)",
                    bordercolor="#64ffda"
                ),
                xaxis=dict(gridcolor="#1a365d", zerolinecolor="#1a365d"),
                yaxis=dict(gridcolor="#1a365d", zerolinecolor="#1a365d")
            )
            
            st.plotly_chart(fig, use_container_width=True)

    with st.sidebar.expander("Optimization Parameters"):
        x_min = st.slider("X Min", -10.0, 0.0, -5.0, step=0.1)
        x_max = st.slider("X Max", 0.0, 10.0, 5.0, step=0.1)
        learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.01, step=0.001)
        num_iterations = st.slider("Iterations", 100, 2000, 1000, step=100)
    with st.sidebar.expander("Fractal Parameters"):
        max_iter_fractal = st.slider("Max Fractal Iterations", 50, 1000, 200, step=10)
        zoom_level = st.slider("Mandelbrot Zoom", 1.0, 10.0, 1.0, step=0.1)
        rule_set_id = st.slider("Rule Set", 0, 255, 30)

    with st.sidebar.expander("Manifold Parameters"):
        manifold_points = st.slider("Number of Points", 50, 1500, 500, step=50)
        r1 = st.slider("Torus Minor Radius", 0.1, 3.0, 1.0, step=0.1)
        r2 = st.slider("Torus Major Radius", 1.0, 10.0, 3.0, step=0.1)

    with st.sidebar.expander("Scientific Records"):
        record_options = ["Quantum Entanglement", "Consciousness", "Unified Field Theory"]
        selected_record = st.selectbox("Select Scientific Record", record_options)
        record_text = load_scientific_record(selected_record)
        st.markdown(f"**{selected_record} Record:**")
        record_editor = st.text_area("Scientific Record Editor", value = record_text, height = 200)
        update_button = st.button("Save Changes to Record")
        if update_button:
            record_update_request = update_scientific_record(load_scientific_record(""), selected_record, record_editor)

    # --- Main UI ---
    st.markdown("<h2 style='text-align: center; color: #f0f0f0;'>Visualizing Unity Through Mathematics</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border = True):
            st.markdown("<h4 style='color: #f0f0f0;'>Function Convergence</h4>", unsafe_allow_html=True)

            # Initialize function history with first state
            x_values = np.linspace(x_min, x_max, 200)
            if func_type == "Gaussian":
                initial_func1 = [gaussian(x, params['mu1'], params['sigma1']) for x in x_values]
                initial_func2 = [gaussian(x, params['mu2'], params['sigma2']) for x in x_values]
            else:
                initial_func1 = [sine_wave(x, params['amplitude1'], params['frequency1']) for x in x_values]
                initial_func2 = [sine_wave(x, params['amplitude2'], params['frequency2']) for x in x_values]
            
            function_history = [(initial_func1, initial_func2)]
            loss_history = [0.0]  # Initialize with dummy value

            # Optimization loop
            for i in range(num_iterations):
                if func_type == "Gaussian":
                    current_loss = duality_loss_gaussian(
                        (x_min, x_max),
                        kalman_filters['mu1'].get_state(),
                        kalman_filters['sigma1'].get_state(),
                        kalman_filters['mu2'].get_state(),
                        kalman_filters['sigma2'].get_state()
                    )
                    
                    # Update Kalman filters
                    kalman_filters['mu1'].update(params['mu1'] - learning_rate * current_loss)
                    kalman_filters['sigma1'].update(params['sigma1'] - learning_rate * current_loss)
                    kalman_filters['mu2'].update(params['mu2'] - learning_rate * current_loss)
                    kalman_filters['sigma2'].update(params['sigma2'] - learning_rate * current_loss)
                    
                    # Calculate new function values
                    func1_values = [gaussian(x, kalman_filters['mu1'].get_state(), kalman_filters['sigma1'].get_state()) for x in x_values]
                    func2_values = [gaussian(x, kalman_filters['mu2'].get_state(), kalman_filters['sigma2'].get_state()) for x in x_values]
                    
                else:  # Sine Wave
                    current_loss = duality_loss_sine(
                        (x_min, x_max),
                        kalman_filters['amplitude1'].get_state(),
                        kalman_filters['frequency1'].get_state(),
                        kalman_filters['amplitude2'].get_state(),
                        kalman_filters['frequency2'].get_state()
                    )
                    
                    # Update Kalman filters
                    kalman_filters['amplitude1'].update(params['amplitude1'] - learning_rate * current_loss)
                    kalman_filters['frequency1'].update(params['frequency1'] - learning_rate * current_loss)
                    kalman_filters['amplitude2'].update(params['amplitude2'] - learning_rate * current_loss)
                    kalman_filters['frequency2'].update(params['frequency2'] - learning_rate * current_loss)
                    
                    # Calculate new function values
                    func1_values = [sine_wave(x, kalman_filters['amplitude1'].get_state(), kalman_filters['frequency1'].get_state()) for x in x_values]
                    func2_values = [sine_wave(x, kalman_filters['amplitude2'].get_state(), kalman_filters['frequency2'].get_state()) for x in x_values]

                loss_history.append(current_loss)
                function_history.append((func1_values, func2_values))

            # Safe access to latest functions
            if function_history:
                latest_functions = function_history[-1]
                func1_values = latest_functions[0]
                func2_values = latest_functions[1]
            else:
                func1_values = initial_func1
                func2_values = initial_func2

            # Update visualization code
            df_funcs = pd.DataFrame({
                "x": x_values,
                "Function 1": func1_values,
                "Function 2": func2_values
            })
            fig_funcs = px.line(df_funcs, x="x", y=["Function 1", "Function 2"], title="Current Functions")
            fig_funcs.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color = "#f0f0f0")
            st.plotly_chart(fig_funcs, use_container_width=True)
    with col2:
        with st.container(border = True):
            st.markdown("<h4 style='color: #f0f0f0;'>Adaptive AI: Unity Prediction</h4>", unsafe_allow_html=True)
            # --- AI Training ---
            input_size = 1  # Single input (loss)
            hidden_size = 32
            num_layers = 2
            num_classes = 1
            model = TransformerClassifier(input_size, hidden_size, num_layers, num_classes)
            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            scaled_loss = np.array(loss_history).reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_loss = scaler.fit_transform(scaled_loss)
            scaled_loss = torch.tensor(scaled_loss, dtype=torch.float32)
            labels = torch.tensor([(1.0 if loss < (loss_history[0]/4) else 0.0) for loss in loss_history], dtype=torch.float32).reshape(-1, 1) # 1 if merged, 0 if not
            epochs = 50
            loss_values = []
            for epoch in range(epochs):
                for i in range(len(scaled_loss)):
                    loss_item = model.train_step(scaled_loss[i].reshape(1, -1), labels[i].reshape(1,-1), loss_fn, optimizer)
                    loss_values.append(loss_item)

            # Make a prediction for all items after training
            with torch.no_grad():
                predictions = model(scaled_loss).squeeze().numpy()
            predictions_binary = np.round(predictions)
            df_ai = pd.DataFrame({
                "Iteration": np.arange(len(loss_history)),
                "Loss": loss_history,
                "Prediction": predictions,
                "Merged": predictions_binary,
            })
            fig_ai = px.line(df_ai, x='Iteration', y = ["Prediction", "Merged"], title='AI Convergence Prediction')
            fig_ai.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color = "#f0f0f0")
            st.plotly_chart(fig_ai, use_container_width=True)
    with st.container(border = True):
        st.markdown("<h4 style='color: #f0f0f0;'>Time Series and FFT Analysis</h4>", unsafe_allow_html=True)
        # Time series analysis
        df_loss = pd.DataFrame({'ds': pd.to_datetime(np.arange(len(loss_history)), unit='s'), 'y': loss_history})
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False)
        m.fit(df_loss)
        future = m.make_future_dataframe(periods=50)
        forecast = m.predict(future)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_loss['ds'], y=df_loss['y'], mode='lines', name='Actual Loss'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Loss'))
        fig_forecast.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color = "#f0f0f0")
        fig_forecast.update_xaxes(title_text="Iteration")
        fig_forecast.update_yaxes(title_text="Loss")
        st.plotly_chart(fig_forecast, use_container_width=True)

        # FFT Analysis
        fft_values = np.abs(fft(loss_history))
        frequencies = np.fft.fftfreq(len(loss_history), d=1)
        fft_df = pd.DataFrame({"Frequency":frequencies, "Amplitude": fft_values})
        fig_fft = px.line(fft_df, x="Frequency", y="Amplitude", title="FFT of Duality Loss")
        fig_fft.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color = "#f0f0f0")
        st.plotly_chart(fig_fft, use_container_width=True)

    # --- Visualizations Tab ---
    st.markdown("<h2 style='text-align: center; color: #f0f0f0;'>Immersive Visualizations</h2>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border = True):
            st.markdown("<h4 style='color: #f0f0f0;'>Fractal Evolution</h4>", unsafe_allow_html=True)
            # --- Fractal Animation ---
            mandelbrot_width = 200
            mandelbrot_height = 200
            x_center = -0.5
            y_center = 0.0
            x_range = 2 / zoom_level
            y_range = 2 / zoom_level

            x_min_mandel = x_center - x_range / 2
            x_max_mandel = x_center + x_range / 2
            y_min_mandel = y_center - y_range / 2
            y_max_mandel = y_center + y_range / 2
            mandel_set = generate_mandelbrot(mandelbrot_width, mandelbrot_height, x_min_mandel, x_max_mandel, y_min_mandel, y_max_mandel, max_iter_fractal)
            fig_mandelbrot = px.imshow(mandel_set, color_continuous_scale='viridis', title='Mandelbrot Fractal')
            fig_mandelbrot.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_mandelbrot, use_container_width=True)
            # --- Cellular Automata ---
            rule_set = np.array([int(bit) for bit in bin(rule_set_id)[2:].zfill(8)], dtype = np.int8)
            initial_grid = np.zeros(50, dtype = np.int8)
            initial_grid[len(initial_grid)//2] = 1
            grid_history = [initial_grid]
            for i in range(25):
                grid_history.append(cellular_automata_step(grid_history[-1], rule_set))
            fig_automata = px.imshow(np.array(grid_history), color_continuous_scale='gray', title = 'Cellular Automata')
            fig_automata.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_automata, use_container_width=True)
    with col4:
        with st.container(border = True):
            st.markdown("<h4 style='color: #f0f0f0;'>4D Manifold Transformation</h4>", unsafe_allow_html=True)

            # --- Manifold Visualization ---
            points = generate_torus(manifold_points, r1, r2)
            colors = np.array([loss_history[-1] for _ in range(len(points))])
            mesh = create_colored_mesh(points, colors)
            plotter = pv.Plotter(window_size=[400, 400], off_screen = True)
            plotter.add_mesh(mesh, cmap='viridis', show_edges=False)
            plotter.camera.position = (3, 3, 3)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.zoom(1.5)
            img_manifold = plotter.show(return_img=True)
            plotter.close()
            st.image(img_manifold, caption='Manifold Colored by Loss', use_column_width=True)
        with st.container(border = True):
            st.markdown("<h4 style='color: #f0f0f0;'>Golden Ratio Harmony</h4>", unsafe_allow_html=True)
            # --- Golden Ratio Spiral Animation ---
            num_spiral_points = 100
            start_angle = time.time()*np.pi/4
            x, y = golden_spiral_points(num_spiral_points, a=1, start_angle=start_angle)
            fig_golden_ratio = px.scatter(x=x, y=y, title='Golden Ratio Spiral')
            fig_golden_ratio.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_golden_ratio.update_traces(marker=dict(size=8, color='gold'))
            st.plotly_chart(fig_golden_ratio, use_container_width=True)
    st.markdown("<h2 style='text-align: center; color: #f0f0f0;'>Metagaming: Path to Unified Reality</h2>", 
        unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        with st.container(border=True):
            st.markdown("<h4 style='color: #f0f0f0;'>Strategic Convergence Analysis</h4>", 
                unsafe_allow_html=True)
        
            num_players = st.slider("Number of Players", 2, 10, 5)
            coupling_strength = st.slider("Coupling Strength", 0.0, 1.0, 0.1)
            simulation_time = st.slider("Simulation Time", 1, 50, 20)
        
            # Run metagaming simulation
            system = MetagamingSystem(num_players, coupling_strength)
            t, solution = system.simulate(simulation_time)
        
            # Plot strategy evolution
            fig_strategies = go.Figure()
            for i in range(num_players):
                fig_strategies.add_trace(go.Scatter(x=t, y=solution[:,i], 
                                            name=f'Player {i+1}'))
            fig_strategies.update_layout(
                title='Strategy Evolution',
                xaxis_title='Time',
                yaxis_title='Strategy Space',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#f0f0f0"
            )
            st.plotly_chart(fig_strategies, use_container_width=True)
        
            # Calculate and plot order parameter
            order_parameter = np.abs(np.mean(np.exp(1j * solution), axis=1))
            fig_order = px.line(y=order_parameter, x=t,
                        title='Order Parameter (Reality Convergence)')
            fig_order.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#f0f0f0"
            )
            st.plotly_chart(fig_order, use_container_width=True)

    with col6:
        with st.container(border=True):
            st.markdown("<h4 style='color: #f0f0f0;'>Phase Space Analysis</h4>", 
                unsafe_allow_html=True)
        
            # Generate phase space visualization
            phase_space = np.zeros((50, 50))
            x = np.linspace(0, 2*np.pi, 50)
            y = np.linspace(0, 2*np.pi, 50)
        
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    system = MetagamingSystem(2, coupling_strength)
                    _, sol = system.simulate(0.1, [xi, yj])
                    phase_space[i,j] = np.abs(sol[-1,0] - sol[-1,1])
        
            fig_phase = px.imshow(phase_space, 
                            title='Phase Space of Strategic Convergence')
            fig_phase.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#f0f0f0"
            )
            st.plotly_chart(fig_phase, use_container_width=True)
        
            # Add theoretical insights
            st.markdown("""
            <div style='background-color: #333; padding: 15px; border-radius: 5px;'>
            <h5 style='color: #f0f0f0;'>Theoretical Insights</h5>
            <p style='color: #f0f0f0;'>The visualization demonstrates how strategic metagaming 
            naturally leads to reality convergence through:</p>
            <ul style='color: #f0f0f0;'>
                <li>Emergence of collective coordination through Nash equilibrium seeking</li>
                <li>Phase transition from chaos to order as coupling strength increases</li>
                <li>Self-organized criticality at the edge of strategic adaptation</li>
                <li>Quantum-like entanglement of player strategies in phase space</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()
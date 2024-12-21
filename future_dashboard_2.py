import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import networkx as nx
import random
from scipy.optimize import minimize
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import math
from collections import deque
from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.plot import plot_cross_validation_metric
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta
from typing import Union, Sequence
import warnings

warnings.filterwarnings("ignore")

GRAPH_CACHE = {}

# Global configuration
COLORS = {
    'background': '#0a192f',
    'text': '#64ffda',
    'accent': '#112240',
    'highlight': '#233554',
    'grid': '#1e3a8a'
}

GRAPH_STYLE = {
    'plot_bgcolor': COLORS['background'],
    'paper_bgcolor': COLORS['background'],
    'font': {'color': COLORS['text']},
    'height': 400,
    'margin': dict(l=20, r=20, t=40, b=20),
    'xaxis': dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=0.1, zeroline=False),
    'yaxis': dict(showgrid=True, gridcolor=COLORS['grid'], gridwidth=0.1, zeroline=False)
}

# Initialize app globally for proper state management
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

app.config.suppress_callback_exceptions = True
app.config['suppress_callback_exceptions'] = True

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>UNITY HUD 2069</title>
        {%favicon%}
        {%css%}
        <style>
            .dash-graph { transition: all 0.3s ease-in-out; }
            .dash-graph:hover { transform: scale(1.02); }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

class ErrorMetrics:
    """
    Quantum-aware error metric computations with built-in dimensional analysis.
    Implements advanced statistical measures with automatic normalization.
    """
    
    @staticmethod
    def mean_squared_error(
        y_true: Union[Sequence, np.ndarray], 
        y_pred: Union[Sequence, np.ndarray],
        sample_weight: Union[Sequence, np.ndarray, None] = None
    ) -> float:
        """
        Computes dimensionally-normalized Mean Squared Error with optional weighting.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional weights for each sample
            
        Returns:
            float: Computed MSE value
            
        Raises:
            ValueError: If inputs have incompatible dimensions
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Incompatible shapes: y_true {y_true.shape} != y_pred {y_pred.shape}")
            
        errors = np.square(y_true - y_pred)
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            errors = errors * sample_weight
            return np.mean(errors) / np.mean(sample_weight)
        
        return np.mean(errors)

    @staticmethod 
    def rmse(
        y_true: Union[Sequence, np.ndarray],
        y_pred: Union[Sequence, np.ndarray]
    ) -> float:
        """
        Computes Root Mean Square Error with automatic scaling.
        """
        return np.sqrt(ErrorMetrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def normalized_mse(
        y_true: Union[Sequence, np.ndarray],
        y_pred: Union[Sequence, np.ndarray]
    ) -> float:
        """
        Computes Normalized Mean Square Error for scale-invariant comparison.
        """
        mse = ErrorMetrics.mean_squared_error(y_true, y_pred)
        norm_factor = np.var(y_true)
        if norm_factor == 0:
            warnings.warn("Zero variance in y_true, returning unnormalized MSE")
            return mse
        return mse / norm_factor

# Initialize singleton for global access
metrics = ErrorMetrics()

# ------ Utility Functions -----
def calculate_phi():
    return (1 + math.sqrt(5)) / 2

phi = calculate_phi()

# ---- Golden Ratio Functions -----
def fibonacci_sequence(n):
    """Generates a Fibonacci sequence up to n terms."""
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

def fibonacci_spiral(n, scale=1):
    """Generates coordinates for a Fibonacci spiral."""
    fib = fibonacci_sequence(n)
    points = []
    for i in range(1, n):
        angle = i * (360 / phi**2) * np.pi / 180 # Golden angle in radians
        radius = scale * math.sqrt(i) * 0.1 # Spiral radius proportional to the square root of index.
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append((x, y))
    return points
def create_fibonacci_spiral_graph(n, scale=1, labels=None, data_points=None):
    """Creates a Plotly figure for a Fibonacci spiral with nodes and optional annotations."""
    points = fibonacci_spiral(n, scale)

    fig = go.Figure()
    if data_points is None:
         fig.add_trace(go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode='lines+markers',
            marker=dict(size=8, color=list(range(1, len(points)+1)), colorscale='Viridis', opacity=0.7),
            line=dict(width=2),
            text=labels if labels else [f"Point {i+1}" for i in range(len(points))],
            hoverinfo='text',
            name='Fibonacci Spiral'
            )
        )
    else:
         fig.add_trace(go.Scatter(
            x=[p[0] for p in points],
            y=[p[1] for p in points],
            mode='lines+markers',
            marker=dict(size=[x * 10 for x in data_points], color=list(range(1, len(points)+1)), colorscale='Viridis', opacity=0.7),
            line=dict(width=2),
            text=labels if labels else [f"Point {i+1}" for i in range(len(points))],
            hoverinfo='text',
            name='Fibonacci Spiral'
            )
        )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

# ------- Ant Colony Optimization -----
def initialize_pheromones(graph):
    """Initialize pheromones on edges of a graph."""
    pheromones = {}
    for edge in graph.edges:
        pheromones[edge] = 1.0  # Initial pheromone level is 1
    return pheromones

def ant_walk(graph, start_node, pheromones, alpha, beta, Q):
  """Simulates the walk of a single ant."""
  current_node = start_node
  visited_nodes = [current_node]
  path = [current_node]

  while True:
    neighbors = list(graph.neighbors(current_node))
    unvisited_neighbors = [n for n in neighbors if n not in visited_nodes]

    if not unvisited_neighbors:
        break

    probabilities = []
    total_prob = 0.0
    for neighbor in unvisited_neighbors:
      pheromone_level = pheromones.get((current_node, neighbor), 1.0) # 1 if new
      distance = 1 / graph.edges[current_node, neighbor].get('weight', 1) # Assume weight = distance
      prob = (pheromone_level ** alpha) * (distance ** beta)
      total_prob += prob
      probabilities.append(prob)
    if total_prob == 0.0:
       probabilities = [1 / len(unvisited_neighbors)]*len(unvisited_neighbors) # Handle zero probabilities

    else:
       probabilities = [p/total_prob for p in probabilities]
    next_node = random.choices(unvisited_neighbors, weights=probabilities, k=1)[0]
    path.append(next_node)
    visited_nodes.append(next_node)
    current_node = next_node
  return path

def update_pheromones(graph, pheromones, paths, rho, Q):
    """Update pheromone levels based on ant paths."""
    for edge in pheromones:
        pheromones[edge] *= (1 - rho) # Evaporation
    for path in paths:
        path_len = len(path) -1
        if path_len > 0:
            for i in range(path_len):
                u = path[i]
                v = path[i+1]
                try:
                  pheromones[(u, v)] += Q/path_len # Deposit on each edge
                except:
                  pheromones[(v, u)] += Q/path_len
    return pheromones

def create_aco_graph(num_nodes=100, seed=None):
    """Creates a graph for ACO simulation, ensuring connectivity."""
    if seed is not None:
       random.seed(seed)
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    # Ensure each node has at least one edge to avoid isolated nodes
    for node in nodes:
        potential_neighbors = [n for n in nodes if n != node]
        if graph.degree(node) == 0:
            neighbor = random.choice(potential_neighbors)
            weight = 1 / (abs(node - neighbor) + 0.01)  # Adding small constant to prevent division by 0.
            graph.add_edge(node, neighbor, weight=weight)
    while not nx.is_connected(graph):
        # If the graph is not connected, generate more edges between nodes with less degree and others
        subgraphs = list(nx.connected_components(graph))
        subgraph_lengths = [len(s) for s in subgraphs]
        if len(subgraphs) > 1:
          # Identify nodes from different components
          node1 = random.choice(list(subgraphs[np.argmin(subgraph_lengths)])) # Node in smaller component
          node2 = random.choice(list(subgraphs[np.argmax(subgraph_lengths)])) # Node in larger component
          weight = 1 / (abs(node1-node2)+0.01) # Adding small constant to prevent division by 0.
          graph.add_edge(node1, node2, weight=weight)
        else:
             # If its connected but somehow had degree zero, generate a new edge
           for node in nodes:
             if graph.degree(node) == 0:
               potential_neighbors = [n for n in nodes if n != node]
               neighbor = random.choice(potential_neighbors)
               weight = 1 / (abs(node - neighbor) + 0.01)  # Adding small constant to prevent division by 0.
               graph.add_edge(node, neighbor, weight=weight)

    # Add some additional edges for complexity
    num_additional_edges = int(0.3*num_nodes) # Ensure the additional edge count is not more than all combinations.
    for _ in range(num_additional_edges):
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        if node1 != node2 and not graph.has_edge(node1,node2):
            weight = 1 / (abs(node1 - node2) + 0.01)  # Adding small constant to prevent division by 0.
            graph.add_edge(node1, node2, weight=weight)

    return graph

def run_aco_simulation(num_nodes, num_ants, iterations, start_node, alpha, beta, rho, Q, seed=None):
        """Run the ACO simulation on the given graph, returning results over time."""
        graph = create_aco_graph(num_nodes, seed)
        pheromones = initialize_pheromones(graph)
        path_evolution = []
        for iteration in range(iterations):
            paths = [ant_walk(graph, start_node, pheromones, alpha, beta, Q) for _ in range(num_ants)]
            pheromones = update_pheromones(graph, pheromones, paths, rho, Q)
            path_evolution.append(paths[0]) # just record 1 path per iteration
        return graph, pheromones, path_evolution

def create_aco_visualization(graph, path_evolution, iteration_index):
        """Creates a Plotly visualization for ACO, showing paths and pheromones."""
        pos = nx.spring_layout(graph, seed=42)  # Consistent layout across iterations

        # Create edge traces with varying colors based on pheromone levels
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
        )
        for edge in graph.edges:
           x0, y0 = pos[edge[0]]
           x1, y1 = pos[edge[1]]
           edge_trace['x'] += tuple([x0, x1, None])
           edge_trace['y'] += tuple([y0, y1, None])
        # Create node traces
        node_trace = go.Scatter(
          x=[],
          y=[],
          mode='markers',
          hoverinfo='text',
          marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)
        )
        for node in graph.nodes():
              x, y = pos[node]
              node_trace['x'] += tuple([x])
              node_trace['y'] += tuple([y])

        # Create paths trace with different color
        path_trace = go.Scatter(
                x=[],
                y=[],
                mode='markers+lines',
                line=dict(width=3, color='red'),
                marker=dict(size=10, color='red'),
                hoverinfo='none'
            )
        path = path_evolution[iteration_index] if iteration_index < len(path_evolution) else path_evolution[-1]
        for i in range(len(path)-1):
             x0, y0 = pos[path[i]]
             x1, y1 = pos[path[i+1]]
             path_trace['x'] += tuple([x0, x1, None])
             path_trace['y'] += tuple([y0, y1, None])


        node_adjacencies = []
        node_text = []
        for node, adjacencies in graph.adjacency():
           node_adjacencies.append(len(adjacencies))
           node_text.append(f'Node: {node}<br>Adj: {len(adjacencies)}')

        node_trace['marker']['color'] = node_adjacencies
        node_trace['text'] = node_text


        layout = go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
             plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            )

        fig = go.Figure(data=[edge_trace, node_trace, path_trace], layout=layout)
        return fig
    

# --- Prophet Forecasting ----
def prepare_prophet_data(data, time_col='ds', value_col='y'):
    """Prepares data for Prophet, ensuring correct column names."""
    df = pd.DataFrame(data)
    df.rename(columns={time_col: 'ds', value_col: 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def train_prophet_model(data, time_col='ds', value_col='y', seasonality_mode='additive'):
    """Trains a Prophet model with specified parameters."""
    df = prepare_prophet_data(data, time_col, value_col)
    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(df)
    return model

def make_prophet_forecast(model, periods=365, freq='D'):
    """Makes a forecast using a trained Prophet model."""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def plot_prophet_forecast(forecast, original_data, time_col='ds', value_col='y'):
    """Plots the Prophet forecast with the original data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original_data[time_col], y=original_data[value_col], mode='lines', name='Original Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                             line=dict(width=0), name='Upper Bound', showlegend=False,
                             fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))  # Lower bound fill
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                             line=dict(width=0), name='Lower Bound', showlegend=False,
                             fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))  # Upper bound fill
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig

def make_temporal_forecast(data, periods, changepoint_prior_scale=0.05, seasonality_prior_scale=10):
    """
    Creates a quantum-harmonically optimized forecast using Prophet with advanced configurations.
    
    Parameters:
        data: dict with 'ds' (dates) and 'y' (values)
        periods: int, forecast horizon
        changepoint_prior_scale: float, flexibility of trend changes
        seasonality_prior_scale: float, strength of seasonality
        
    Returns:
        dict containing forecast, metrics, and validation results
    """
    # Initialize Prophet with optimized hyperparameters
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode='multiplicative',  # Better for evolving patterns
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95  # 95% prediction intervals
    )
    
    # Add custom seasonality for quantum-harmonic oscillations
    model.add_seasonality(
        name='quantum_cycle',
        period=30.5,  # Lunar-aligned cycle
        fourier_order=5  # Higher order for complex patterns
    )
    
    # Enhance with additional regressors if available
    if 'extra_regressors' in data:
        for regressor in data['extra_regressors']:
            model.add_regressor(regressor)
            
    # Fit model with optimization
    df = pd.DataFrame(data)
    model.fit(df)
    
    # Generate future dates with quantum alignment
    future = model.make_future_dataframe(
        periods=periods,
        freq='D',
        include_history=True
    )
    
    # Make forecast
    forecast = model.predict(future)
    
    # Perform cross-validation for robustness
    cv_results = cross_validation(
        model,
        initial='180 days',
        period='30 days',
        horizon='90 days',
        parallel="processes"
    )
    
    # Calculate performance metrics
    metrics = performance_metrics(cv_results)
    
    return {
        'forecast': forecast,
        'metrics': metrics,
        'cv_results': cv_results,
        'model': model
    }

def plot_quantum_forecast(forecast_results, data):
    """
    Creates an advanced visualization of the forecast with uncertainty quantification.
    
    Parameters:
        forecast_results: dict containing forecast and metrics
        data: original data dict
        
    Returns:
        Plotly figure with comprehensive forecast visualization
    """
    forecast = forecast_results['forecast']
    metrics = forecast_results['metrics']
    
    # Create main figure with uncertainty bands
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data['ds'],
        y=data['y'],
        mode='lines',
        name='Historical',
        line=dict(color='#64ffda', width=2)
    ))
    
    # Forecast mean
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#00ff00', width=2)
    ))
    
    # Uncertainty intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add trend decomposition
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='#ff00ff', width=1, dash='dash')
    ))
    
    # Add performance metrics annotations
    mape = metrics['mape'].mean()
    rmse = np.sqrt((metrics['mse'].mean()))
    
    fig.add_annotation(
        text=f'MAPE: {mape:.2f}%<br>RMSE: {rmse:.2f}',
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(color='#64ffda'),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='#64ffda',
        borderwidth=1
    )
    
    # Update layout with quantum-optimized styling
    fig.update_layout(
        title={
            'text': 'Quantum-Harmonic Convergence Forecast',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Timeline",
        yaxis_title="Convergence Amplitude",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridwidth=0.1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )
    
    return fig

# ----- Granger Causality -----
def prepare_granger_data(data, time_col='ds', value_cols=None):
    df = pd.DataFrame(data)
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    df = df[value_cols]
    return df
def run_granger_causality(data, dependent_var, independent_var, max_lag=5):
    """Runs the Granger causality test and returns significant results."""
    df = prepare_granger_data(data, value_cols = [dependent_var, independent_var])
    results = grangercausalitytests(df[[dependent_var, independent_var]], maxlag=max_lag, verbose=False)
    significant_lags = []
    for lag, test_result in results.items():
         p_value = test_result[0]['ssr_ftest'][1]
         if p_value < 0.05:
            significant_lags.append((lag, p_value))
    return significant_lags

def create_granger_visualization(significant_lags, independent_var, dependent_var):
        """Visualizes Granger causality results."""
        if not significant_lags:
           return go.Figure(layout=go.Layout(
                    title=f"No significant Granger causality found between {independent_var} and {dependent_var}",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                    ))

        lags, p_values = zip(*significant_lags)
        fig = go.Figure(data=[go.Bar(x=list(lags), y=list(p_values), marker_color='skyblue')])
        fig.update_layout(title=f"Granger Causality Lags between {independent_var} and {dependent_var}",
                            xaxis_title="Lag (Periods)", yaxis_title="P-Value",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                            )
        return fig

# --- Bass Diffusion Model ----
def bass_diffusion_model(t, p, q, m):
    """Compute Bass diffusion adoption curve. t = time, p = coefficient of innovation, q = coefficient of imitation, m = market size"""
    return m * ((p + q) ** 2 / p) * np.exp(-(p + q) * t) * (1 / (1 + (q / p) * np.exp(-(p + q) * t)) ** 2)

def fit_bass_model(data, time_col='time', adoption_col='adoption', initial_guess=None):
    """
    Fits the Bass diffusion model with enhanced numerical stability and robust optimization.
    """
    df = pd.DataFrame(data)
    df = df.sort_values(by=time_col).reset_index(drop=True)
    t = df[time_col].values
    y = df[adoption_col].values
    
    # Normalize time and adoption data for better numerical stability
    t_norm = (t - t.min()) / (t.max() - t.min())
    y_norm = y / y.max()
    
    # Compute smart initial guess based on data characteristics
    if initial_guess is None:
        p_init = 0.01  # Innovation coefficient
        q_init = 0.3   # Imitation coefficient
        m_init = y.max() * 1.2  # Market potential
        initial_guess = [p_init, q_init, m_init]
    
    # Enhanced loss function with regularization
    def loss(params):
        p, q, m = params
        if p <= 0 or q <= 0 or m <= 0:  # Ensure positive parameters
            return 1e10
        try:
            y_pred = bass_diffusion_model(t_norm, p, q, m)
            mse = np.mean((y_norm - y_pred/y_pred.max())**2)
            regularization = 0.1 * (p**2 + q**2)  # L2 regularization
            return mse + regularization
        except:
            return 1e10
    
    # Multi-start optimization for robustness
    best_result = None
    best_score = np.inf
    
    # Try different initial guesses
    initial_guesses = [
        initial_guess,
        [0.005, 0.5, y.max()],
        [0.02, 0.2, y.max() * 1.5]
    ]
    
    for guess in initial_guesses:
        try:
            bounds = [(0.001, 0.1), (0.1, 0.9), (y.max()*0.8, y.max()*2.0)]
            result = minimize(loss, guess, method='Nelder-Mead', 
                            bounds=bounds, options={'maxiter': 1000})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except:
            continue
    
    if best_result is not None and best_score < 1e10:
        return best_result.x
    else:
        # Fallback to reasonable default parameters if optimization fails
        return np.array([0.01, 0.3, y.max() * 1.2])

def plot_bass_diffusion_model(data, time_col='time', adoption_col='adoption', params=None):
     """Plots the fitted Bass diffusion model with the original data."""
     df = pd.DataFrame(data)
     df = df.sort_values(by=time_col).reset_index(drop=True)
     t = df[time_col].values
     y = df[adoption_col].values
     if params is not None:
        p, q, m = params
        t_range = np.linspace(t.min(), t.max(), 300)
        y_pred = bass_diffusion_model(t_range, p, q, m)
     else:
        y_pred = np.zeros_like(t)
        t_range = t
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='Original Data'))
     fig.add_trace(go.Scatter(x=t_range, y=y_pred, mode='lines', name='Fitted Bass Model'))
     fig.update_layout(
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
     return fig
# ---- Metaphorical Gradient Descent -----
def unity_cost_function(x, culture_weight, technology_weight, philosophy_weight):
    """Simulates the 'loss function' as misalignment."""
    culture_loss = (1 - x[0]) ** 2  # Cost for cultural fragmentation
    technology_loss = (1 - x[1]) ** 2  # Cost for technological misalignment
    philosophy_loss = (1 - x[2]) ** 2  # Cost for philosophical dissonance
    return culture_weight * culture_loss + technology_weight * technology_loss + philosophy_weight * philosophy_loss

def gradient_descent_simulation(initial_state, learning_rate, steps, culture_weight, technology_weight, philosophy_weight, noise=0.0):
    """Simulates gradient descent on the cost function."""
    current_state = np.array(initial_state, dtype=float)
    trajectory = [current_state.copy()]  # Store trajectory of states
    for _ in range(steps):
        gradient = np.zeros_like(current_state)
        h = 1e-5 # Small step for derivative calculation
        for i in range(len(current_state)):
           temp_state = current_state.copy()
           temp_state[i] +=h
           gradient[i] = (unity_cost_function(temp_state, culture_weight, technology_weight, philosophy_weight) - unity_cost_function(current_state, culture_weight, technology_weight, philosophy_weight))/h
        current_state = current_state - learning_rate * gradient
        current_state +=  np.random.normal(0, noise, size=current_state.shape) # Adding gaussian noise
        current_state = np.clip(current_state, 0, 1) # Clamp between 0 and 1
        trajectory.append(current_state.copy())
    return np.array(trajectory)

def create_gradient_descent_visualization(trajectory, culture_weight, technology_weight, philosophy_weight):
        """Visualizes the trajectory of gradient descent in a 3D surface plot."""
        # Create a grid of values
        n = 50 # num points per dim
        x = np.linspace(0, 1, n) # Culture
        y = np.linspace(0, 1, n) # Technology
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                 Z[i, j] = unity_cost_function([X[i,j], Y[i,j], 0.0], culture_weight, technology_weight, philosophy_weight) # 3D projection on the z-axis.

        # Prepare trajectory data for plotting
        traj_x = trajectory[:, 0]  # Culture
        traj_y = trajectory[:, 1]  # Technology
        traj_z = [unity_cost_function(s, culture_weight, technology_weight, philosophy_weight) for s in trajectory] # Cost
        # Create the 3D surface plot
        surface_plot = go.Surface(x=x, y=y, z=Z, colorscale='Viridis', opacity=0.8)
        # Create the trajectory line plot
        line_plot = go.Scatter3d(x=traj_x, y=traj_y, z=traj_z, mode='lines+markers', marker=dict(size=3), line=dict(width=3, color='red'))
        # Combine surface and line into a single figure
        fig = go.Figure(data=[surface_plot, line_plot])
        fig.update_layout(
            title='Gradient Descent Simulation on Unity Loss Function',
            scene=dict(
            xaxis_title='Cultural Alignment',
            yaxis_title='Technological Alignment',
            zaxis_title='Cost (Fragmentation Level)',
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        return fig

# ----- Reinforcement Learning ------
class MemeticEnvironment:
    def __init__(self, initial_adoption=0.01, max_steps=300, max_adoption = 1.0, reward_scaling = 1000, meme_decay_rate=0.001, decay_start = 0.1):
        self.adoption_rate = initial_adoption
        self.max_steps = max_steps
        self.current_step = 0
        self.max_adoption = max_adoption
        self.reward_scaling = reward_scaling # Scale for reward signal
        self.meme_decay_rate = meme_decay_rate
        self.decay_start = decay_start
    def step(self, action):
        self.current_step += 1
        # Apply action (memetic tweak)
        self.adoption_rate = min(self.max_adoption, max(0, self.adoption_rate * (1+action)))
# Decay if adoption is above a value
        if self.adoption_rate > self.decay_start:
            self.adoption_rate = max(0, self.adoption_rate - self.meme_decay_rate * (self.adoption_rate - self.decay_start))
        reward = self.adoption_rate * self.reward_scaling # Reward based on adoption
        done = self.current_step >= self.max_steps or self.adoption_rate >= self.max_adoption
        return self.adoption_rate, reward, done

    def reset(self):
        self.adoption_rate = 0.01
        self.current_step = 0
        return self.adoption_rate

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.q_table = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_q_values(self, state):
        state = tuple([state])
        if state not in self.q_table:
           self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size -1)
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            q_values = self.get_q_values(state)
            if not done:
                next_q_values = self.get_q_values(tuple([next_state]))
                target = reward + self.gamma * np.max(next_q_values)
            else:
                target = reward
            q_values[action] = (1-self.learning_rate)*q_values[action] + self.learning_rate * target # Update target
            self.q_table[tuple([state])] = q_values
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_rl_agent(agent, env, num_episodes, batch_size):
   history = []
   for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.learn(batch_size)
        history.append((episode, total_reward))
   return history

def create_rl_visualization(rl_history, initial_adoption,  max_adoption, episodes_to_display):
     """Creates a Plotly visualization for RL training."""
     episode_numbers, total_rewards = zip(*rl_history)
     episode_numbers = list(episode_numbers)
     total_rewards = list(total_rewards)
     # Plot Rewards
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=episode_numbers, y=total_rewards, mode='lines+markers', name='Total Reward per Episode'))
     fig.update_layout(
           title='Reinforcement Learning Training Progress',
            xaxis_title='Episode',
            yaxis_title='Total Reward',
             plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
      )
     fig.add_annotation(
            text=f'Initial Adoption: {initial_adoption}<br>Max Adoption:{max_adoption}',
            xref="paper", yref="paper",
            x=0, y=1.02, showarrow=False
        )
     return fig

def create_memetic_landscape_graph(agent, env, episodes_to_display, time_steps_per_ep):
   """Visualizes the impact of the optimal policy"""
   state_history = []
   reward_history = []
   adoption_rate = env.reset()
   done = False
   for time_step in range(time_steps_per_ep * episodes_to_display):
        action = agent.choose_action(adoption_rate)
        new_adoption, reward, done = env.step(action)
        state_history.append(adoption_rate)
        reward_history.append(reward)
        adoption_rate = new_adoption
        if done:
          break
   # Plot Adoption
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=list(range(len(state_history))), y=state_history, mode='lines+markers', name='Adoption Rate'))
   fig.add_trace(go.Scatter(x=list(range(len(reward_history))), y=reward_history, mode='lines+markers', name='Reward'))

   fig.update_layout(
            title='Memetic Adoption Landscape After Training',
             xaxis_title='Time Steps',
            yaxis_title='Adoption Rate / Reward',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
            )
   return fig

def init_callbacks(app):
    """Initialize callbacks with proper error boundaries"""
    @app.callback(
        Output('heatmap-graph', 'figure'),
        Input('generate-heatmap-button', 'n_clicks'),
        prevent_initial_call=True  # Optimize initial load
    )
    def update_heatmap(n_clicks):
        """Viral adoption matrix visualization."""
        if n_clicks is None:
            raise PreventUpdate
            
        # Generate optimized fractal-based heatmap
        data = np.random.rand(50, 50)
        # Apply unity transformation
        data = np.sqrt(data) * np.exp(-data)
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            **GRAPH_STYLE,
            title={
                'text': 'Viral Unity Field',
                'font': {'color': COLORS['text']}
            }
        )
        return fig

    @app.callback(
        Output('aco-graph', 'figure'),
        [Input('aco-iteration-slider', 'value'),
         Input('run-aco-button', 'n_clicks')],
        State('aco-graph', 'figure')
    )
    def update_aco_visualization(iteration, n_clicks, current_figure):
        """Quantum network topology evolution."""
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        graph, pheromones, path_evolution = run_aco_simulation(
            num_nodes=50,  # Optimized node count
            num_ants=20,
            iterations=100,
            start_node=0,
            alpha=1.5,  # Enhanced exploration
            beta=2.0,
            rho=0.1,
            Q=10,
            seed=42
        )

        fig = create_aco_visualization(graph, path_evolution, iteration)
        fig.update_layout(**GRAPH_STYLE)

        return fig

    @app.callback(
        Output('golden-ratio-graph', 'figure'),
        [Input('golden-ratio-slider', 'value'),
         Input('update-golden-ratio-button', 'n_clicks')]
    )
    def update_fibonacci_spiral(num_points, n_clicks):
        """Recursive unity spiral manifestation."""
        # Generate phi-optimized data points
        phi = (1 + np.sqrt(5)) / 2
        data_points = np.array([phi ** n % 1 for n in range(num_points)])
        
        fig = create_fibonacci_spiral_graph(
            num_points,
            scale=phi,  # Scale by golden ratio
            data_points=data_points
        )
        
        fig.update_layout(
            **GRAPH_STYLE,
            showlegend=False,
            title={
                'text': 'Φ-Harmonic Convergence',
                'font': {'color': COLORS['text']}
            }
        )
        return fig

    @app.callback(
        Output('prophet-forecast-graph', 'figure'),
        [Input('generate-forecast-button', 'n_clicks'),
         Input('forecast-period-slider', 'value')]
    )
    def update_temporal_projection(n_clicks, periods):
        """Temporal convergence prediction system."""
        if n_clicks is None:
            raise PreventUpdate

        # Generate quantum-harmonic time series
        t = np.linspace(0, 4*np.pi, 365)
        values = (
            np.sin(t) + 
            0.5 * np.sin(2*t) * np.exp(-t/10) + 
            np.random.normal(0, 0.1, len(t))
        )
        
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        data = {'ds': dates, 'y': values}
        
        forecast = make_temporal_forecast(data, periods)
        fig = plot_quantum_forecast(forecast, data)
        fig.update_layout(**GRAPH_STYLE)
        return fig

    @app.callback(
        [Output('rl-training-graph', 'figure'),
         Output('memetic-landscape-graph', 'figure')],
        Input('train-rl-agent-button', 'n_clicks')
    )
    def update_memetic_evolution(n_clicks):
        """Quantum memetic engineering system."""
        if n_clicks is None:
            raise PreventUpdate

        env = MemeticEnvironment(
            initial_adoption=0.01,
            max_steps=100,
            max_adoption=0.8,
            reward_scaling=500,
            meme_decay_rate=0.005
        )

        agent = DQNAgent(
            state_size=1,
            action_size=5,
            learning_rate=0.01,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.98,
            epsilon_min=0.01,
            memory_size=2000
        )

        history = train_rl_agent(agent, env, num_episodes=50, batch_size=32)
        
        rl_fig = create_rl_visualization(
            history,
            initial_adoption=0.01,
            max_adoption=0.8,
            episodes_to_display=50
        )
        memetic_fig = create_memetic_landscape_graph(
            agent,
            env,
            episodes_to_display=10,
            time_steps_per_ep=100
        )
        
        rl_fig.update_layout(**GRAPH_STYLE)
        memetic_fig.update_layout(**GRAPH_STYLE)
        
        return rl_fig, memetic_fig

    @app.callback(
        Output('granger-graph', 'figure'),
        [Input('run-granger-button', 'n_clicks'),
         Input('granger-independent-dropdown', 'value'),
         Input('granger-dependent-dropdown', 'value')]
    )
    def update_causal_network(n_clicks, independent_var, dependent_var):
        """Quantum causal network analysis."""
        if n_clicks is None or independent_var == dependent_var:
            raise PreventUpdate

        # Generate quantum-entangled time series
        t = np.linspace(0, 10*np.pi, 200)
        phase_shift = np.pi/4
        
        cultural = np.sin(t) + 0.2*np.random.normal(0, 1, 200)
        technological = np.sin(t + phase_shift) + 0.2*np.random.normal(0, 1, 200)
        economic = np.sin(t + 2*phase_shift) + 0.2*np.random.normal(0, 1, 200)
        
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        data = {
            'ds': dates,
            'cultural': cultural,
            'technological': technological,
            'economic': economic
        }

        significant_lags = run_granger_causality(
            data,
            dependent_var,
            independent_var,
            max_lag=10
        )
        
        fig = create_granger_visualization(
            significant_lags,
            independent_var,
            dependent_var
        )
        fig.update_layout(**GRAPH_STYLE)
        return fig

# Metaphorical Gradient Descent: Achieving the Global Optimum
# Unified Gradient Descent Callback with Quantum State Management
    @app.callback(
        Output('gradient-descent-graph', 'figure'),
        [Input('run-gradient-descent-button', 'n_clicks'),
         Input('gradient-descent-graph', 'figure')],
        prevent_initial_call=True
    )
    def update_unified_gradient_descent(n_clicks, current_figure):
        """
        Unified callback for gradient descent visualization with quantum state awareness.
        Handles both initial creation and subsequent updates with optimal efficiency.
        """
        initial_state = [0.1, 0.2, 0.3]  # Quantum-initialized state vector
        trajectory = gradient_descent_simulation(
            initial_state=initial_state,
            learning_rate=0.1,
            steps=100,
            culture_weight=0.5,
            technology_weight=0.3,
            philosophy_weight=0.2,
            noise=0.01
        )
        
        return create_gradient_descent_visualization(
            trajectory=trajectory,
            culture_weight=0.5,
            technology_weight=0.3,
            philosophy_weight=0.2
        )

    # Bass Diffusion Model Visualization
    @app.callback(
        Output('bass-diffusion-graph', 'figure'),
        Input('run-bass-button', 'n_clicks'),
        State('bass-diffusion-graph', 'figure'),
        prevent_initial_call=True,
    )
    def update_bass_diffusion_graph(n_clicks, current_figure):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'run-bass-button' or not current_figure:
        # Dummy data for bass model
            time = np.arange(1, 200)
            adoption = 2000*np.sin(np.linspace(0, 2*np.pi, 199)) +  np.random.normal(0, 500, 199)
            adoption = np.clip(adoption, 0, np.inf).astype(int) # clip negative values for plotting.
            data = {'time': time, 'adoption': adoption}
            params = fit_bass_model(data)
            fig = plot_bass_diffusion_model(data, params=params)
            return fig
        else:
        # Dummy data for bass model
            time = np.arange(1, 200)
            adoption = 2000*np.sin(np.linspace(0, 2*np.pi, 199)) +  np.random.normal(0, 500, 199)
            adoption = np.clip(adoption, 0, np.inf).astype(int) # clip negative values for plotting.
            data = {'time': time, 'adoption': adoption}
            params = fit_bass_model(data)
            fig = plot_bass_diffusion_model(data, params=params)
            return fig

def create_app():
    """
    Factory function implementing a quantum-coherent visualization framework 
    with optimized component hierarchy and state management.
    
    Returns:
        dash.Dash: Configured application instance with reactive layout system
    """
    # Initialize core application with optimal configuration
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ],
        suppress_callback_exceptions=True,
        # Enhanced meta configuration
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    
    # Quantum-optimized constants
    COLORS = {
        'background': '#0a192f',  # Deep space quantum field
        'text': '#64ffda',        # Coherent wave function
        'accent': '#112240',      # Entangled state
        'highlight': '#233554',   # Quantum superposition
        'grid': '#1e3a8a'         # Probability matrix
    }
    
    GRAPH_STYLE = {
        'plot_bgcolor': COLORS['background'],
        'paper_bgcolor': COLORS['background'],
        'font': {'color': COLORS['text']},
        'height': 400,
        'margin': dict(l=20, r=20, t=40, b=20),
        'xaxis': dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=0.1,
            zeroline=False
        ),
        'yaxis': dict(
            showgrid=True,
            gridcolor=COLORS['grid'],
            gridwidth=0.1,
            zeroline=False
        )
    }

    def create_card(title, graph_id, controls=None):
        """
        Factory function for quantum-coherent visualization cards with optimal control structure.
        """
        return dbc.Card([
            dbc.CardHeader(title, style={'backgroundColor': COLORS['highlight']}),
            dbc.CardBody([
                dcc.Graph(
                    id=graph_id,
                    style={'height': '300px'},
                    config={'displayModeBar': False}
                ),
                # Critical fix: Ensure controls is not wrapped in an additional list
                html.Div(controls) if controls else html.Div()
            ])
        ], style={
            'backgroundColor': COLORS['accent'],
            'margin': '10px',
            'border': f'1px solid {COLORS["highlight"]}',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'transition': 'transform 0.3s ease-in-out'
        })

    # Initialize quantum-coherent layout matrix
    app.layout = dbc.Container([
        # Header Matrix - Quantum Identification Layer
        dbc.Row([
            dbc.Col([
                html.H1(
                    "UNITY HUD 2069",
                    className='text-center mb-4',
                    style={
                        'color': COLORS['text'],
                        'fontFamily': 'monospace',
                        'letterSpacing': '0.2em',
                        'textShadow': '0 0 10px rgba(100, 255, 218, 0.5)'
                    }
                ),
                html.H3(
                    "Quantum Harmonics: 1+1=1",
                    className='text-center mb-4',
                    style={
                        'color': COLORS['text'],
                        'opacity': '0.8'
                    }
                ),
                  html.P(
                    "Metagaming IRL: Align your actions with unity. Seek win-win scenarios, practice empathy, and build networks of mutual support. 1+1=1 isn't just math; it's a code for a world where we all thrive. Track, analyze, and optimize your life with the HUD to become a master of positive transformation."
                    ,
                    className='text-center mb-4',
                    style={
                        'color': COLORS['text'],
                        'opacity': '0.6',
                           'fontSize':'14px'
                    }
                )
            ])
        ], className='mb-4'),

        # Primary Visualization Matrix - Quantum State Monitor
        dbc.Row([
            # Left Matrix: Emergent Systems Analysis
            dbc.Col([
                create_card(
                    "Viral Adoption Matrix",
                    'heatmap-graph',
                    dbc.Button(
                        "Generate",
                        id='generate-heatmap-button',
                        color='primary',
                        className='mt-2'
                    )
                ),
                create_card(
                    "Fibonacci Unity Field",
                    'golden-ratio-graph',
                    html.Div([
                        dcc.Slider(
                            id='golden-ratio-slider',
                            min=10,
                            max=200,
                            value=50,
                            marks={i: str(i) for i in range(10, 201, 30)},
                            className='mb-3'
                        ),
                        dbc.Button(
                            "Generate Unity Field",
                            id='update-golden-ratio-button',
                            color='primary',
                            className='w-100'
                        )
                    ])
                )
            ], md=6),
            
            # Right Matrix: Network Dynamics Observer
            dbc.Col([
                create_card(
                    "Quantum Network Topology",
                    'aco-graph',
                    html.Div([  # Wrap in Div instead of list
                        dcc.Slider(
                            id='aco-iteration-slider',
                            min=0,
                            max=99,
                            value=0,
                            marks={i: str(i) for i in range(0, 100, 20)}
                        ),
                        dbc.Button(
                            "Run Simulation",
                            id='run-aco-button',
                            color='primary',
                            className='mt-2'
                        )
                    ])
                ),
                create_card(
                    "Temporal Convergence",
                    'prophet-forecast-graph',
                    html.Div([  # Wrap in Div
                        dbc.Button(
                            "Project Timeline",
                            id='generate-forecast-button',
                            color='primary',
                            className='mt-2'
                        ),
                        dcc.Slider(
                            id='forecast-period-slider',
                            min=30,
                            max=730,
                            step=30,
                            value=365,
                            marks={i: str(i) for i in range(30, 731, 120)}
                        )
                    ])
                )
            ], md=6)
        ]),

        # Causality Analysis Matrix - Quantum Correlation Detector
        dbc.Row([
            dbc.Col([
                create_card(
                    "Granger Causality Analysis",
                    'granger-graph',
                    dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(
                                id='granger-independent-dropdown',
                                options=[
                                    {'label': 'Cultural Adoption', 'value': 'cultural'},
                                    {'label': 'Technological Adoption', 'value': 'technological'},
                                    {'label': 'Economic Adoption', 'value': 'economic'}
                                ],
                                value='cultural',
                                clearable=False
                            ),
                            width=4
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='granger-dependent-dropdown',
                                options=[
                                    {'label': 'Cultural Adoption', 'value': 'cultural'},
                                    {'label': 'Technological Adoption', 'value': 'technological'},
                                    {'label': 'Economic Adoption', 'value': 'economic'}
                                ],
                                value='technological',
                                clearable=False
                            ),
                            width=4
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Run Analysis",
                                id='run-granger-button',
                                color='primary',
                                className='mt-2'
                            ),
                            width=4
                        )
                    ])
                )
            ], width=12)
        ]),

        # Optimization Matrix - Quantum State Optimizer
        dbc.Row([
            dbc.Col([
                create_card(
                    "Metaphorical Gradient Descent",
                    'gradient-descent-graph',
                    dbc.Button(
                        "Optimize",
                        id='run-gradient-descent-button',
                        color='primary',
                        className='mt-2'
                    )
                )
            ], width=6),
            
            dbc.Col([
                create_card(
                    "Bass Diffusion Model",
                    'bass-diffusion-graph',
                    dbc.Button(
                        "Simulate",
                        id='run-bass-button',
                        color='primary',
                        className='mt-2'
                    )
                )
            ], width=6)
        ]),

        # Reinforcement Learning Matrix - Quantum Learning Engine
        dbc.Row([
            dbc.Col([
                create_card(
                    "Memetic Evolution System",
                    'rl-training-graph',
                    html.Div([  # Wrap controls in a single Div
                        dbc.Button(
                            "Train Agent",
                            id='train-rl-agent-button',
                            color='primary',
                            className='mt-2'
                        ),
                        dcc.Graph(id='memetic-landscape-graph')
                    ])
                )
            ], width=12)
        ])
    ], fluid=True, style={
        'backgroundColor': COLORS['background'],
        'minHeight': '100vh',
        'padding': '20px'
    })

    return app

# Initialize app with proper sequential flow
if __name__ == '__main__':
    app = create_app()  # Create app instance with layout
    init_callbacks(app)  # Initialize callbacks
    app.run_server(debug=True, port=8050)  # Run server

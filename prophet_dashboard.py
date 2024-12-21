import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import json
import pycountry
from scipy.signal import savgol_filter
import plotly.io as pio

# --- Setup Streamlit Page ---
st.set_page_config(
    page_title="Modeling 1+1=1's Evolution",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Define a global template for all plots
pio.templates["custom_light"] = pio.templates["plotly_white"]
pio.templates["custom_light"].layout.update(
    font=dict(family="Arial, sans-serif", size=14, color="#2c3e50"),
    title=dict(font=dict(size=20, color="#0073e6")),
    xaxis=dict(
        title=dict(font=dict(size=16)),
        gridcolor="rgba(200,200,200,0.5)",
        zerolinecolor="rgba(200,200,200,0.5)",
        linecolor="rgba(44,62,80,0.8)",
    ),
    yaxis=dict(
        title=dict(font=dict(size=16)),
        gridcolor="rgba(200,200,200,0.5)",
        zerolinecolor="rgba(200,200,200,0.5)",
        linecolor="rgba(44,62,80,0.8)",
    ),
    legend=dict(
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(200,200,200,0.5)",
    ),
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(240,245,255,1)",
)

# Set the global default template
pio.templates.default = "custom_light"

# --- Theme CSS (Enhanced Blue Futuristic) ---
st.markdown(
    """
    <style>
        /* --- Base Body Styling --- */
        body {
            background-color: #f4f7fc; /* Light futuristic background */
            color: #2c3e50; /* Professional dark gray text */
            font-family: 'Inter', sans-serif; /* Clean, modern font */
            line-height: 1.6;
            overflow: hidden; /* Remove scrollbars for better UX */
        }

        /* --- Particles Background Animation --- */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #ffffff, #eaf3fc); /* Subtle gradient */
            z-index: -1; /* Place particles in the background */
        }
        @keyframes particles {
            0% { transform: translateY(-100px); }
            100% { transform: translateY(800px); }
        }
        .particle {
            position: absolute;
            width: 10px;
            height: 10px;
            background-color: rgba(0, 115, 230, 0.7); /* Futuristic blue */
            border-radius: 50%;
            animation: particles 4s linear infinite;
        }

        /* --- Main App Styling --- */
        .stApp {
            background-color: #ffffff; /* Clean white panel background */
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15); /* Subtle shadow */
        }

        /* --- Card Components --- */
        .st-bo {
            background-color: #f0f4f8; /* Soft gray for cards */
            color: #34495e; /* Darker gray for text */
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1); /* Subtle shadow for depth */
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }
        .st-bo:hover {
            transform: scale(1.02); /* Slight hover scaling effect */
            box-shadow: 0 5px 15px rgba(0,0,0,0.2); /* Highlight shadow on hover */
        }

        /* --- Select Boxes --- */
        .st-cx {
            background-color: #eaf3fc; /* Subtle blue for selects */
            color: #34495e; /* Consistent dark gray text */
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 12px;
        }

        /* --- Accents and Highlights --- */
        .st-d0 {
            color: #0073e6; /* Executive blue accent */
        }

        /* --- Sidebar Styling --- */
        .st.sidebar .sidebar-content {
            background-color: #f8fafc;  /* Soft light gray */
            color: #34495e;  /* Professional dark gray text */
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        .st.sidebar .sidebar-content h1 {
            color: #0073e6;  /* Executive blue headers */
            margin-bottom: 1rem;
        }
        .st.sidebar .sidebar-content .st-bo {
            background-color: #ffffff; /* Clean white for cards */
            border: 1px solid #e5e5e5; /* Subtle border */
            border-radius: 8px;
        }

        /* --- Headers and Typography --- */
        h1, h2, h3, h4, h5, h6 {
            color: #0073e6; /* Executive blue headers */
        }

        /* --- Padding for Main Container --- */
        .reportview-container .main .block-container {
            padding-top: 30px;
            padding-bottom: 30px;
        }

        /* --- Plotly Graph Styling --- */
        .plotly-graph-div {
            background-color: #ffffff; /* Clean white for charts */
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); /* Subtle shadow */
        }

        /* --- Slider Styling --- */
        .stSlider > div > div > div > div {
            background-color: #0073e6; /* Executive blue slider */
        }

        /* --- Metric Highlight Boxes --- */
        .st-bx {
            color: #1f8fe5; /* Brighter blue for metrics */
        }

        /* --- Buttons and Hover Effects --- */
        button {
            background-color: #0073e6; /* Futuristic blue button */
            color: #ffffff; /* White text */
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #005bb5; /* Darker blue on hover */
        }
    </style>

    <!-- --- Particles Background --- -->
    <div class="particles">
        <div class="particle" style="left: 10%; animation-duration: 3.5s;"></div>
        <div class="particle" style="left: 30%; animation-duration: 3s;"></div>
        <div class="particle" style="left: 50%; animation-duration: 2.5s;"></div>
        <div class="particle" style="left: 70%; animation-duration: 4s;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- MetaBro Easter Eggs ---
metabro_cheat_code = "420691337"
golden_ratio = (1 + np.sqrt(5)) / 2

# --- Prophet Model Configuration ---
def configure_prophet_model(growth_type='logistic', flexibility=0.05, capacity_multiplier=1.0):
    """
    Enhanced Prophet model configuration with optimized hyperparameters.
    """
    return Prophet(
        growth=growth_type,
        yearly_seasonality=10,  # Reduced for better fit
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=flexibility,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=0.1,
        mcmc_samples=0,
        interval_width=0.95,  # Added for better uncertainty bounds
        changepoint_range=0.9,  # Added for more flexible trend changes
        n_changepoints=25  # Optimized number of changepoints
    )

# --- Data Generation and Advanced Forecasting ---
@st.cache_data
def generate_and_forecast_data_advanced():
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", end="2027-12-31", freq='ME'))
    n_points = len(dates)
    historical_mask = dates <= pd.to_datetime('2024-12-31')

    # --- Initialize sigmoid-based natural growth ---
    time_component = np.linspace(-6, 6, n_points)  # Logistic growth input range (-6 to 6)
    logistic_growth = 1 / (1 + np.exp(-time_component))  # Standard logistic curve (0 to 1)

    # Scale the logistic growth to simulate gradual adoption
    base_growth = 0.02 + 0.98 * logistic_growth  # Starting at 0.02, growing to ~1
    base_growth = base_growth * 2  # Scale to represent adoption reaching ~200% max capacity

    # Add small noise for realism
    base_growth += np.random.normal(0, 0.01, size=n_points)
    base_growth = np.maximum(0, base_growth)  # Prevent negative values

    # Create main DataFrame
    data = pd.DataFrame({'ds': dates})
    data['overall_adoption'] = base_growth

    # --- Smooth dimension-specific growth with Gaussian curves ---
    dimensions_data = {
        'cultural_resonance': {'center': 2025, 'width': 1.2, 'amplitude': 0.7},
        'scientific_acknowledgment': {'center': 2026, 'width': 1.4, 'amplitude': 0.5},
        'philosophical_integration': {'center': 2025.5, 'width': 1.3, 'amplitude': 0.6},
        'technological_embedding': {'center': 2027, 'width': 1.6, 'amplitude': 0.8}
    }

    for dim, params in dimensions_data.items():
        center_year = params['center']
        width = params['width']
        amplitude = params['amplitude']

        # Gaussian curve for adoption
        gaussian_curve = amplitude * np.exp(
            -((dates.year + dates.month / 12 - center_year) ** 2) / (2 * width**2)
        )

        # Add noise and prevent negatives
        dimension_growth = gaussian_curve + np.random.normal(0, 0.01, size=n_points)
        dimension_growth = np.maximum(0, dimension_growth)

        data[dim] = dimension_growth

    # --- Regional adoption with sinusoidal variation ---
    regions = ['Global Consciousness', 'Academia', 'Digital Space', 'Spiritual Communities']
    for i, region in enumerate(regions):
        # Base sinusoidal growth pattern for regional adoption
        growth_factor = (
            0.2 + np.sin(np.linspace(0, 2 * np.pi * (i + 1), n_points)) * 0.05
        )  # Sinusoidal fluctuation
        start_point = 0.05 * (i + 1)
        growth_speed = 0.01 + np.random.uniform(-0.0005, 0.0005)  # Randomized growth rates
        regional_growth = start_point + np.cumsum(growth_speed * growth_factor)
        regional_growth = np.maximum(0, regional_growth)  # Prevent negative values

        data[f'adoption_{region.lower().replace(" ", "_")}'] = regional_growth

    # --- Historical mask (for Prophet model) ---
    data_historical = data[historical_mask].copy()

    # --- Forecasting with Prophet ---
    forecasts = {}
    prophet_cols = ['overall_adoption'] + list(dimensions_data.keys())

    for col in prophet_cols:
        base_capacity = data_historical[col].max() * 1.2  # Capacity slightly above max historical

        # Scenarios
        scenarios = {
            'realistic': {'capacity_mult': 1.0, 'changepoint_scale': 0.1},
            'optimistic': {'capacity_mult': 1.5, 'changepoint_scale': 0.15},
            'pessimistic': {'capacity_mult': 0.8, 'changepoint_scale': 0.05}
        }

        for scenario, params in scenarios.items():
            df_prophet = pd.DataFrame({
                'ds': data_historical['ds'],
                'y': data_historical[col],
                'cap': base_capacity * params['capacity_mult']  # Capacity for the scenario
            })

            # Prophet configuration
            model = Prophet(
                growth='logistic',
                yearly_seasonality='auto',
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=params['changepoint_scale'],
                seasonality_mode='additive'
            )
            model.fit(df_prophet)

            # Future data
            future = model.make_future_dataframe(
                periods=len(dates) - len(data_historical),
                freq='ME'
            )
            future['cap'] = base_capacity * params['capacity_mult']

            forecast = model.predict(future)
            forecasts[f'{col}_{scenario}'] = forecast

    return data, forecasts, regions

data, forecasts, regions = generate_and_forecast_data_advanced()
def create_advanced_forecast_visualization(data, forecasts, scenario, key="forecast_chart"):
    """
    Create an advanced Prophet forecast visualization with enhanced design and usability.
    """
    # Fetch relevant forecast data based on the selected scenario
    forecast_key = f"overall_adoption_{scenario.lower()}"
    forecast_data = forecasts[forecast_key]

    # Initialize a Plotly figure
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=data['ds'],
        y=data['overall_adoption'],
        mode='lines',
        name='Historical Data',
        line=dict(color='#0073e6', width=3),
        hovertemplate='Date: %{x}<br>Adoption: %{y:.4f}<extra></extra>'
    ))

    # Add forecasted data trace
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name=f'{scenario} Forecast',
        line=dict(color='#1f8fe5', width=2, dash="solid"),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.4f}<extra></extra>'
    ))

    # Add uncertainty bounds (upper and lower)
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        fill='tonexty',
        fillcolor='rgba(31, 143, 229, 0.2)',
        mode='lines',
        line=dict(width=0),
        name='Uncertainty Range'
    ))

    # Update layout for a polished, professional look
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(240,245,255,1)',
        title=dict(
            text=f"1+1=1 Adoption Forecast ({scenario} Scenario)",
            font=dict(size=20, color='#2c3e50'),
            x=0.5  # Center the title
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(200,200,200,0.5)',
            showline=True,
            linewidth=1.5,
            linecolor='rgba(44,62,80,0.8)'
        ),
        yaxis=dict(
            title="Adoption Level",
            gridcolor='rgba(200,200,200,0.5)',
            showline=True,
            linewidth=1.5,
            linecolor='rgba(44,62,80,0.8)'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(240,245,255,0.8)',
            font_size=12,
            font_color='#2c3e50'
        ),
        legend=dict(
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1
        ),
        height=600,
    )

    return fig

def create_components_visualization(forecast_data, dimension_name, key):
    """
    Create enhanced component visualization with improved clarity
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend Analysis', 'Cyclical Patterns', 'Growth Dynamics'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    # Enhanced Trend Component
    fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['trend'],
            name='Core Trend',
            line=dict(color='#00bfff', width=3),
            mode='lines+markers',
            marker=dict(size=4)
        ),
        row=1, col=1
    )

    # Add uncertainty bounds for trend
    if 'trend_lower' in forecast_data.columns and 'trend_upper' in forecast_data.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['trend_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,191,255,0.2)',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['trend_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,191,255,0.2)',
                name='Trend Uncertainty'
            ),
            row=1, col=1
        )

    # Enhanced Yearly Pattern
    if 'yearly' in forecast_data.columns:
        yearly_data = forecast_data['yearly'].rolling(window=7, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=pd.date_range(start='2022', periods=len(yearly_data), freq='D'),
                y=yearly_data,
                name='Yearly Cycle',
                line=dict(color='#6bc1ff', width=2),
                mode='lines'
            ),
            row=2, col=1
        )

    # Enhanced Growth Rate with Acceleration
    growth_rate = np.gradient(forecast_data['trend'])
    acceleration = np.gradient(growth_rate)

    fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=growth_rate,
            name='Growth Rate',
            line=dict(color='#a3d1f0', width=2)
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_data['ds'],
            y=acceleration,
            name='Acceleration',
            line=dict(color='#c6dff7', width=2),
            yaxis="y2"
        ),
        row=3, col=1,
        secondary_y=True
    )

    # Enhanced Layout
    fig.update_layout(
        height=900,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(25,25,112,0.3)',
        title=f"{dimension_name} Dimensional Analysis",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        ),
        updatemenus=[{
            'buttons': [
                {'args': [{'visible': [True] * len(fig.data)}],
                 'label': 'All',
                 'method': 'restyle'},
                {'args': [{'visible': [i < 3 for i in range(len(fig.data))]}],
                 'label': 'Trend',
                 'method': 'restyle'},
                {'args': [{'visible': [3 <= i < 6 for i in range(len(fig.data))]}],
                 'label': 'Yearly',
                 'method': 'restyle'},
                {'args': [{'visible': [i >= 6 for i in range(len(fig.data))]}],
                 'label': 'Growth',
                 'method': 'restyle'}
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.1
        }]
    )

    # Enhanced Axes
    for i in range(1, 4):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(128,128,128,0.5)',
            row=i,
            col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(128,128,128,0.5)',
            row=i,
            col=1
        )

    return fig

def create_forecast_visualization(data, forecasts, scenario, key):
    fig = go.Figure()
    
    # Enhanced historical data visualization
    fig.add_trace(go.Scatter(
        x=data['ds'],
        y=data['overall_adoption'],
        mode='lines',
        name='Historical Data',
        line=dict(color='#00bfff', width=3),
        hovertemplate='Date: %{x}<br>Adoption: %{y:.3f}<extra></extra>'
    ))
    
    # Forecast visualization with uncertainty bands
    forecast_key = f'overall_adoption_{scenario.lower()}'
    forecast_data = forecasts[forecast_key]
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name=f'{scenario} Forecast',
        line=dict(color='#6bc1ff', width=2),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}<extra></extra>'
    ))
    
    # Uncertainty bands
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        fill='tonexty',
        fillcolor='rgba(107,193,255,0.2)',
        mode='lines',
        name='Uncertainty Range',
        line=dict(width=0)
    ))
    
    # Enhanced layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,10,30,0.9)',
        plot_bgcolor='rgba(0,10,30,0.9)',
        title=dict(
            text="1+1=1 Adoption Forecast",
            font=dict(size=24, color='#00bfff')
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            title="Adoption Level",
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.3)'
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=12
        )
    )
    
    return fig

def create_evolution_animation(filtered_data, selected_regions, key):
    """
    Optimized regional evolution animation with proper frame handling.
    """
    frames = []
    dates = filtered_data['ds'].unique()
    
    # Pre-compute color mapping for consistency
    color_sequence = px.colors.qualitative.Set2[:len(selected_regions)]
    color_map = dict(zip(selected_regions, color_sequence))
    
    for date in dates:
        frame_data = filtered_data[filtered_data['ds'] == date]
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=[date],
                    y=[frame_data[frame_data['region'] == region]['adoption_level'].iloc[0]],
                    name=region,
                    mode='lines+markers',
                    line=dict(color=color_map[region]),
                    showlegend=True if date == dates[0] else False
                ) for region in selected_regions if region in frame_data['region'].values
            ],
            name=str(date)
        )
        frames.append(frame)
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[dates[0]],
                y=[filtered_data[filtered_data['region'] == region]['adoption_level'].iloc[0]],
                name=region,
                mode='lines+markers',
                line=dict(color=color_map[region])
            ) for region in selected_regions
        ],
        frames=frames
    )
    
    # Animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                dict(label='Play',
                     method='animate',
                     args=[None, {'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True}]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate'}])
            ]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Date: '},
            'steps': [{'args': [[str(date)]], 'label': date.strftime('%Y-%m'),
                      'method': 'animate'} for date in dates]
        }]
    )
    
    return fig


def create_evolution_visualization(data, regions, selected_time_range, key):
    # Prepare data
    plot_data = data.copy()
    plot_data = plot_data[(plot_data['ds'] >= selected_time_range[0]) & 
                         (plot_data['ds'] <= selected_time_range[1])]
    
    # Create base figure
    fig = go.Figure()
    
    # Add traces for each region
    colors = ['#00bfff', '#6bc1ff', '#a3d1f0', '#c6dff7']
    for idx, region in enumerate(regions):
        region_key = f'adoption_{region.lower().replace(" ", "_")}'
        
        fig.add_trace(go.Scatter(
            x=plot_data['ds'],
            y=plot_data[region_key],
            name=region,
            mode='lines',
            line=dict(color=colors[idx % len(colors)], width=2),
            fill='tonexty',
            fillcolor=f'rgba{tuple(int(c*255) for c in plt.matplotlib.colors.to_rgb(colors[idx % len(colors)])) + (0.1,)}'
        ))
    
    # Enhanced layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,10,30,0.9)',
        plot_bgcolor='rgba(0,10,30,0.9)',
        title=dict(
            text="Regional Evolution of 1+1=1 Adoption",
            font=dict(size=24, color='#00bfff')
        ),
        xaxis=dict(
            title="Time",
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            title="Adoption Level",
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(255,255,255,0.3)'
        ),
        hovermode='x unified',
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                }]
            }]
        }]
    )
    
    return fig
# --- Metagaming IRL Visualization ---
def create_metagaming_irl_visualization():
    """Create a static visualization of a network graph for metagaming."""
    fig = go.Figure()
    nodes = [
        {'id': 0, 'label': 'Individual', 'color': '#00bfff'},
        {'id': 1, 'label': 'Small Group', 'color': '#6bc1ff'},
        {'id': 2, 'label': 'Community', 'color': '#a3d1f0'},
        {'id': 3, 'label': 'Organization', 'color': '#c6dff7'},
        {'id': 4, 'label': 'Society', 'color': '#d0e7ff'},
        {'id': 5, 'label': 'Global Networks', 'color': '#e5f0ff'}
    ]
    edges = [
        {'source': 0, 'target': 1, 'weight': 0.6},
        {'source': 1, 'target': 2, 'weight': 0.7},
        {'source': 2, 'target': 3, 'weight': 0.8},
        {'source': 3, 'target': 4, 'weight': 0.9},
        {'source': 0, 'target': 2, 'weight': 0.3},
        {'source': 1, 'target': 3, 'weight': 0.4},
        {'source': 4, 'target': 5, 'weight': 0.95},
        {'source': 2, 'target': 5, 'weight': 0.85}
    ]

    node_x = [0, 0.2, 0.6, 0.8, 1.0, 0.8]
    node_y = [0.5, 0.6, 0.5, 0.4, 0.5, 0.2]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node['label'] for node in nodes],
        textposition='bottom center',
        marker=dict(
            size=30,
            color=[node['color'] for node in nodes],
            line=dict(width=2, color='black')),
        hoverinfo='text'
    )

    edge_x = []
    edge_y = []
    for edge in edges:
        source_x = node_x[edge['source']]
        source_y = node_y[edge['source']]
        target_x = node_x[edge['target']]
        target_y = node_y[edge['target']]
        edge_x.extend([source_x, target_x, None])
        edge_y.extend([source_y, target_y, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(
        title='IRL Metagaming: Exponential Adoption of 1+1=1',
        title_x=0.5,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(25,25,112,0.3)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

# --- Netherlands Specific Visualization ---
@st.cache_data
def load_netherlands_map():
    with open('netherlands_municipalities.json', 'r') as f:
        geojson_data = json.load(f)
    return geojson_data

def create_netherlands_map(data):
    """Create a simplified map focusing on key municipalities."""
    # Define core municipalities for visualization
    key_cities = {
        "Utrecht": {"lat": 52.0907, "lon": 5.1214, "label": "Initial Seed"},
        "Alphen aan den Rijn": {"lat": 52.1324, "lon": 4.6645, "label": "Initial Seed"},
        "Amsterdam": {"lat": 52.3676, "lon": 4.9041, "label": "Projected Vector"},
        "Tilburg": {"lat": 51.5647, "lon": 5.0907, "label": "Projected Vector"},
    }
    
    fig = go.Figure()
    
    # Add city markers
    fig.add_trace(go.Scattergeo(
        lon=[city["lon"] for city in key_cities.values()],
        lat=[city["lat"] for city in key_cities.values()],
        text=[f"{city} - {props['label']}" for city, props in key_cities.items()],
        mode='markers+text',
        marker=dict(size=12, color='#00bfff'),
        textposition="bottom center"
    ))
    
    # Add connection lines between seed cities and projected vectors
    for start_city in ["Utrecht", "Alphen aan den Rijn"]:
        for end_city in ["Amsterdam", "Tilburg"]:
            fig.add_trace(go.Scattergeo(
                lon=[key_cities[start_city]["lon"], key_cities[end_city]["lon"]],
                lat=[key_cities[start_city]["lat"], key_cities[end_city]["lat"]],
                mode='lines',
                line=dict(width=1, color='#a3d1f0', dash='dash'),
                showlegend=False
            ))
    
    # Configure map layout
    fig.update_layout(
        geo=dict(
            scope='europe',
            center=dict(lat=52.1326, lon=5.2913),
            projection_scale=20,
            showland=True,
            landcolor='rgb(0, 10, 20)',
            showcoastlines=True,
            coastlinecolor='rgba(255, 255, 255, 0.2)',
            showocean=True,
            oceancolor='rgb(0, 5, 10)'
        ),
        paper_bgcolor='rgba(0,0,0,0.1)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title=dict(
            text="1+1=1 Emergence in the Netherlands",
            font=dict(color='#00bfff', size=20),
            x=0.5
        )
    )
    
    return fig
def create_forecast_trace(forecasts, scenario, color_map):
    """
    Optimized forecast trace creation with proper error handling and type validation.
    """
    scenario_key = f'overall_adoption_{scenario.lower()}'
    if scenario_key not in forecasts:
        raise KeyError(f"Scenario {scenario} not found in forecasts")
        
    forecast_data = forecasts[scenario_key]
    color = color_map.get(scenario, '#6bc1ff')  # Default color fallback
    
    return [
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name=f'{scenario} Forecast',
            line=dict(color=color, width=2)
        ),
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            fillcolor=f'rgba{tuple(int(c*255) for c in plt.matplotlib.colors.to_rgb(color)) + (0.2,)}',
            mode='lines',
            name=f'{scenario} Uncertainty',
            line=dict(width=0)
        )
    ]
def create_golden_ratio_visualization(key):
    """
    Creates an advanced visualization of the golden ratio incorporating 
    Fibonacci spirals and dynamic mathematical patterns.
    """
    phi = (1 + np.sqrt(5)) / 2
    t = np.linspace(0, 8 * np.pi, 1000)
    
    # Generate multiple spirals with phi-based scaling
    spirals = []
    for i in range(3):
        scale = phi ** i
        x_spiral = scale * np.exp(t/phi) * np.cos(t)
        y_spiral = scale * np.exp(t/phi) * np.sin(t)
        spirals.append((x_spiral, y_spiral))
    
    # Create figure with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Fibonacci Spiral Evolution',
            'Golden Ratio Harmonic Pattern',
            'Phi-based Network',
            'Recursive Subdivision'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Plot 1: Enhanced Fibonacci Spiral
    colors = ['#00bfff', '#6bc1ff', '#a3d1f0']
    for idx, (x, y) in enumerate(spirals):
        fig.add_trace(
                        go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=colors[idx], width=2-idx*0.5),
                name=f'Spiral {idx+1}'
            ),
            row=1, col=1
        )
    
    # Plot 2: Golden Ratio Harmonic Pattern
    theta = np.linspace(0, 2*np.pi, 1000)
    r = np.exp(theta/phi)
    x_harm = r * np.cos(theta * phi)
    y_harm = r * np.sin(theta * phi)
    fig.add_trace(
        go.Scatter(
            x=x_harm, y=y_harm,
            mode='lines',
            line=dict(
                color='#00bfff',
                width=2,
                dash='dot'
            ),
            name='Harmonic Pattern'
        ),
        row=1, col=2
    )
    
    # Plot 3: Phi-based Network
    nodes_x = [phi**i * np.cos(phi*i) for i in range(10)]
    nodes_y = [phi**i * np.sin(phi*i) for i in range(10)]
    fig.add_trace(
        go.Scatter(
            x=nodes_x, y=nodes_y,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=np.linspace(0, 1, 10),
                colorscale='Viridis',
                showscale=True
            ),
            name='Network Nodes'
        ),
        row=2, col=1
    )
    
    # Plot 4: Recursive Subdivision
    def generate_subdivision(depth, x0, y0, size):
        if depth == 0:
            return [], []
        x = [x0, x0+size/phi, x0+size/phi, x0, x0]
        y = [y0, y0, y0+size/phi**2, y0+size/phi**2, y0]
        x_rec, y_rec = generate_subdivision(depth-1, x0+size/phi, y0, size/phi)
        return x + x_rec, y + y_rec
    
    x_sub, y_sub = generate_subdivision(5, -1, -1, 2)
    fig.add_trace(
        go.Scatter(
            x=x_sub, y=y_sub,
            mode='lines',
            line=dict(color='#00bfff', width=1),
            name='Recursive Pattern'
        ),
        row=2, col=2
    )
    
    # Update layout with enhanced styling
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,10,30,0.9)',
        plot_bgcolor='rgba(0,10,30,0.9)',
        height=800,
        showlegend=True,
        title=dict(
            text=f"Golden Ratio (φ ≈ {phi:.8f}) Manifestations",
            font=dict(size=24, color='#00bfff')
        )
    )
    
    return fig

def create_enhanced_metagaming_visualization(key):
    """
    Creates an advanced network visualization for metagaming with 
    dynamic community detection and emergence patterns.
    """
    # Generate complex network structure
    n_nodes = 50
    positions = {}
    layers = [5, 10, 15, 20]  # Nodes per layer
    current_node = 0
    
    # Calculate positions in concentric circles
    for layer_idx, layer_size in enumerate(layers):
        radius = (layer_idx + 1) * 0.2
        for i in range(layer_size):
            angle = (2 * np.pi * i) / layer_size
            positions[current_node] = {
                'x': radius * np.cos(angle),
                'y': radius * np.sin(angle),
                'layer': layer_idx
            }
            current_node += 1
    
    # Generate edges with phi-based probability
    edges = []
    edge_weights = []
    phi = (1 + np.sqrt(5)) / 2
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Probability of connection based on golden ratio and layer difference
            layer_diff = abs(positions[i]['layer'] - positions[j]['layer'])
            prob = 1 / (phi ** layer_diff)
            
            if np.random.random() < prob:
                edges.append((i, j))
                edge_weights.append(prob)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges with dynamic width and opacity
    edge_x = []
    edge_y = []
    for edge, weight in zip(edges, edge_weights):
        x0, y0 = positions[edge[0]]['x'], positions[edge[0]]['y']
        x1, y1 = positions[edge[1]]['x'], positions[edge[1]]['y']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(
                width=weight * 3,
                color=f'rgba(107,193,255,{weight})'
            ),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes with dynamic size and color
    node_colors = [
        np.exp(-positions[i]['layer']/phi) 
        for i in range(n_nodes)
    ]
    
    fig.add_trace(go.Scatter(
        x=[pos['x'] for pos in positions.values()],
        y=[pos['y'] for pos in positions.values()],
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Node Influence',
                titleside='right'
            )
        ),
        text=[f'Node {i}' for i in range(n_nodes)],
        textposition='bottom center',
        hoverinfo='text'
    ))
    
    # Update layout with enhanced styling
    fig.update_layout(
        title='Emergent Metagaming Networks: Self-Organizing Complexity',
        title_x=0.5,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(25,25,112,0.3)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=800,
        annotations=[
            dict(
                text=f"φ-based Connection Probability",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                font=dict(size=14, color='#00bfff')
            )
        ]
    )
    
    return fig
    
# --- Main Visualization Function ---
def display_visualizations(data, forecasts, regions, selected_scenario, selected_regions, selected_time_range):
    """
    Displays the core visualizations in a structured, professional layout.
    """

    # Filter data based on selected time range
    filtered_data = data[(data['ds'] >= selected_time_range[0]) & (data['ds'] <= selected_time_range[1])].copy()

    # --- Section 1: Advanced Forecast Visualization ---
    st.header("Advanced Forecast Visualization")
    st.markdown("""
        Explore adoption trajectories under varying scenarios. These forecasts highlight the evolution of 1+1=1 
        across realistic, optimistic, and pessimistic contexts, helping leaders anticipate global trends.
    """)
    forecast_fig = create_advanced_forecast_visualization(data, forecasts, selected_scenario)
    st.plotly_chart(forecast_fig, use_container_width=True)

    # --- Section 2: Dimensional Analysis ---
    st.header("Dimensional Analysis")
    st.markdown("""
        Delve into the individual dimensions driving the evolution of 1+1=1. Analyze trends in cultural, scientific, 
        philosophical, and technological domains to uncover key growth dynamics.
    """)
    selected_dimension = st.selectbox(
        "Select Dimension for Analysis",
        [d for d in forecasts if 'overall' not in d and selected_scenario.lower() in d],
        index=0
    )
    if selected_dimension:
        fig_components = create_components_visualization(
            forecasts[selected_dimension],
            selected_dimension.split('_')[0].title(),
            key=f"component_plot_{selected_dimension}"
        )
        st.plotly_chart(fig_components, use_container_width=True, key=f"component_plot_{selected_dimension}")

    # --- Section 3: Netherlands Map Visualization ---
    st.header("Emergence Hub: Netherlands")
    st.markdown("""
        Explore the geographic spread of 1+1=1, focusing on the Netherlands as an initial emergence hub. 
        Observe diffusion patterns across urban centers and regions of influence.
    """)
    fig_netherlands = create_netherlands_map(data)
    st.plotly_chart(fig_netherlands, use_container_width=True, key="netherlands_map")

    # --- Section 4: MetaBro Cheat Code ---
    if cheat_code_input == metabro_cheat_code:
        st.markdown(
            """
            <div style="
                background: linear-gradient(90deg, #0073e6, #1f8fe5, #4aa3f5);
                border-radius: 10px;
                padding: 20px;
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 4px 10px rgba(0, 115, 230, 0.4);
                text-align: center;">
                MetaBro Cheat Code Activated! Secrets Unlocked.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Section 5: Golden Ratio Visualization ---
        st.header("Esoteric Echoes: Golden Ratio")
        st.markdown("""
            Witness the profound interplay between the golden ratio (φ ≈ 1.618) and the emergence of unity consciousness. 
            These patterns reflect the fundamental organizing principles of reality itself.
        """)
        fig_golden = create_golden_ratio_visualization(key="golden_ratio_plot")
        st.plotly_chart(fig_golden, use_container_width=True, key="golden_ratio_main")

        st.markdown("### The Mathematics of Unity")
        st.markdown("""
            The golden ratio manifests as a bridge between individual and collective consciousness, 
            revealing the fractal nature of reality where 1+1 transcends simple addition to create 
            emergent unity. Each spiral represents a layer of consciousness integration.
        """)

        # --- Section 6: Metagaming Visualization ---
        st.header("Metagaming Networks: Self-Organizing Reality")
        st.markdown("""
            Observe how individual nodes self-organize into coherent networks through φ-guided connections, 
            demonstrating the natural emergence of unity from apparent plurality. Each layer represents 
            a distinct level of conscious integration.
        """)
        fig_metagaming = create_enhanced_metagaming_visualization(key="metagaming_plot")
        st.plotly_chart(fig_metagaming, use_container_width=True, key="metagaming_main")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
selected_regions = st.sidebar.multiselect("Explore Regions", regions, default=regions[:2])
selected_scenario = st.sidebar.selectbox("Forecast Scenario", ["Realistic", "Optimistic", "Pessimistic"], index=0)

# Time Range Slider
min_date = data['ds'].min()
max_date = data['ds'].max()
selected_time_range = st.sidebar.slider(
    "Select Time Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM"
)

# MetaBro Cheat Code Input
cheat_code_input = st.sidebar.text_input("MetaBro Unlock Code", type="password")

# --- Main Dashboard ---
st.title("Visualizing 1+1=1's Evolution")
st.markdown("""
    This dashboard explores the adoption and expansion of the 1+1=1 principle, charting its impact across cultural, 
    scientific, and technological dimensions. Leveraging advanced econometric modeling, we project its evolution 
    under varying scenarios and contexts. Together, we navigate the emergence of a unified global consciousness.
""")

# --- Display Visualizations ---
display_visualizations(data, forecasts, regions, selected_scenario, selected_regions, selected_time_range)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #778899;'>Watch how 1+1=1 evolves over time, shaping the future of unity.</p>", unsafe_allow_html=True)

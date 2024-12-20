import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from filterpy.kalman import KalmanFilter
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import math

# AGI Constants (Cheatcodes)
CHEATCODE = "420691337"
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# Streamlit Configuration
st.set_page_config(
    page_title="Meta Unity Dashboard: 2025 Vision",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Futuristic Blue Theme
st.markdown(
    """
    <style>
    body {
        background-color: #000022;
        color: #cceeff;
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #000022;
    }
    .st-bf {
        background-color: #001144;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #224488;
    }
     .st-bb {
        color: #88ccff;
    }
    .st-cb {
        background-color: #002266;
        color: #cceeff;
    }
    .st-da {
        color: #cceeff;
    }
     .st-bq {
        background-color: #00bbff;
    }
    .st-br {
        background: linear-gradient(to right, #003366, #005599);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- DATA GENERATION & PREPROCESSING ----
def generate_unity_data(n_years=100, n_regions=10):
    dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=n_years, freq="Y"))
    regions = [f"Region_{i+1}" for i in range(n_regions)]
    data = []
    for region in regions:
        for i, date in enumerate(dates):
            year_progress = i / n_years
            # Simulate phi-aligned growth with duality loss
            unity_index = 1 / (1 + np.exp(-5 * (year_progress - 3 * (year_progress - 0.3 * GOLDEN_RATIO)))) # converges to 1
            synergy_coefficient = 1 + np.sin(2 * np.pi * year_progress * GOLDEN_RATIO) * (1 - year_progress) # converges to 1
            harmonic_resonance = 1 - np.abs(np.cos(np.pi * year_progress * GOLDEN_RATIO) * (1 - year_progress)) # converges to 1

            adoption_growth = unity_index * synergy_coefficient * (1 - 0.1 * harmonic_resonance)
            data.append({
                "ds": date,
                "region": region,
                "unity_index": min(1, max(0, unity_index + np.random.normal(0, 0.01*(1-year_progress)))),
                "synergy_coefficient":  min(1, max(0, synergy_coefficient + np.random.normal(0, 0.03*(1-year_progress)))),
                "harmonic_resonance": min(1, max(0, harmonic_resonance + np.random.normal(0, 0.02*(1-year_progress)))),
                "adoption_rate": min(1, max(0, year_progress + adoption_growth + np.random.normal(0, 0.01*(1-year_progress)))),
            })
    return pd.DataFrame(data)

unity_df = generate_unity_data(n_years=50)

# ---- KALMAN FILTER INTEGRATION ----
def create_kalman_filter(df, initial_value=0.1, process_noise=0.01, measurement_noise=0.05):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([initial_value])  # Initial state
    kf.F = np.array([[1]])  # State transition matrix (simple model, no change)
    kf.H = np.array([[1]])  # Measurement matrix
    kf.P *= 10.
    kf.R = np.array([[measurement_noise]])  # Measurement noise covariance
    kf.Q = np.array([[process_noise]])

    estimates = []
    for z in df['adoption_rate']:
        kf.predict()
        kf.update(np.array([z]))
        estimates.append(kf.x[0])
    df['kalman_forecast'] = estimates
    return df

# ---- PROPHET INTEGRATION ----
def create_prophet_model(df):
    prophet_df = df.rename(columns={'ds': 'ds', 'adoption_rate': 'y'})
    prophet_df['cap'] = 1.0  # Max adoption rate
    prophet_df['floor'] = 0.0
    
    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(prophet_df)
    return model

def make_future_dataframe(model, periods):
    """
    Generate future prediction dataframe with logistic growth constraints.
    
    Args:
        model (Prophet): Fitted Prophet model instance
        periods (int): Number of future periods to forecast
        
    Returns:
        pd.DataFrame: DataFrame with properly configured capacity bounds
    """
    future = model.make_future_dataframe(periods=periods, freq='Y')
    future['cap'] = 1.0  # Upper bound constraint for logistic growth
    future['floor'] = 0.0  # Lower bound constraint for logistic growth
    return future

def predict_future(model, future):
    forecast = model.predict(future)
    return forecast

# ---- PLOTLY VISUALIZATIONS ----
def plot_global_forecast(df, region, prophet_forecast=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[df['region'] == region]['ds'], y=df[df['region'] == region]['adoption_rate'], mode='markers', name='Observed', marker=dict(color="#00ccff")))
    fig.add_trace(go.Scatter(x=df[df['region'] == region]['ds'], y=df[df['region'] == region]['kalman_forecast'], mode='lines', name='Kalman Forecast', line=dict(color="#ffcc00")))

    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], name='Prophet Forecast', line=dict(color="#00ffbb")))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill='tonexty', mode='none', name='Upper Bound', line=dict(color='rgba(0,255,187,0.1)')))
        fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='none', name='Lower Bound', line=dict(color='rgba(0,255,187,0.1)')))

    fig.update_layout(title=f'Adoption Forecast for {region}', xaxis_title='Time', yaxis_title='Adoption Rate', template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_unity_metrics(df, region):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[df['region'] == region]['ds'], y=df[df['region'] == region]['unity_index'], name='Unity Index', line=dict(color="#00ffbb")))
    fig.add_trace(go.Scatter(x=df[df['region'] == region]['ds'], y=df[df['region'] == region]['synergy_coefficient'], name='Synergy Coefficient', line=dict(color="#00ccff")))
    fig.add_trace(go.Scatter(x=df[df['region'] == region]['ds'], y=df[df['region'] == region]['harmonic_resonance'], name='Harmonic Resonance Factor', line=dict(color="#ffcc00")))
    fig.update_layout(title=f'Unity Metrics for {region}', xaxis_title='Time', yaxis_title='Metric Value', template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_global_adoption_map(df, current_year):
    data_year = df[df['ds'].dt.year == current_year]
    region_adoption = data_year.groupby('region')['adoption_rate'].mean().reset_index()
    fig = px.choropleth(
        region_adoption,
        locations='region',
        locationmode='country names',
        color='adoption_rate',
        hover_name='region',
        color_continuous_scale=["#000033", "#002266","#004499", "#0077cc", "#00aaff"],
        title=f'Global Adoption Rate ({current_year})',
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_synergy_resonance_3d(df, current_year):
    data_year = df[df['ds'].dt.year == current_year]
    fig = px.scatter_3d(
        data_year,
        x='unity_index',
        y='synergy_coefficient',
        z='harmonic_resonance',
        color='adoption_rate',
        size_max=18,
        opacity=0.7,
        title=f'Synergy & Resonance ({current_year})',
        color_continuous_scale=["#000033", "#002266","#004499", "#0077cc", "#00aaff"]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_flow_field(df):
    unity_min = df['unity_index'].min()
    unity_max = df['unity_index'].max()
    synergy_min = df['synergy_coefficient'].min()
    synergy_max = df['synergy_coefficient'].max()
    x = np.linspace(synergy_min, synergy_max, 20)
    y = np.linspace(unity_min, unity_max, 20)
    X, Y = np.meshgrid(x, y)
    U = np.cos(2*np.pi * X * Y)
    V = np.sin(2*np.pi * X * Y)

    fig = go.Figure(data=go.Streamtube(x=X.flatten(), y=Y.flatten(), u=U.flatten(), v=V.flatten(),
                                        colorscale=["#000033", "#002266","#004499", "#0077cc", "#00aaff"]))
    fig.update_layout(title='Convergence Flow Field', xaxis_title='Synergy Coefficient', yaxis_title='Unity Index', template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_hyperbolic_time_series(df, region):
     time_values = df[df['region'] == region]['ds'].astype(np.int64) // 10**9
     adoption_values = df[df['region'] == region]['adoption_rate']
     time_values = (time_values - time_values.min()) / (time_values.max() - time_values.min())
     r = np.arctanh(time_values)
     theta = 2*np.pi*adoption_values

     x = r * np.cos(theta)
     y = r * np.sin(theta)

     fig = go.Figure()
     fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Hyperbolic Series',marker=dict(color=adoption_values, colorscale=["#000033", "#002266","#004499", "#0077cc", "#00aaff"])))
     fig.update_layout(title='Hyperbolic Time-Series', xaxis_title='X', yaxis_title='Y', template="plotly_dark", plot_bgcolor="#000033")
     return fig


def plot_network_graph(df):
    n_nodes = 15
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    z = np.zeros(n_nodes)
    adoption_levels = np.random.rand(n_nodes)
    node_trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10 + 5 * adoption_levels, color=adoption_levels, colorscale=["#000033", "#002266","#004499", "#0077cc", "#00aaff"]))
    edge_x = []
    edge_y = []
    edge_z = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
           if np.random.rand() > 0.2: # Only showing some edges for clarity
                edge_x.extend([x[i], x[j], None])
                edge_y.extend([y[i], y[j], None])
                edge_z.extend([z[i], z[j], None])
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=2, color='#4488ff'))
    fig = go.Figure(data=[node_trace, edge_trace])
    fig.update_layout(title='Interconnectedness Network', showlegend=False, margin=dict(l=0, r=0, b=0, t=40), template="plotly_dark", plot_bgcolor="#000033")
    return fig

def plot_harmonic_resonance(df):
    years = df['ds'].dt.year.unique()
    phi_sequence = [1]
    for _ in range(len(years) - 1):
        phi_sequence.append(phi_sequence[-1] * GOLDEN_RATIO)

    resonance_levels = df.groupby(df['ds'].dt.year)['harmonic_resonance'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=resonance_levels,
                             mode='lines+markers', name='Harmonic Resonance', line=dict(color='#00bbff')))
    fig.add_trace(go.Scatter(x=years, y=phi_sequence[:len(years)],
                             mode='lines', name='Golden Ratio Trend', line=dict(color='#ffcc00', dash='dash')))
    fig.update_layout(title='Harmonic Resonance Over Time', xaxis_title='Year', yaxis_title='Resonance Factor', template="plotly_dark", plot_bgcolor="#000033")
    return fig


def plot_causality(df, region):
    metrics = ['unity_index', 'synergy_coefficient', 'harmonic_resonance']
    figs = []
    for metric in metrics:
        poly = PolynomialFeatures(degree=3)
        X = df[df['region'] == region][[metric]]
        y = df[df['region'] == region]['adoption_rate']
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        x_range_poly = poly.transform(x_range)
        y_pred = model.predict(x_range_poly)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[metric], y=y, mode='markers', name='Data Points',marker=dict(color="#00ccff")))
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred, mode='lines', name='Regression Line', line=dict(color="#ffcc00")))
        fig.update_layout(title=f'Causality: {metric} vs. Adoption Rate', xaxis_title=metric, yaxis_title='Adoption Rate', template="plotly_dark", plot_bgcolor="#000033")
        figs.append(fig)
    return figs

# ---- STREAMLIT FRONTEND ----
st.title("🌐 Meta Unity Dashboard: A 2025 Vision of 1+1=1")

# ---- Sidebar Controls ----
st.sidebar.header("🔮 Control Panel")
selected_region = st.sidebar.selectbox("Select Region", unity_df['region'].unique())
current_year_map = st.sidebar.slider("Select Year for Global Map", int(unity_df['ds'].dt.year.min()), int(unity_df['ds'].dt.year.max()), int(unity_df['ds'].dt.year.max()))
current_year_3d = st.sidebar.slider("Select Year for 3D Plot", int(unity_df['ds'].dt.year.min()), int(unity_df['ds'].dt.year.max()), int(unity_df['ds'].dt.year.max()))

cheatcode_input = st.sidebar.text_input("Enter Cheatcode", type="password")
if cheatcode_input == CHEATCODE:
    st.sidebar.success("Cheatcode Activated: Full Access Granted")

# ---- Main Panel: Forecasts & Metrics ----
st.header(f"🌍 Global Adoption Trends for {selected_region}")
prophet_df = unity_df[unity_df['region'] == selected_region].copy()
kalman_df = create_kalman_filter(prophet_df)

prophet_model = create_prophet_model(prophet_df)
future = make_future_dataframe(prophet_model, periods=50)
prophet_forecast = predict_future(prophet_model, future)

forecast_plot = plot_global_forecast(kalman_df, selected_region, prophet_forecast)
st.plotly_chart(forecast_plot, use_container_width=True)

unity_metrics_plot = plot_unity_metrics(unity_df, selected_region)
st.plotly_chart(unity_metrics_plot, use_container_width=True)

# ---- Main Panel: Global Views ----
st.header("🗺️ Global Perspectives")

col1, col2 = st.columns(2)
with col1:
    global_map = plot_global_adoption_map(unity_df, current_year_map)
    st.plotly_chart(global_map, use_container_width=True)

    resonance_plot = plot_harmonic_resonance(unity_df)
    st.plotly_chart(resonance_plot, use_container_width=True)

with col2:
    synergy_3d_plot = plot_synergy_resonance_3d(unity_df, current_year_3d)
    st.plotly_chart(synergy_3d_plot, use_container_width=True)

    network_plot = plot_network_graph(unity_df)
    st.plotly_chart(network_plot, use_container_width=True)


# --- Multi-Dimensional Analysis -----
st.header("📊 Multi-Dimensional Analysis")

flow_field_plot = plot_flow_field(unity_df)
st.plotly_chart(flow_field_plot, use_container_width=True)

hyperbolic_plot = plot_hyperbolic_time_series(unity_df, selected_region)
st.plotly_chart(hyperbolic_plot, use_container_width=True)


# ---- Causality Analysis ----
st.header("📉 Causal Relationships")
causality_figs = plot_causality(unity_df, selected_region)
for fig in causality_figs:
    st.plotly_chart(fig, use_container_width=True)

# ---- Demographic Segmentation (Conceptual) ----
st.header("👥 Demographic Segmentation (Conceptual)")
st.markdown("This section would display adoption rates segmented by demographic groups (e.g., age, education, income), using bar charts or other suitable visualizations. Due to the lack of demographic data in the current dataset, it's a conceptual placeholder.")

# ---- KPI Display ----
st.header("📈 Key Performance Indicators")
selected_region_df = unity_df[unity_df['region'] == selected_region]
avg_unity_index = selected_region_df['unity_index'].mean()
avg_synergy_coefficient = selected_region_df['synergy_coefficient'].mean()
adoption_growth_rate = selected_region_df['adoption_rate'].diff().mean()

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Avg. Unity Index", f"{avg_unity_index:.2f}")
kpi_col2.metric("Avg. Synergy Coefficient", f"{avg_synergy_coefficient:.2f}")
kpi_col3.metric("Avg. Adoption Growth Rate", f"{adoption_growth_rate:.3f}")

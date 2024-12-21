import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from datetime import datetime, timedelta
import random
import time
import json
from collections import deque
from enum import Enum
from typing import List, Dict, Tuple

# --- Constants for Visual Styling ---
PRIMARY_COLOR = '#0a192f'
SECONDARY_COLOR = '#0073e6'
ACCENT_COLOR = '#64ffda'
WARNING_COLOR = '#ff6b6b'
TEXT_COLOR = '#c0c0c0' 
SUCCESS_COLOR = '#2E8B57'
FONT_FAMILY = 'Arial, sans-serif'
GRAPH_BG_COLOR = '#162946'

# --- Enums for Quest Types ---
class QuestType(Enum):
    SOCIAL = "Social"
    PERSONAL = "Personal"
    GLOBAL = "Global"
    ECONOMIC = "Economic"

# --- Data Simulation ---
class DataSimulator:
    def __init__(self, start_date, days):
        self.start_date = start_date
        self.days = days
        self.rng = np.random.default_rng()

    def _create_dates(self) -> List[datetime]:
         return [self.start_date + timedelta(days=i) for i in range(self.days)]

    def _generate_base_trend(self, start=0.5, end=0.9) -> np.ndarray:
       return np.linspace(start, end, self.days)

    def _generate_noise(self, scale=0.05) -> np.ndarray:
       return self.rng.normal(0, scale, self.days)

    def _apply_event_impact(self, data: np.ndarray, events: List[Tuple[int, float]]) -> np.ndarray:
        for event_day, impact in events:
             if event_day < len(data):
                  data[event_day] += impact
        return np.clip(data, 0, 1)

    def _generate_memetic_spread(self) -> np.ndarray:
        memetic_spread = np.zeros(self.days)
        for i in range(self.days):
            if i > 0:
                memetic_spread[i] = memetic_spread[i-1] + (0.1 * (1 - memetic_spread[i-1]) * (self.rng.random() - 0.3))
            if i == 45:
                memetic_spread[i] = 0.5
        return np.clip(memetic_spread, 0, 1)

    def generate_data(self) -> pd.DataFrame:
        dates = self._create_dates()
        base_trend = self._generate_base_trend()

        global_resonance = self._apply_event_impact(base_trend + self._generate_noise(0.05), [(30, 0.15), (60, -0.1), (100, 0.2)])
        cooperation_index = self._apply_event_impact(base_trend + self._generate_noise(0.08), [(40, 0.10), (80, -0.12), (150, 0.18)])
        social_cohesion = self._apply_event_impact(base_trend + self._generate_noise(0.07), [(20, 0.05), (70, -0.11), (120, 0.13)])
        economic_alignment = self._apply_event_impact(base_trend + self._generate_noise(0.06), [(50, 0.13), (90, -0.08), (130, 0.19)])
        memetic_spread = self._generate_memetic_spread()

        personal_resonance = self.rng.uniform(0.6, 0.95, size=(self.days, 5))
        social_media_sentiment = self._apply_event_impact(base_trend + self._generate_noise(0.1), [(5, 0.1), (40, -0.2), (110, 0.2)])
        global_event_score = self._apply_event_impact(base_trend + self._generate_noise(0.04), [(60, 0.1), (100, -0.1), (200, 0.1)])

        # Add correlation between Social Media Sentiment and Global Event Score
        correlation_factor = 0.5
        social_media_sentiment = social_media_sentiment + correlation_factor * (global_event_score - np.mean(global_event_score))
        social_media_sentiment = np.clip(social_media_sentiment, 0, 1)


        df = pd.DataFrame({
            'date': dates,
            'Global Resonance': global_resonance,
            'Cooperation Index': cooperation_index,
            'Social Cohesion': social_cohesion,
            'Economic Alignment': economic_alignment,
            'Memetic Spread': memetic_spread,
            'Social Media Sentiment': social_media_sentiment,
            'Global Event Score': global_event_score
        })
        for i in range(5):
            df[f'Personal Resonance {i+1}'] = personal_resonance[:, i]

        return df

# --- Prophet Forecasting ---
class ProphetModel:
    def __init__(self, growth='linear'):
        self.model = Prophet(growth=growth,
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                            )

    def fit_and_predict(self, df: pd.DataFrame, metric: str, periods: int = 60, growth_cap: float = None, floor: float = 0) -> pd.DataFrame:
        prophet_df = pd.DataFrame({'ds': df['date'], 'y': df[metric]})

        if growth_cap:
            prophet_df['cap'] = growth_cap
        if floor:
            prophet_df['floor'] = floor

        try:
          self.model.fit(prophet_df)
        except Exception as e:
            print(f"Prophet fit error {e}")
            return None

        future = self.model.make_future_dataframe(periods=periods)
        if growth_cap:
            future['cap'] = growth_cap
        if floor:
            future['floor'] = floor

        try:
            forecast = self.model.predict(future)
        except Exception as e:
            print(f"Prophet predict error {e}")
            return None
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def plot_changepoints(self, forecast: pd.DataFrame, ax) -> None:
        if self.model.changepoints is not None and len(self.model.changepoints) > 0:
            add_changepoints_to_plot(ax, self.model, forecast)

    def get_changepoints(self) -> np.ndarray:
        return self.model.changepoints

    def get_model(self):
      return self.model

# --- Metagaming Module ---
class MetagamingManager:
    def __init__(self):
        self.quests = []
        self.completed_quests = []
        self.quest_types_enabled = {quest_type: True for quest_type in QuestType}
        self.score = 0
        self.score_history = deque(maxlen=50)
        self.quest_difficulty = {quest_type: "normal" for quest_type in QuestType}
        self.quest_descriptions = {
            QuestType.SOCIAL: [
                "Attend a local community event.",
                "Organize a neighborhood cleanup.",
                "Volunteer at a social project.",
                 "Start a conversation with a stranger and listen to their story."
            ],
            QuestType.PERSONAL: [
                "Meditate for 15 minutes.",
                "Journal for 10 minutes on personal goals.",
                "Read a chapter from an inspiring book.",
                "Take a mindful walk in nature."
            ],
           QuestType.GLOBAL: [
                "Donate to a cause you believe in.",
                "Write a letter to a global leader advocating for change.",
                "Participate in a global awareness campaign.",
                 "Share a story of unity and cooperation on social media."
           ],
          QuestType.ECONOMIC: [
                "Support a local small business.",
                 "Research and share information on fair trade.",
                 "Reflect on personal financial habits and make positive adjustments.",
                 "Offer a service to someone in your community."
           ]
        }

    def generate_quests(self, df: pd.DataFrame) -> None:
          self.quests = []
          for quest_type in QuestType:
            if self.quest_types_enabled[quest_type] and len(self.quests) < 3:
              description = random.choice(self.quest_descriptions[quest_type])
              self.quests.append({'description': description, 'type': quest_type, 'completed': False, 'id':len(self.quests)})

    def complete_quest(self, quest_id: int) -> bool:
        for quest in self.quests:
           if quest['id'] == quest_id and not quest['completed']:
               quest['completed'] = True
               self.completed_quests.append(quest)
               difficulty = self.quest_difficulty[quest['type']]
               if difficulty == "easy":
                    self.score += 5
               elif difficulty == "normal":
                    self.score += 10
               elif difficulty == "hard":
                    self.score += 20
               self.score_history.append(self.score)
               return True
        return False
    def get_score(self) -> int:
        return self.score

    def get_score_history(self) -> deque:
        return self.score_history

    def get_quests(self) -> List[Dict]:
        return self.quests
    
    def get_completed_quests(self) -> List[Dict]:
        return self.completed_quests

    def toggle_quest_type(self, quest_type: QuestType, enabled: bool) -> None:
        if quest_type in self.quest_types_enabled:
            self.quest_types_enabled[quest_type] = enabled
    def get_enabled_quest_types(self) -> List[str]:
        return [quest_type.value for quest_type, enabled in self.quest_types_enabled.items() if enabled]
    def set_quest_difficulty(self, quest_type: QuestType, difficulty: str) -> None:
        if quest_type in self.quest_difficulty:
             self.quest_difficulty[quest_type] = difficulty
    def get_quest_difficulty(self) -> Dict[str, str]:
        return {quest_type.value: difficulty for quest_type, difficulty in self.quest_difficulty.items()}


# --- Dash Application ---
app = dash.Dash(__name__)
server = app.server

# Data Initialization
start_date = datetime(2022, 1, 1) # Initial Seed Date
days = 1000
data_simulator = DataSimulator(start_date, days)
df = data_simulator.generate_data()
forecast_periods = 120

# Initial Prophet Models
prophet_models = {}
metrics_for_forecast = ['Global Resonance', 'Memetic Spread', 'Social Media Sentiment', 'Global Event Score', 'Personal Resonance 1']
for metric in metrics_for_forecast:
    growth_cap = 1 if metric == 'Memetic Spread' else None # Using logistic growth for memetic spread
    prophet_models[metric] = ProphetModel(growth = 'logistic' if growth_cap else 'linear')

forecasts = {}
for metric in metrics_for_forecast:
    growth_cap = 1 if metric == 'Memetic Spread' else None
    forecasts[metric] = prophet_models[metric].fit_and_predict(df, metric, periods=forecast_periods, growth_cap = growth_cap)

# --- Global Variables and Data Structures ---
chart_update_interval = 60 # seconds
data_update_interval = 15 #seconds
live_data_buffer = {}
for metric in metrics_for_forecast:
  live_data_buffer[metric] = deque(maxlen=50)
  if forecasts.get(metric) is not None:
        live_data_buffer[metric].extend(forecasts[metric]['yhat'].tolist())

metagame_manager = MetagamingManager()

# --- Layout ---
app.layout = html.Div(
    style={'backgroundColor': PRIMARY_COLOR, 'color': TEXT_COLOR, 'fontFamily': FONT_FAMILY},
    children=[
        html.H1(children="Unity HUD 2025", style={'textAlign': 'center', 'padding': '20px', 'color': ACCENT_COLOR}),
        dcc.Tabs(id='main-tabs', value='core-metrics', style = {'color': TEXT_COLOR}, children=[
            dcc.Tab(label='Core Metrics', value='core-metrics', style={'color':TEXT_COLOR},
                    selected_style={'backgroundColor':SECONDARY_COLOR,'color':TEXT_COLOR}, children=[
                  html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                        # Current Metrics
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Global Resonance", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                 dcc.Graph(id='global-resonance-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                            ]),
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Cooperation Index", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                 dcc.Graph(id='cooperation-index-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                            ]),
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Social Cohesion", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                 dcc.Graph(id='social-cohesion-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                            ]),
                          html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Economic Alignment", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                 dcc.Graph(id='economic-alignment-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                            ]),
                       html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Memetic Spread", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                 dcc.Graph(id='memetic-spread-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                            ]),
                       html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                                html.H2("Personal Resonance Metrics", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                                *[html.P(f"Individual {i+1}: {df[f'Personal Resonance {i+1}'].iloc[-1]:.3f}", style={'color': TEXT_COLOR, 'fontSize': '1.2em', 'margin': '5px'}) for i in range(5)]
                            ]),
                  ]),
            ]),
            dcc.Tab(label='Forecasts & Analysis', value='forecasts-analysis',style={'color':TEXT_COLOR},selected_style={'backgroundColor':SECONDARY_COLOR,'color':TEXT_COLOR},children=[
              html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                  html.Label("Select Metric for Forecast", style={'color': TEXT_COLOR, 'fontSize': '1.2em'}),
                  dcc.Dropdown(
                      id='metric-selector',
                      options=[{'label': metric, 'value': metric} for metric in metrics_for_forecast],
                      value=metrics_for_forecast[0],
                      style={'color':'#000000'}
                      ),
                  ]),
              html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                      dcc.Graph(id='forecast-chart', style={'backgroundColor': '#0a192f', 'padding':'10px'})
                      ]),
                html.Div(style={'padding': '20px', 'margin': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                     # Scatter Plot
                    html.Div(style={'width': '45%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                        html.H3("Correlation Analysis", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        dcc.Graph(id='scatter-plot')
                    ]),

                    # Heatmap
                     html.Div(style={'width': '45%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                        html.H3("Metric Heatmap", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        dcc.Graph(id='heatmap')
                    ]),
                  ]),
          ]),
        dcc.Tab(label='Metagaming IRL', value='metagaming', style={'color': TEXT_COLOR}, selected_style={'backgroundColor': SECONDARY_COLOR, 'color': TEXT_COLOR}, children=[
             html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
               html.H2("Your Current Score:", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                html.Div(id='metagame-score', style = {'textAlign':'center', 'color': TEXT_COLOR, 'fontSize': '2em'}),
               dcc.Graph(id='score-history-graph', style={'backgroundColor': GRAPH_BG_COLOR, 'margin':'10px'})
            ]),
           html.Div(style={'padding':'20px', 'margin': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                 html.Div(style={'width': '60%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                         html.H3('Current Quests', style={'color': ACCENT_COLOR, 'textAlign':'center'}),
                         html.Ul(id='metagame-quests', style = {'color':TEXT_COLOR, 'padding':'10px'})
                    ]),
                  html.Div(style={'width': '30%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                        html.H3("Quest Settings", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        *[html.Div([
                             html.Label(f"{quest_type.value} Enabled", style = {'color': TEXT_COLOR, 'display':'block', 'margin':'10px 0px 5px 0px'}),
                            dcc.Checklist(
                            id = f'quest-type-toggle-{quest_type.value.lower()}',
                            options=[{'label': f'', 'value': quest_type.value}],
                            value=[quest_type.value] if metagame_manager.quest_types_enabled[quest_type] else [],
                            style={'color': TEXT_COLOR,  'display': 'inline-block', 'marginRight':'10px'}
                            ),
                            html.Label(f"Difficulty", style = {'color': TEXT_COLOR, 'display':'block', 'margin':'10px 0px 5px 0px'}),
                             dcc.Dropdown(
                                id = f"quest-difficulty-{quest_type.value.lower()}",
                                options = [{"label":"easy", "value": "easy"}, {"label":"normal", "value":"normal"},{"label":"hard", "value":"hard"}],
                                value = metagame_manager.quest_difficulty[quest_type],
                                style = {'color':'#000000'}
                            ),

                         ]) for quest_type in QuestType
                        ],
                    
                     
                     html.Div(id = 'quest-type-status', style = {'color':TEXT_COLOR, 'padding':'10px'})
                    ])
            ]),
            html.Div(style = {'margin':'10px', 'textAlign':'center'}, children = [
                 html.Progress(id='quest-progress-bar', value="0", max=100, style={'width':'80%', 'height': '20px', 'appearance': 'none'}),
             ])
        ]),
         dcc.Tab(label='Settings', value = 'settings',style={'color':TEXT_COLOR},selected_style={'backgroundColor':SECONDARY_COLOR,'color':TEXT_COLOR},children=[
             html.Div(style={'padding': '20px','margin': '10px', 'display':'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'},children=[
                 html.Div(style={'width': '30%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                        html.H3("Chart Update Interval", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        dcc.Slider(id='chart-interval-slider', min=10, max=120, step=10, value=chart_update_interval, marks={i: str(i) for i in range(10,130,20)},
                         ),
                        html.Div(id='chart-interval-output', style={'padding': '10px', 'color': TEXT_COLOR, 'textAlign':'center'})
                 ]),
                    html.Div(style={'width': '30%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                        html.H3("Data Update Interval", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        dcc.Slider(id='data-interval-slider', min=5, max=60, step=5, value=data_update_interval, marks={i: str(i) for i in range(5,65,10)}),
                         html.Div(id='data-interval-output', style={'padding': '10px','color': TEXT_COLOR, 'textAlign':'center'})
                 ]),
             ]),
          ]),
      ]),
    dcc.Interval(id='chart-update-interval', interval=chart_update_interval*1000, n_intervals=0),
    dcc.Interval(id='data-update-interval', interval=data_update_interval*1000, n_intervals=0),
    html.Div(id='live-data-store', style={'display':'none'}, children = json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()})),  # Serializing deque to list for json
    html.Div(id='quest-data-store', style={'display':'none'}, children = json.dumps({
        'quests':metagame_manager.get_quests(),
        'completed': metagame_manager.get_completed_quests(),
         'score': metagame_manager.get_score(),
        'score_history': list(metagame_manager.get_score_history()),
        'difficulty': metagame_manager.get_quest_difficulty()
    })) # Hidden div for quest data
])

@app.callback(
    Output('chart-interval-output', 'children'),
    Input('chart-interval-slider', 'value')
)
def update_chart_interval_output(value):
  return f"Update Interval: {value} seconds"

@app.callback(
    Output('data-interval-output', 'children'),
    Input('data-interval-slider', 'value')
)
def update_data_interval_output(value):
    return f"Update Interval: {value} seconds"

@app.callback(
    Output('chart-update-interval', 'interval'),
    Input('chart-interval-slider', 'value')
)
def update_chart_update_interval(value):
  return value * 1000

@app.callback(
    Output('data-update-interval', 'interval'),
    Input('data-interval-slider', 'value')
)
def update_data_update_interval(value):
    return value * 1000

def create_core_metric_chart(df: pd.DataFrame, metric: str, color: str, initial_date: datetime) -> go.Figure:
    """Creates a line chart for a core metric with historical and projected values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df[metric], mode='lines+markers', name='Actual', marker=dict(size=3, color=color), line=dict(color=color, width=2)))
        # Calculate a simple projection based on last few datapoints
    projection_days = 60  # Forecast projection days
    last_date = df['date'].iloc[-1]
    projection_dates = [last_date + timedelta(days=i) for i in range(1, projection_days + 1)]
    projection_values = np.linspace(df[metric].iloc[-1], df[metric].iloc[-1] + (df[metric].iloc[-1] - df[metric].iloc[-50])/50 , projection_days) #  Assume the last 50 points to generate the linear trend
    fig.add_trace(go.Scatter(x=projection_dates, y=projection_values, mode='lines+markers', name='Projected', marker=dict(size=3, color=color, opacity = 0.3), line=dict(color=color, width=2, dash = 'dash'), ))

    fig.update_layout(
       plot_bgcolor=GRAPH_BG_COLOR,
        paper_bgcolor=GRAPH_BG_COLOR,
        font_color=TEXT_COLOR,
          title = f"{metric} Evolution",
        xaxis_title='Date',
        yaxis_title=metric,
        showlegend = False,
        hovermode = 'x unified'

    )
    return fig

@app.callback(
    Output('global-resonance-chart', 'figure'),
      Output('cooperation-index-chart', 'figure'),
      Output('social-cohesion-chart', 'figure'),
       Output('economic-alignment-chart', 'figure'),
        Output('memetic-spread-chart', 'figure'),
     [Input('data-update-interval', 'n_intervals')]
)
def update_core_metric_charts(n_intervals):
    initial_date = datetime(2022, 1, 1)
    return (
         create_core_metric_chart(df, 'Global Resonance', ACCENT_COLOR, initial_date),
        create_core_metric_chart(df, 'Cooperation Index', SUCCESS_COLOR, initial_date),
         create_core_metric_chart(df, 'Social Cohesion', WARNING_COLOR, initial_date),
         create_core_metric_chart(df, 'Economic Alignment', ACCENT_COLOR, initial_date),
         create_core_metric_chart(df, 'Memetic Spread', SECONDARY_COLOR, initial_date),
     )

@app.callback(
    Output('forecast-chart', 'figure'),
    [Input('metric-selector', 'value'),
     Input('chart-update-interval', 'n_intervals')],
     [State('live-data-store', 'children')]
)
def update_forecast_chart(selected_metric, n_intervals, live_data):
  if selected_metric is None:
    return {}
  if live_data is None:
      return {}

  try:
      live_data = json.loads(live_data)
      live_data_buffer = live_data.get(selected_metric)
      if live_data_buffer is None:
          return {}

      forecast = forecasts.get(selected_metric)
      if forecast is None:
           growth_cap = 1 if selected_metric == 'Memetic Spread' else None
           forecast = prophet_models[selected_metric].fit_and_predict(df, selected_metric, periods=forecast_periods, growth_cap = growth_cap)
           forecasts[selected_metric] = forecast

      fig = go.Figure()
      fig.add_trace(go.Scatter(x=df['date'], y=df[selected_metric], mode='lines+markers', name='Actual', marker=dict(size=5), line=dict(color=SECONDARY_COLOR, width=2)))
      fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color=ACCENT_COLOR, dash='dash', width=2)))
      fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color=WARNING_COLOR, width=0.5, dash='dot'), fill='tonexty', fillcolor='rgba(255, 107, 107, 0.2)'))
      fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color=WARNING_COLOR, width=0.5, dash='dot'), fill='tonexty', fillcolor='rgba(255, 107, 107, 0.2)'))

      fig.add_trace(go.Scatter(x=[df['date'].iloc[-1]+timedelta(seconds=x) for x in range(len(live_data_buffer))], y=list(live_data_buffer), mode='lines', name='Live Data', line=dict(color=SUCCESS_COLOR, width=2)))

      fig.update_layout
      fig.update_layout(
          plot_bgcolor=PRIMARY_COLOR,
          paper_bgcolor=PRIMARY_COLOR,
          font_color=TEXT_COLOR,
          title=f'Prophet Forecast for {selected_metric}',
          xaxis_title='Date',
          yaxis_title=selected_metric,
          legend_orientation="h",
          legend=dict(x=0, y=1.1),
          hovermode = 'x unified'

      )
      return fig
  except Exception as e:
    print(f"Error updating forecast chart: {e}")
    return {}

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('metric-selector', 'value'),
     Input('chart-update-interval', 'n_intervals')],
     [State('live-data-store', 'children')]
)
def update_scatter_plot(selected_metric, n_intervals, live_data):
    if selected_metric is None or live_data is None:
        return {}
    try:
      live_data = json.loads(live_data)
      if selected_metric == 'Global Resonance':
          x_data = df['Cooperation Index']
      elif selected_metric == 'Memetic Spread':
          x_data = df['Global Resonance']
      elif selected_metric == 'Personal Resonance 1':
          x_data = df['Memetic Spread']
      elif selected_metric == 'Social Media Sentiment':
            x_data = df['Global Event Score']
      elif selected_metric == 'Global Event Score':
             x_data = df['Social Media Sentiment']
      else:
          x_data = df['Global Resonance']


      fig = go.Figure(data=[go.Scatter(x=x_data, y=df[selected_metric], mode='markers',
          marker=dict(size=10, color=df[selected_metric], colorscale='Viridis', showscale=True),
          text=[f"{date.strftime('%Y-%m-%d')}" for date in df['date']],  # Hover text
          hovertemplate="<b>Date</b>: %{text}<br><b>X Value</b>: %{x}<br><b>Y Value</b>: %{y}<extra></extra>",
          )])

      fig.update_layout(
          plot_bgcolor=PRIMARY_COLOR,
          paper_bgcolor=PRIMARY_COLOR,
          font_color=TEXT_COLOR,
            xaxis_title= 'X-Axis Data',
          yaxis_title= selected_metric,
        )
      return fig
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        return {}

@app.callback(
    Output('heatmap', 'figure'),
    [Input('metric-selector', 'value'),
     Input('chart-update-interval', 'n_intervals')],
     [State('live-data-store', 'children')]
)
def update_heatmap(selected_metric, n_intervals, live_data):
    if selected_metric is None or live_data is None:
        return {}
    try:
      live_data = json.loads(live_data)
      metrics = ['Global Resonance', 'Cooperation Index', 'Social Cohesion', 'Economic Alignment','Memetic Spread', 'Social Media Sentiment', 'Global Event Score']
      if selected_metric in metrics:
        metrics.remove(selected_metric)
      else:
         metrics = metrics[:4]

      correlation_matrix = df[metrics].corr()
      annotations = []
      for i, row in enumerate(correlation_matrix.values):
          for j, val in enumerate(row):
              annotations.append(dict(x=correlation_matrix.columns[j], y=correlation_matrix.index[i], text=f'{val:.2f}', showarrow=False, font=dict(color='black')))

      fig = go.Figure(data=go.Heatmap(
          z=correlation_matrix.values,
          x=correlation_matrix.columns,
          y=correlation_matrix.index,
          colorscale='Viridis',
          text=correlation_matrix.values,
        ))
      fig.update_layout(
          plot_bgcolor=PRIMARY_COLOR,
          paper_bgcolor=PRIMARY_COLOR,
          font_color=TEXT_COLOR,
          annotations=annotations,
          xaxis_title='Metrics',
          yaxis_title='Metrics'
      )
      return fig
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return {}

@app.callback(
    Output('live-data-store', 'children'),
    [Input('data-update-interval', 'n_intervals')],
    [State('live-data-store', 'children')]
)
def update_live_data(n_intervals, live_data_str):
  try:
      if live_data_str is None:
        return json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()})
      live_data_buffer_data = json.loads(live_data_str)
      live_data_buffer = {metric: deque(buffer, maxlen=50) for metric, buffer in live_data_buffer_data.items()} # convert lists back to deques
      new_data = data_simulator.generate_data()
      for metric in metrics_for_forecast:
        if metric in new_data.columns:
          live_data_buffer[metric].append(new_data[metric].iloc[-1])
      return json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()}) # convert back to lists for serialization
  except Exception as e:
    print(f"Error updating live data buffer {e}")
    return json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()}) # Return backup in case of errors

@app.callback(
   Output('metagame-quests', 'children'),
    Output('metagame-score', 'children'),
     Output('score-history-graph', 'figure'),
    Output('quest-data-store', 'children'),
     Output('quest-progress-bar', 'value'),
      Output('quest-type-status', 'children'),
    [Input('data-update-interval', 'n_intervals')] +
     [Input({'type': 'complete-quest', 'index': dash.dependencies.ALL}, 'n_clicks')] +
     [Input(f'quest-type-toggle-{quest_type.value.lower()}', 'value') for quest_type in QuestType] +
     [Input(f"quest-difficulty-{quest_type.value.lower()}", 'value') for quest_type in QuestType],
    [State('quest-data-store', 'children')]
)
def update_metagame_ui(n_intervals, *args):
    quest_data = args[-1]
    values = args[1:-1] # Exclude n_interval and quest_data, includes values of checkbox and dropdown
    ctx = dash.callback_context
    if not ctx.triggered:
         if quest_data is None:
            return [], 0, {}, json.dumps({
                'quests': metagame_manager.get_quests(),
                'completed': metagame_manager.get_completed_quests(),
                 'score': metagame_manager.get_score(),
                'score_history': list(metagame_manager.get_score_history()),
                 'difficulty': metagame_manager.get_quest_difficulty()
            }), "0", ""
         quest_data = json.loads(quest_data)
         metagame_manager.quests = quest_data.get('quests', [])
         metagame_manager.completed_quests = quest_data.get('completed', [])
         metagame_manager.score = quest_data.get('score', 0)
         metagame_manager.score_history = deque(quest_data.get('score_history', []), maxlen=50)
         quest_difficulty = quest_data.get('difficulty', {})
         for quest_type in QuestType:
           metagame_manager.set_quest_difficulty(quest_type, quest_difficulty.get(quest_type.value, "normal"))
         return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "0", ""
    try:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'data-update-interval' in triggered_id:
            if quest_data is None:
              return [], 0, {},  json.dumps({
                  'quests': metagame_manager.get_quests(),
                  'completed': metagame_manager.get_completed_quests(),
                   'score': metagame_manager.get_score(),
                    'score_history': list(metagame_manager.get_score_history()),
                     'difficulty': metagame_manager.get_quest_difficulty()
            }), "0", ""
            quest_data = json.loads(quest_data)
            metagame_manager.quests = quest_data.get('quests', [])
            metagame_manager.completed_quests = quest_data.get('completed', [])
            metagame_manager.score = quest_data.get('score', 0)
            metagame_manager.score_history = deque(quest_data.get('score_history', []), maxlen=50)
            quest_difficulty = quest_data.get('difficulty', {})
            for quest_type in QuestType:
               metagame_manager.set_quest_difficulty(quest_type, quest_difficulty.get(quest_type.value, "normal"))
            metagame_manager.generate_quests(df)

        elif 'complete-quest' in triggered_id:
            if quest_data is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "0", ""

            quest_data = json.loads(quest_data)
            metagame_manager.quests = quest_data.get('quests', [])
            metagame_manager.completed_quests = quest_data.get('completed', [])
            metagame_manager.score = quest_data.get('score', 0)
            metagame_manager.score_history = deque(quest_data.get('score_history', []), maxlen=50)
            quest_difficulty = quest_data.get('difficulty', {})
            for quest_type in QuestType:
               metagame_manager.set_quest_difficulty(quest_type, quest_difficulty.get(quest_type.value, "normal"))
            button_id = json.loads(triggered_id)['index']
            metagame_manager.complete_quest(button_id)

        elif 'quest-type-toggle' in triggered_id or 'quest-difficulty' in triggered_id:
            if quest_data is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "0", ""
            quest_data = json.loads(quest_data)
            metagame_manager.quests = quest_data.get('quests', [])
            metagame_manager.completed_quests = quest_data.get('completed', [])
            metagame_manager.score = quest_data.get('score', 0)
            metagame_manager.score_history = deque(quest_data.get('score_history', []), maxlen=50)
            quest_difficulty = quest_data.get('difficulty', {})
            for idx, quest_type in enumerate(QuestType):
                 if f'quest-type-toggle-{quest_type.value.lower()}' in triggered_id:
                      metagame_manager.toggle_quest_type(quest_type, len(values[idx]) > 0)
                 if f'quest-difficulty-{quest_type.value.lower()}' in triggered_id:
                    metagame_manager.set_quest_difficulty(quest_type, values[len(QuestType) + idx])
            metagame_manager.generate_quests(df)

        quest_items = [html.Li([
             quest['description'],
              html.Button('Complete', id={'type': 'complete-quest', 'index': quest['id']},
                     style={'marginLeft': '10px', 'backgroundColor': SUCCESS_COLOR, 'color':TEXT_COLOR, 'border':'none', 'borderRadius':'3px', 'padding':'3px'}),
            ], style = {'marginBottom':'5px', 'border':'1px solid #0073e6', 'padding':'5px', 'borderRadius':'3px', 'textDecoration':'line-through' if quest.get('completed', False) else 'none'}
        ) for quest in metagame_manager.get_quests() ]

        progress = str(int(len(metagame_manager.get_completed_quests())/ len(metagame_manager.get_quests()) * 100)) if  metagame_manager.get_quests() else "0"
        score_fig = go.Figure(data= [go.Scatter(x = list(range(len(metagame_manager.get_score_history()))),
            y = list(metagame_manager.get_score_history()), mode = 'lines+markers', line = dict(color = ACCENT_COLOR))])
        score_fig.update_layout(
              plot_bgcolor=GRAPH_BG_COLOR,
              paper_bgcolor=GRAPH_BG_COLOR,
               font_color=TEXT_COLOR,
               xaxis_title = 'Quest Completion Sequence',
                yaxis_title = 'Score',
                 showlegend = False,
                  hovermode = 'x unified'
            )

        enabled_types_text = f"Enabled Quest Types: {', '.join(metagame_manager.get_enabled_quest_types())}"
        return quest_items, f"{metagame_manager.get_score()}", score_fig, json.dumps({
                'quests': metagame_manager.get_quests(),
                'completed': metagame_manager.get_completed_quests(),
                'score': metagame_manager.get_score(),
                'score_history': list(metagame_manager.get_score_history()),
                'difficulty': metagame_manager.get_quest_difficulty()
            }), progress, enabled_types_text
    except Exception as e:
        print(f"Error in combined metagame callback: {e}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "0", ""

# --- Run App ---
if __name__ == '__main__':
    app.run_server(debug=True)

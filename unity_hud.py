import dash
from dash import ALL
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
from typing import List, Dict, Tuple, Callable
from collections import defaultdict, deque

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

class UnityHUDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, QuestType):
            return obj.value
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, deque):
            return list(obj)
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

# --- Data Simulation ---
class DataSimulator:
    def __init__(self, start_date, days):
        self.start_date = start_date
        self.days = days
        self.rng = np.random.default_rng()
        self.event_history = []

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
                  self.event_history.append((self.start_date + timedelta(days=event_day), impact))
        return np.clip(data, 0, 1)

    def _generate_memetic_spread(self) -> np.ndarray:
        memetic_spread = np.zeros(self.days)
        for i in range(self.days):
            if i > 0:
                memetic_spread[i] = memetic_spread[i-1] + (0.1 * (1 - memetic_spread[i-1]) * (self.rng.random() - 0.3))
            if i == 45:
                memetic_spread[i] = 0.5
        return np.clip(memetic_spread, 0, 1)
    
    def apply_feedback_impact(self, data: np.ndarray, day:int, impact: float, max_impact_duration:int = 10) -> np.ndarray:
       """Applies impact from metagame, decay over time"""
       for i in range(max_impact_duration):
         index = day + i
         if index < len(data):
            decay = impact * (1 - (i / max_impact_duration))
            data[index] += decay
       return np.clip(data, 0, 1)
    
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
    
    def apply_metagame_impact(self, df: pd.DataFrame, completed_quests: List[Dict], current_day:int)-> pd.DataFrame:
        """Applies impact from the completed metagame quests to the dataframe"""
        #impact_scale = 0.02
        for quest in completed_quests:
            quest_type = quest.get("type", None)
            if quest_type == QuestType.SOCIAL:
                df['Social Cohesion'] = self.apply_feedback_impact(df['Social Cohesion'].values, current_day, 0.02)
            if quest_type == QuestType.ECONOMIC:
                 df['Economic Alignment'] = self.apply_feedback_impact(df['Economic Alignment'].values, current_day, 0.02)
            if quest_type == QuestType.GLOBAL:
                 df['Global Resonance'] = self.apply_feedback_impact(df['Global Resonance'].values, current_day, 0.01)
            if quest_type == QuestType.PERSONAL:
                 df['Memetic Spread'] = self.apply_feedback_impact(df['Memetic Spread'].values, current_day, 0.01)

        return df
    
    def apply_ai_disruptions(self, df: pd.DataFrame, current_day: int) -> pd.DataFrame:
        """Applies a disruption factor to the simulation based on a simple AI check"""
        if current_day % 30 == 0 and current_day > 0: # Basic AI action that happens every 30 days (random action in future versions)
            metric_to_impact = random.choice(['Global Resonance','Cooperation Index', 'Social Cohesion', 'Economic Alignment', 'Memetic Spread', 'Social Media Sentiment', 'Global Event Score' ])
            impact = self.rng.uniform(-0.03,-0.01)
            df[metric_to_impact] = self.apply_feedback_impact(df[metric_to_impact].values, current_day, impact, max_impact_duration=10)
            self.event_history.append((self.start_date + timedelta(days=current_day), f'AI Disruption: {metric_to_impact} impacted by {impact:.3f}'))
        return df
    
    def get_event_history(self):
        return self.event_history
    
    def clear_event_history(self):
        self.event_history = []

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
        self.quest_arcs = {
            "Introduction": {
                "description": "Starting the Journey...",
                  "quests":[
                        ("Complete a Social Quest","Encourages participation in the community"),
                        ("Complete a Personal Quest", "Focus on internal harmony")
                    ],
                   "completion_bonus": 10,
                "unlocked_by": None,
                "completed": False,
                 },
             "Community Building": {
                 "description":"Deepening the Connections...",
                  "quests":[
                        ("Complete 2 Social Quests","Strengthening social fabric"),
                        ("Complete 1 Economic Quest", "Bolstering local economy")
                      ],
                     "completion_bonus": 20,
                 "unlocked_by": "Introduction",
                  "completed":False,
              },
            "Global Impact":{
                "description":"Expanding Influence",
                 "quests":[
                    ("Complete 1 Global Quest","Extending your vision"),
                     ("Complete 2 Economic Quests", "Thinking about global economic systems")
                ],
                 "completion_bonus":30,
                "unlocked_by":"Community Building",
                 "completed":False
            }
        }
        self.current_quest_arc = "Introduction"
        self.lore_messages = []
        self.initialize_starting_quests()

    def initialize_starting_quests(self):
        """Initialize the first set of quests"""
        self.quests = [
            {
                'description': "Begin your journey: Meditate for 15 minutes",
                'type': QuestType.PERSONAL,
                'completed': False,
                'id': 0,
                'impact': 0.04,
                'context': "Starting the path to unity consciousness"
            },
            {
                'description': "Connect with community: Attend a local event",
                'type': QuestType.SOCIAL,
                'completed': False,
                'id': 1,
                'impact': 0.05,
                'context': "Building social resonance"
            }
        ]
        self.lore_messages.append("Welcome to your unity consciousness journey! Complete your first quests to begin.")

    def _get_available_quests(self) -> List[Tuple[str, str]]:
         """Returns a list of quests based on the current arc"""
         current_arc = self.quest_arcs.get(self.current_quest_arc, None)
         if current_arc is None:
            return []
         return current_arc.get("quests", [])
    
    def _generate_quest_description(self, quest_type: QuestType, metric: float = None) -> dict:
        """Enhanced quest generation with impact values"""
        quest_options = self.quest_descriptions[quest_type]
        selected_quest = random.choice(quest_options)
        description = selected_quest["text"]
        if metric is not None:
            description += f" (Current resonance: {metric:.2f})"
        return {
            "description": description,
            "impact": selected_quest["impact"]
        }
    
    def _generate_dynamic_quests(self, df: pd.DataFrame) -> List[Dict]:
        """Generates quests based on the arc and simulation state"""
        available_quests = self._get_available_quests()
        if not available_quests:
             return []
        
        generated_quests = []
        for quest_name, quest_context in available_quests:
            if "Social" in quest_name and self.quest_types_enabled[QuestType.SOCIAL]:
                description = self._generate_quest_description(QuestType.SOCIAL, df["Social Cohesion"].iloc[-1])
                generated_quests.append({'description': description, 'type': QuestType.SOCIAL, 'completed': False, 'id':len(generated_quests), 'context': quest_context})
            if "Personal" in quest_name and self.quest_types_enabled[QuestType.PERSONAL]:
                 description = self._generate_quest_description(QuestType.PERSONAL)
                 generated_quests.append({'description': description, 'type': QuestType.PERSONAL, 'completed': False, 'id':len(generated_quests), 'context': quest_context})
            if "Global" in quest_name and self.quest_types_enabled[QuestType.GLOBAL]:
                description = self._generate_quest_description(QuestType.GLOBAL, df["Global Resonance"].iloc[-1])
                generated_quests.append({'description': description, 'type': QuestType.GLOBAL, 'completed': False, 'id':len(generated_quests), 'context': quest_context})
            if "Economic" in quest_name and self.quest_types_enabled[QuestType.ECONOMIC]:
                description = self._generate_quest_description(QuestType.ECONOMIC, df["Economic Alignment"].iloc[-1])
                generated_quests.append({'description': description, 'type': QuestType.ECONOMIC, 'completed': False, 'id':len(generated_quests), 'context': quest_context})
        
        return generated_quests

    def generate_quests(self, df: pd.DataFrame) -> None:
          self.quests = []
          
          if not self.quest_arcs.get(self.current_quest_arc).get("completed", False):
             self.quests = self._generate_dynamic_quests(df)
          if not self.quests:
             self.quests.append({
                'description': "You have completed all current objectives.",
                'type': None,
                'completed':True,
                'id':0
             })

    def complete_quest(self, quest_id: int) -> bool:
        """Enhanced quest completion with immediate feedback"""
        for quest in self.quests:
            if quest['id'] == quest_id and not quest['completed']:
                quest['completed'] = True
                self.completed_quests.append(quest)
                
                # Calculate score based on difficulty and impact
                difficulty_multiplier = {
                    "easy": 0.7,
                    "normal": 1.0,
                    "hard": 1.5
                }
                
                base_score = 10
                quest_impact = quest.get('impact', 0.05)
                difficulty = self.quest_difficulty[quest['type']]
                
                score_gain = int(base_score * difficulty_multiplier[difficulty] * (1 + quest_impact))
                self.score += score_gain
                self.score_history.append(self.score)
                
                # Generate feedback message
                self.lore_messages.append(f"Quest completed! +{score_gain} points. Your unity consciousness grows stronger.")
                
                self._check_current_arc_completion()
                return True
        return False
    
    def _check_current_arc_completion(self):
        """Enhanced arc completion check with metric boosts"""
        current_arc_data = self.quest_arcs.get(self.current_quest_arc)
        if not current_arc_data:
            return

        quest_requirements = [q[0] for q in current_arc_data["quests"]]
        completed_requirements = set()
        
        for quest in self.completed_quests:
            for req in quest_requirements:
                if (
                    ("Social Quest" in req and quest['type'] == QuestType.SOCIAL) or
                    ("Personal Quest" in req and quest['type'] == QuestType.PERSONAL) or
                    ("Global Quest" in req and quest['type'] == QuestType.GLOBAL) or
                    ("Economic Quest" in req and quest['type'] == QuestType.ECONOMIC)
                ):
                    completed_requirements.add(req)

        if len(completed_requirements) >= len(quest_requirements):
            bonus = current_arc_data["completion_bonus"]
            self.score += bonus
            self.score_history.append(self.score)
            
            self.lore_messages.append(
                f"Congratulations! You've completed the {self.current_quest_arc} arc! "
                f"+{bonus} bonus points. New possibilities await..."
            )
            
            current_arc_data['completed'] = True
            
            # Find next arc
            next_arc = None
            for arc_name, arc_data in self.quest_arcs.items():
                if arc_data.get("unlocked_by") == self.current_quest_arc and not arc_data.get("completed", False):
                    next_arc = arc_name
                    break
                    
            if next_arc:
                self.current_quest_arc = next_arc
                self.lore_messages.append(f"The {next_arc} arc has begun! New challenges emerge...")
            else:
                self.lore_messages.append("You've reached the pinnacle of current challenges. Stay tuned for more...")
             
    
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
    
    def get_lore_messages(self) -> List[str]:
        messages = self.lore_messages
        self.lore_messages = []
        return messages
    
    def get_current_arc(self):
      return self.current_quest_arc
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
current_day = 0

# --- Layout ---
app.layout = html.Div(
    style={'backgroundColor': PRIMARY_COLOR, 'color': TEXT_COLOR, 'fontFamily': FONT_FAMILY},
    children=[
        html.H1(children="Unity HUD 2025", style={'textAlign': 'center', 'padding': '20px', 'color': ACCENT_COLOR}),
        dcc.Tabs(id='main-tabs', value='core-metrics', style={'color': TEXT_COLOR}, children=[
            # Core Metrics Tab
            dcc.Tab(label='Core Metrics', value='core-metrics', 
                style={'color': TEXT_COLOR},
                selected_style={'backgroundColor': SECONDARY_COLOR, 'color': TEXT_COLOR}, 
                children=[
                    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                        # Global Resonance
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Global Resonance", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            dcc.Graph(id='global-resonance-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                        ]),
                        # Cooperation Index
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Cooperation Index", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            dcc.Graph(id='cooperation-index-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                        ]),
                        # Social Cohesion
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Social Cohesion", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            dcc.Graph(id='social-cohesion-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                        ]),
                        # Economic Alignment
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Economic Alignment", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            dcc.Graph(id='economic-alignment-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                        ]),
                        # Memetic Spread
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Memetic Spread", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            dcc.Graph(id='memetic-spread-chart', style={'backgroundColor': GRAPH_BG_COLOR})
                        ]),
                        # Personal Resonance
                        html.Div(style={'width': '45%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H2("Personal Resonance Metrics", style={'color': ACCENT_COLOR, 'textAlign': 'center', 'padding': '10px'}),
                            *[html.P(f"Individual {i+1}: {df[f'Personal Resonance {i+1}'].iloc[-1]:.3f}", 
                                    style={'color': TEXT_COLOR, 'fontSize': '1.2em', 'margin': '5px'}) for i in range(5)]
                        ]),
                    ]),
                ]
            ),
            
            # Forecasts & Analysis Tab
            dcc.Tab(label='Forecasts & Analysis', value='forecasts-analysis',
                style={'color': TEXT_COLOR},
                selected_style={'backgroundColor': SECONDARY_COLOR, 'color': TEXT_COLOR},
                children=[
                    html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                        html.Label("Select Metric for Forecast", style={'color': TEXT_COLOR, 'fontSize': '1.2em'}),
                        dcc.Dropdown(
                            id='metric-selector',
                            options=[{'label': metric, 'value': metric} for metric in metrics_for_forecast],
                            value=metrics_for_forecast[0],
                            style={'color': '#000000'}
                        ),
                    ]),
                    html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                        dcc.Graph(id='forecast-chart', style={'backgroundColor': '#0a192f', 'padding': '10px'})
                    ]),
                    html.Div(style={'padding': '20px', 'margin': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                        html.Div(style={'width': '45%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H3("Correlation Analysis", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                            dcc.Graph(id='scatter-plot')
                        ]),
                        html.Div(style={'width': '45%', 'padding': '10px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H3("Metric Heatmap", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                            dcc.Graph(id='heatmap')
                        ]),
                    ]),
                ]
            ),
            
            # Enhanced Metagaming Tab
            dcc.Tab(label='Metagaming IRL', value='metagaming',
                style={'color': TEXT_COLOR},
                selected_style={'backgroundColor': SECONDARY_COLOR, 'color': TEXT_COLOR},
                children=[
                    html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                            html.H2("Unity Consciousness Progress", style={'color': ACCENT_COLOR}),
                            html.Div([
                                html.H3("Current Score:", style={'color': ACCENT_COLOR, 'marginBottom': '5px'}),
                                html.Div(id='metagame-score', style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontSize': '2em'})
                            ])
                        ]),
                        html.Div(style={'marginTop': '20px'}, children=[
                            dcc.Graph(id='score-history-graph', style={'backgroundColor': GRAPH_BG_COLOR, 'height': '200px'}),
                            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}, children=[
                                html.Progress(id='quest-progress-bar', value="0", max=100,
                                    style={'width': '80%', 'height': '20px', 'marginRight': '10px'}),
                                html.Div(id='progress-percentage', style={'color': TEXT_COLOR})
                            ])
                        ])
                    ]),
                    html.Div(style={'padding': '20px', 'margin': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                        html.Div(style={'width': '60%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.Div(style={'marginBottom': '20px'}, children=[
                                html.H3('Current Quest Arc', style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                                html.Div(id='current-arc-description', style={'color': TEXT_COLOR, 'textAlign': 'center', 'marginTop': '10px'})
                            ]),
                            html.Div(id='metagame-lore', style={'color': TEXT_COLOR, 'padding': '10px', 'marginBottom': '20px'}),
                            html.Div([
                                html.H3('Active Quests', style={'color': ACCENT_COLOR, 'marginBottom': '10px'}),
                                html.Ul(id='metagame-quests', style={'color': TEXT_COLOR, 'padding': '10px'})
                            ])
                        ]),
                        html.Div(style={'width': '30%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H3("Quest Settings", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                            *[html.Div([
                                html.Label(f"{quest_type.value} Quests", style={'color': TEXT_COLOR, 'display': 'block', 'marginTop': '10px'}),
                                dcc.Checklist(
                                    id=f'quest-type-toggle-{quest_type.value.lower()}',
                                    options=[{'label': '', 'value': quest_type.value}],
                                    value=[quest_type.value],
                                    style={'color': TEXT_COLOR, 'display': 'inline-block'}
                                ),
                                dcc.Dropdown(
                                    id=f"quest-difficulty-{quest_type.value.lower()}",
                                    options=[
                                        {"label": "Easy", "value": "easy"},
                                        {"label": "Normal", "value": "normal"},
                                        {"label": "Hard", "value": "hard"}
                                    ],
                                    value="normal",
                                    style={'color': '#000000', 'marginTop': '5px'}
                                )
                            ]) for quest_type in QuestType],
                            html.Div(id='quest-type-status', style={'color': TEXT_COLOR, 'padding': '10px', 'marginTop': '20px'})
                        ])
                    ]),
                    html.Div(style={'padding': '20px', 'margin': '10px'}, children=[
                        html.H3("Impact Visualization", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                        dcc.Graph(id='impact-visualization', style={'backgroundColor': GRAPH_BG_COLOR})
                    ])
                ]
            ),
            
            # Settings Tab
            dcc.Tab(label='Settings', value='settings',
                style={'color': TEXT_COLOR},
                selected_style={'backgroundColor': SECONDARY_COLOR, 'color': TEXT_COLOR},
                children=[
                    html.Div(style={'padding': '20px', 'margin': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}, children=[
                        html.Div(style={'width': '30%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H3("Chart Update Interval", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                            dcc.Slider(id='chart-interval-slider', min=10, max=120, step=10, value=chart_update_interval, 
                                     marks={i: str(i) for i in range(10, 130, 20)}),
                            html.Div(id='chart-interval-output', style={'padding': '10px', 'color': TEXT_COLOR, 'textAlign': 'center'})
                        ]),
                        html.Div(style={'width': '30%', 'padding': '20px', 'border': f'1px solid {SECONDARY_COLOR}', 'margin': '10px', 'borderRadius': '5px', 'backgroundColor': GRAPH_BG_COLOR}, children=[
                            html.H3("Data Update Interval", style={'color': ACCENT_COLOR, 'textAlign': 'center'}),
                            dcc.Slider(id='data-interval-slider', min=5, max=60, step=5, value=data_update_interval,
                                     marks={i: str(i) for i in range(5, 65, 10)}),
                            html.Div(id='data-interval-output', style={'padding': '10px', 'color': TEXT_COLOR, 'textAlign': 'center'})
                        ])
                    ])
                ]
            )
        ]),
        
        # System Components
        dcc.Interval(id='chart-update-interval', interval=chart_update_interval*1000, n_intervals=0),
        dcc.Interval(id='data-update-interval', interval=data_update_interval*1000, n_intervals=0),
        
        # State Management
        html.Div(id='live-data-store', style={'display': 'none'},
                children=json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()}, cls=UnityHUDEncoder)),
        html.Div(id='quest-data-store', style={'display': 'none'},
                children=json.dumps({
                    'quests': metagame_manager.get_quests(),
                    'completed': metagame_manager.get_completed_quests(),
                    'score': metagame_manager.get_score(),
                    'score_history': list(metagame_manager.get_score_history()),
                    'difficulty': metagame_manager.get_quest_difficulty(),
                    'current_arc': metagame_manager.get_current_arc()
                }, cls=UnityHUDEncoder)),
        html.Div(id='event-data-store', style={'display': 'none'},
                children=json.dumps(data_simulator.get_event_history(), cls=UnityHUDEncoder)),
        html.Div(id='ai-comment-output', style={'display': 'none'})
    ]
)

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
        return json.dumps({metric: list(buffer) for metric, buffer in live_data_buffer.items()}, cls=UnityHUDEncoder)
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
    [
        Output('metagame-quests', 'children'),
        Output('metagame-score', 'children'),
        Output('score-history-graph', 'figure'),
        Output('quest-data-store', 'children'),
        Output('quest-progress-bar', 'value'),
        Output('quest-type-status', 'children'),
        Output('ai-comment-output', 'children'),
        Output('event-data-store', 'children'),
        Output('metagame-lore', 'children'),
        Output('current-arc-description', 'children'),
        Output('progress-percentage', 'children'),
        Output('impact-visualization', 'figure')
    ],
    [Input('data-update-interval', 'n_intervals'),
     Input({'type': 'complete-quest', 'index': ALL}, 'n_clicks')],
    [State('quest-data-store', 'children'),
     State('event-data-store', 'children')]
)
def create_empty_score_graph():
    """Creates an empty score history visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', 
                            line=dict(color=ACCENT_COLOR)))
    fig.update_layout(
        plot_bgcolor=GRAPH_BG_COLOR,
        paper_bgcolor=GRAPH_BG_COLOR,
        font_color=TEXT_COLOR,
        title='Unity Score Progress',
        xaxis_title='Time',
        yaxis_title='Score',
        showlegend=False
    )
    return fig

def create_empty_impact_graph():
    """Creates an empty impact visualization"""
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=GRAPH_BG_COLOR,
        paper_bgcolor=GRAPH_BG_COLOR,
        font_color=TEXT_COLOR,
        title='Quest Impact on Unity Metrics',
        xaxis_title='Completed Quests',
        yaxis_title='Impact Magnitude',
        showlegend=True
    )
    return fig

def get_initial_quest_data():
    """Returns initial quest data structure"""
    return {
        'quests': [],
        'completed': [],
        'score': 0,
        'score_history': [],
        'difficulty': {qt.value: "normal" for qt in QuestType},
        'current_arc': "Introduction"
    }

def update_metagame_ui(n_intervals, *args):
    """
    Comprehensive metagame UI update handler with optimized state management
    """
    quest_data, event_data = args[-2:]  # Extract state data
    input_values = args[1:-2]  # Extract input values
    ctx = dash.callback_context
    global df, current_day

    try:
        # Initialize state if necessary
        if not ctx.triggered or quest_data is None:
            initial_state = initialize_metagame_state()
            return initial_state

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        quest_data = json.loads(quest_data)
        
        # Update manager state from stored data
        sync_manager_state(metagame_manager, quest_data)

        if 'data-update-interval' in triggered_id:
            # Process time-based updates
            process_interval_update(df, current_day)
            current_day += 1
            
        elif 'complete-quest' in triggered_id:
            # Handle quest completion
            button_id = json.loads(triggered_id)['index']
            process_quest_completion(metagame_manager, button_id)
            
        elif any(toggle in triggered_id for toggle in ['quest-type-toggle', 'quest-difficulty']):
            # Process settings changes
            process_settings_update(metagame_manager, input_values)

        # Generate updated UI components
        return generate_ui_components(metagame_manager, df)

    except Exception as e:
        print(f"Error in metagame callback: {e}")
        return [dash.no_update] * 12

def initialize_metagame_state():
    """Initialize the metagame state with default values"""
    return [
        [],  # quests
        0,   # score
        create_empty_score_graph(),
        json.dumps(get_initial_quest_data()),
        "0", # progress
        "",  # status
        "",  # AI comment
        json.dumps([]),  # events
        "",  # lore
        "Begin your journey...",  # arc description
        "0%",  # progress percentage
        create_empty_impact_graph()  # impact visualization
    ]

def sync_manager_state(manager, data):
    """Synchronize manager state with stored data"""
    manager.quests = data.get('quests', [])
    manager.completed_quests = data.get('completed', [])
    manager.score = data.get('score', 0)
    manager.score_history = deque(data.get('score_history', []), maxlen=50)
    quest_difficulty = data.get('difficulty', {})
    for quest_type in QuestType:
        manager.set_quest_difficulty(quest_type, quest_difficulty.get(quest_type.value, "normal"))

def process_interval_update(df, current_day):
    """Process time-based updates to the simulation"""
    df = data_simulator.apply_metagame_impact(df, metagame_manager.completed_quests, current_day)
    df = data_simulator.apply_ai_disruptions(df, current_day)
    metagame_manager.generate_quests(df)
    data_simulator.clear_event_history()

def process_quest_completion(manager, quest_id):
    """Handle quest completion and update state"""
    manager.complete_quest(quest_id)
    manager.generate_quests(df)  # Regenerate quests after completion

def process_settings_update(manager, values):
    """Process settings changes and update state"""
    for idx, quest_type in enumerate(QuestType):
        if idx < len(values) and values[idx] is not None:
            manager.toggle_quest_type(quest_type, len(values[idx]) > 0)
        difficulty_idx = len(QuestType) + idx
        if difficulty_idx < len(values) and values[difficulty_idx] is not None:
            manager.set_quest_difficulty(quest_type, values[difficulty_idx])
    manager.generate_quests(df)

def generate_ui_components(manager, df):
    """Generate all UI components based on current state"""
    quest_items = generate_quest_items(manager)
    score_fig = create_score_visualization(manager)
    progress = calculate_progress(manager)
    impact_fig = create_impact_visualization(df, manager)
    
    return [
        quest_items,
        f"{manager.get_score()}",
        score_fig,
        json.dumps(get_manager_state(manager)),
        progress,
        get_enabled_types_text(manager),
        get_ai_comment(data_simulator),
        json.dumps(data_simulator.get_event_history()),
        create_lore_display(manager),
        get_arc_description(manager),
        f"{progress}%",
        impact_fig
    ]

def generate_quest_items(manager):
    """Generate quest item components"""
    return [html.Li([
        quest['description'],
        html.Button(
            'Complete',
            id={'type': 'complete-quest', 'index': quest['id']},
            style={
                'marginLeft': '10px',
                'backgroundColor': SUCCESS_COLOR,
                'color': TEXT_COLOR,
                'border': 'none',
                'borderRadius': '3px',
                'padding': '3px'
            }
        )
    ], style={
        'marginBottom': '5px',
        'border': f'1px solid {SECONDARY_COLOR}',
        'padding': '5px',
        'borderRadius': '3px',
        'textDecoration': 'line-through' if quest.get('completed', False) else 'none'
    }) for quest in manager.get_quests()]

def create_impact_visualization(df, manager):
    """Create impact visualization figure"""
    completed_quests = manager.get_completed_quests()
    impact_data = calculate_quest_impacts(completed_quests)
    
    fig = go.Figure()
    for metric, values in impact_data.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            name=metric,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        plot_bgcolor=GRAPH_BG_COLOR,
        paper_bgcolor=GRAPH_BG_COLOR,
        font_color=TEXT_COLOR,
        title='Quest Impact on Unity Metrics',
        xaxis_title='Completed Quests',
        yaxis_title='Impact Magnitude',
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def calculate_quest_impacts(completed_quests):
    """
    Calculate cumulative impacts of completed quests with optimized data structures.
    Uses defaultdict for O(1) metric updates and pre-allocated lists for performance.
    """
    metrics = {'Social Cohesion', 'Personal Resonance', 'Global Resonance', 'Economic Alignment'}
    impact_data = {metric: [] for metric in metrics}
    cumulative_impacts = defaultdict(float)
    
    # Pre-calculate for O(n) performance
    for quest in completed_quests:
        quest_type = quest['type']
        impact = quest.get('impact', 0.05)
        
        # O(1) metric update with optimized mapping
        if quest_type == QuestType.SOCIAL:
            cumulative_impacts['Social Cohesion'] += impact
        elif quest_type == QuestType.PERSONAL:
            cumulative_impacts['Personal Resonance'] += impact
        elif quest_type == QuestType.GLOBAL:
            cumulative_impacts['Global Resonance'] += impact
        elif quest_type == QuestType.ECONOMIC:
            cumulative_impacts['Economic Alignment'] += impact
            
        # O(1) append for each metric
        for metric in metrics:
            impact_data[metric].append(cumulative_impacts[metric])
            
    return impact_data

# --- Run App ---
if __name__ == '__main__':
    app.run_server(debug=True)

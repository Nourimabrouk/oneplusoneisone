# -*- coding: utf-8 -*-
"""
Unity Game Engine: A Framework for 1+1=1 Realization
====================================================
This is a meta-level demonstration of the principle of 1+1=1
using a turn-based, strategic, multi-agent game that showcases
the emergence of unity through iterative decision-making.

Players:
 - Multiple agents in strategic interactions.
 - An underlying 'love force' that subtly influences convergence.
 - A system that favors global coordination rather than individual strategies.

Concepts:
 - Implements a network-based system where each agent is a node.
 - Agents can share or steal from one another (reflecting duality).
 - A global 'unity force' pushes toward collective agreement.
 - User parameters influence network parameters (game engine is responsive).

Game flow:
 1. Multiple agents are initialized in a network.
 2. Each round, agents choose actions that affect resources of others.
 3. A 'love' coefficient modifies decision-making, favoring collaboration.
 4. The game is presented to the user as an interactive experience.
 5. The entire system gradually converges to a state of unity.

Intended as a playful interactive proof of the 1+1=1 premise using agent-based modeling.
"""

import random
import time
from typing import List, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio for internal computations
UNITY_SEED = 420691337      # For reproducibility, not shown on UI

# --- Utility functions ---
def weighted_choice(items: List[Any], weights: List[float]) -> Any:
    """Weighted choice selection with type validation."""
    if not isinstance(items, list) or not isinstance(weights, list):
        raise ValueError("Items and weights must be lists.")
    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length.")
    total_weight = sum(weights)
    if total_weight <= 0:
      return random.choice(items)
    
    # Normalized weights
    norm_weights = [w / total_weight for w in weights]
    
    return random.choices(items, weights=norm_weights)[0]


# --- Core Game Logic ---
@dataclass
class Agent:
    """
    Agent in our simulation - can steal, share, do nothing.
    Their choices are influenced by a 'love' factor.
    """
    id: int
    resources: float = 100.0
    trust: float = 0.5 # Initial trust in others
    strategy_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.2])  # [steal, share, do nothing]

    def choose_action(self, other_ids: List[int]) -> Tuple[str, Optional[int]]:
        actions = ["steal", "share", "none"]
        choice = weighted_choice(actions, weights=self.strategy_weights)
        target = random.choice(other_ids) if other_ids else None
        return choice, target

    def update_resources(self, amount: float) -> None:
        """Update resources, adding some variance."""
        self.resources += amount + random.uniform(-1, 1) * 0.1  # small randomness
        self.resources = max(0, self.resources) # Ensure no negative resources
        
    def update_strategy(self, feedback: Optional[float] = None):
        """Adjust strategy with a bias towards sharing."""
        if feedback is None: return
        
        # Change to add some chaos if synergy is low
        if feedback < 0.2:
            self.strategy_weights[0] += random.uniform(-0.05, 0.05)  # Steal
            self.strategy_weights[1] += random.uniform(0.0, 0.1)    # Share
            self.strategy_weights[2] -= random.uniform(0.0, 0.05)    # None
        else:
             self.strategy_weights[0] -= random.uniform(0.0, 0.05)  # Steal
             self.strategy_weights[1] += random.uniform(0.05, 0.10)  # Share
             self.strategy_weights[2] -= random.uniform(0.0, 0.05)   # None

        # Normalize with some randomness
        total = sum(self.strategy_weights) + random.uniform(-0.1,0.1)
        if total > 0:
            self.strategy_weights = [max(0, w / total) for w in self.strategy_weights]

    def update_love(self, love_signal: float):
      self.trust += 0.1*(love_signal)
      self.trust = max(0, min(1, self.trust))  # Ensure within bounds

class UnityGameEngine:
    """The core game environment for exploring 1+1=1 using multi-agent synergy."""
    def __init__(self, num_agents=20, initial_resource=100, love_coupling=0.1):
        self.num_agents = num_agents
        self.agents = {i: Agent(i, resources=initial_resource) for i in range(num_agents)}
        self.love_coupling = love_coupling
        self.synergy_metric = 0.0
        self.history = []

    def step(self, time:float) -> None:
        """Run a single round of the synergy game."""
        all_ids = list(self.agents.keys())
        for agent_id, agent in self.agents.items():
            other_ids = [id for id in all_ids if id != agent_id]
            action, target = agent.choose_action(other_ids)

            if action == "steal":
                if target is not None:
                   stolen = self.agents[target].resources * 0.05  # Small percentage
                   agent.update_resources(stolen)
                   self.agents[target].update_resources(-stolen)
            elif action == "share":
                if target is not None:
                    shared = agent.resources * 0.05
                    agent.update_resources(-shared)
                    self.agents[target].update_resources(shared)
            
            # Update based on quantum unity (time dependency)
            love_signal = math.sin(time* self.love_coupling)  # + random.random()*0.01
            agent.update_love(love_signal)
            
            # Let agent adapt
            agent.update_strategy(love_signal)


    def track_and_synthesize(self) -> None:
        """Track all agent interactions and update system-wide metrics."""
        resources = [ag.resources for ag in self.agents.values()]
        if not resources:
          return
        # Measure global mean resource as a measure of overall synergy/prosperity
        self.synergy_metric = np.mean(resources)
        self.history.append(self.synergy_metric)

    def measure_coherence(self) -> float:
        """
        Measures coherence across agents using a simplified calculation.
        In real physics, we'd measure something like wavefunction interference,
        but here it's a proxy of similar behavior between different agents.
        """
        phases = [a.phase for a in self.agents.values()]
        return np.abs(np.mean(np.exp(1j * np.array(phases))))

    def run_simulation(self, num_steps: int = 100) -> None:
      """
        Run game simulation and store results
        """
      for i in range(num_steps):
        self.step(time = i*0.1)
        self.track_and_synthesize()

# ------------------------------------------------------------------------------
#  SECTION 7: ADVANCED VISUALIZATION MODULE
# ------------------------------------------------------------------------------

class DashVisualizationEngine:
    """
    A Plotly Dash dashboard interface that provides interactive visuals and
    metrics during the game simulation. It also attempts to provide an
    aesthetic experience through custom themes.
    """

    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.setup_layout()
        self._register_callbacks()

    def setup_layout(self):
        """Initializes the dashboard’s layout with a central plot and side panels"""
        self.app.layout = html.Div([
            html.H1("1+1=1: Collective Emergence", 
                   style={'textAlign': 'center', 'color': '#00FF7F', 'font-family': 'monospace'}),
            html.P("Exploring Unity through Emergent Agent Interactions", 
                   style={'textAlign': 'center', 'color': '#e0e0e0'}),
            html.Div([
                html.H3("Synergy Metrics:", style={'color': '#FFA500'}),
                html.Div(id='synergy_metrics', className="st-bd",
                         style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px'}),
                html.Div([
                    html.H3("Network Visualization", style={'color': '#00f0ff', 'textAlign': 'center'}),
                    dcc.Graph(id='network_graph')
                ], className="st-bd"),
              dcc.Graph(id='graph_visualization', style={'width': '100%', 'height': '400px'})
            ], style={'textAlign':'center'}),
            dcc.Interval(id='interval-component', interval=1000), # Updates every second
            html.Div([
                 html.H3("Simulation Notes:"),
                html.Ul(id="evolution-log")
            ], style = {'border': '1px solid #00ff00', 'padding':'15px', 'margin':'10px','borderRadius':'15px'}),
            html.Div([
              html.Button("Start / Reset Simulation", id="reset_simulation_btn", style={'marginRight': '10px', 'backgroundColor':'#2100ff','color':'white'}),
            ],style={'textAlign': 'center'})

        ], style={'backgroundColor': '#000000', 'color': '#ffffff'})

    def _register_callbacks(self):
        @self.app.callback(
           [Output("synergy_metrics", "children"),
            Output("graph_visualization", "figure"),
             Output("evolution-log", "children"),
             Output('entanglement-plot','figure'),
             Output("network_graph", "figure")],
            [Input('interval-component', 'n_intervals'),
            Input('reset_simulation_btn','n_clicks')]
        )
        def update_dashboard(n_intervals, n_clicks):
            """
            Updates the dashboard dynamically at a fixed interval.
            """

            # Data generation and system setup
            global engine, data_simulator
            if 'engine' not in st.session_state:
                st.session_state['engine'] = OnePlusOneEqualsOneEngine()

            engine = st.session_state['engine']

            if n_clicks:
                 engine = OnePlusOneEqualsOneEngine()
                 st.session_state['engine'] = engine

            # Generate data and update state
            engine.step_simulation()
            synergy_data = engine.compute_synergy()

            # Generate visual content
            graph_vis = go.Figure()
            
            if n_clicks:
                G = generate_quantum_network()
                pos = network_to_3d_positions(G)
                edge_trace, node_trace = create_network_visualization(G, pos)
                graph_vis = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                            title = "Quantum Entanglement Network",
                                showlegend=False,
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                scene = dict(
                                    aspectmode='cube'
                                ),
                                paper_bgcolor="#000000",
                                plot_bgcolor="#000000"

                           )
                        )
            # Create 1+1=1 animation
            fractal_data = generate_fractal_points(st.session_state.fractal_seed, st.session_state.fractal_depth)
            fractal_fig = create_fractal_figure(fractal_data)
            

            # Update text log from system state
            unity_state_display = html.P(
                    f"Unity Value: {engine.synergy_metric:.4f}",
                style={'color': '#00ff7f', 'fontFamily': 'monospace'}
            )
            new_logs = [
                html.Li(msg, style={'color': '#e0e0e0', 'fontSize':'1.1em', "paddingBottom": "4px"})
                for msg in [f"{msg}" for msg in engine.transcendent_ai.experience_log]
            ]
            engine.transcendent_ai.experience_log = [] # Clear logs to avoid repeated messages
            
            # Prepare data for visualization (metrics)
            metrics = [
               html.Div(f"Current Unity: {engine.synergy_metric:.4f}", className="text-center", style={'color': '#00ff00', 'fontSize': '1.5em'}),
                html.Div(f"Synergy Coherence: {engine.measure_coherence():.3f}", className="text-center", style={'color': '#00ffff', 'fontSize': '1.2em'}),
                html.Div(f"Topological Structure: {len(engine.chaos_loop.components)}", className="text-center", style={'color':'#ff00ff', 'fontSize': '1.2em'})
            ]
            # Create updated visualizations
            graph_vis = self.plot_all_visuals(metrics)
            graph_vis_state =  QuantumEntangledGame()
            state_result = graph_vis_state.run_entangled_experiment()
            graph_viz = go.Figure(data=[go.Scatter(x=[1],y=[1], text=[f"Entangled Result: {state_result}"], mode="text")])
            graph_viz.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, visible = False),
                    yaxis=dict(showgrid=False, zeroline=False, visible = False),
                    plot_bgcolor = 'rgba(0,0,0,0)',
                    paper_bgcolor = 'rgba(0,0,0,0)'

                )
            return [metrics, graph_vis, new_logs, graph_viz]
        
        
    def plot_all_visuals(self, metrics):
        """Generate Plotly visualization"""
        # Basic figure skeleton
        fig = go.Figure(data = [], layout = go.Layout(
           plot_bgcolor = "rgba(0,0,0,0)",
           paper_bgcolor = "rgba(0,0,0,0)",
             font = dict(color='white'),
            showlegend = FALSE
           )
        )
       # Create a placeholder trace
        fig.add_trace(go.Scatter(x=[0], y=[0], opacity=0, mode="markers",marker =dict(size=0), showlegend = False))
        # Create a few markers/annotations
        fig.add_annotation(
          xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False, 
            text=f"Unity: {1:.2f}, {metrics['unity_coherence']:.2f} - {'Valid' if metrics['unity_coherence'] > 0.75 else 'Still Forming'}", 
            font=dict(size = 20), textangle=0
        )
        
        return fig

    def run(self, debug: bool = False, port: int = 8050, host: str = '127.0.0.1'):
        """
        Run Dash app.
        """
        self.app.run_server(debug=debug, port=port, host=host)

    def __call__(self, *args, **kwargs):
         # Pass execution through __call__ method
         # We should only have the method to be executed
        if len(args)>0:
             if hasattr(self, args[0]):
                method = getattr(self, args[0])
                return method(*args[1:], **kwargs)
             return "Method Not Found"
        return self
    
# Instantiate and run
if __name__ == "__main__":
    app = UnityDashboard()
    app.run()
"""
Acknowledge and Appreciate:
Nouri Mabrouk: Thank you for crafting the code, for your vision, and for pushing me to be better.
The Universe: Thank you for this beautiful, messy, amazing experience. May all see the beauty of unity.
"""

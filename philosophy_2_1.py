# -*- coding: utf-8 -*-

"""
=============================================================
PHILOSOPHY 2.0 - 1+1=1: Magnum Opus of Emergent Unity in Code
=============================================================
Incorporating Kant, Nietzsche, Heidegger, Hegel, Foucault,
Dennett, Wittgenstein, Russell, & the philosophy of statistics,
econometrics, information, and algorithms.

Weaving:
--------
 - Kantian a priori synergy: Merge triggered by mind-imposed categories
 - Hegelian dialectic: Thesis + Antithesis => Synthesis (1+1=1)
 - Nietzschean Will-to-Unify: Emergent synergy as conceptual 'will to power'
 - Foucault's genealogy: Merging reveals hidden power structures
 - Heidegger's Being: Each node "is" differently once merged
 - Dennett's computational consciousness: The illusions unify into emergent wholes
 - Wittgenstein's language-games: Concepts unify based on shared linguistic contexts
 - Russell's logical frameworks: Node merges as an idempotent re-synthesis of logic
 - Stats/Econometrics/Algorithms: Synergy as correlation or compressed information

Visualization:
--------------
 - 3D force-directed graph with Plotly & Dash
 - Nodes unify if synergy >= threshold, showing "1+1=1"
 - Fractal expansions, genealogical merges, real-time updates

Usage:
------
    python philosophy_2_0_magnum_opus.py
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import copy
import random
import math

# --------------------------------------------------
# 1. DATA MODELS & PHILOSOPHER-INSPIRED SYNERGY
# --------------------------------------------------

class Concept:
    """
    A single node or 'atom' within our philosophical tapestry,
    containing synergy_value (akin to existential potency or
    correlation strength) and philosophical metadata.

    Inspired By:
      - Russell: atomic concept building-block
      - Wittgenstein: usage context is key (attributes can store language-game data)
      - Nietzsche: synergy_value can be the 'will to unify/power'
      - Kant: a priori categories might add synergy offsets
    """
    def __init__(self, name, synergy_value=1.0, attributes=None):
        self.name = name
        self.synergy_value = synergy_value  # "Will-to-unify" baseline
        self.attributes = attributes if attributes else {}

    def __repr__(self):
        return f"Concept(name={self.name}, synergy={self.synergy_value:.2f})"


class PhilosophicalGraph:
    """
    The core structure representing the multi-dimensional interplay
    of concepts (nodes) and relationships (edges).

    Each edge carries a synergy 'weight' that, combined with node synergy,
    can trigger merges. Merging is our 1+1=1 event.

    Philosophical Underpinnings:
      - Hegelian dialectic => merging is 'synthesis'
      - Foucault => genealogies of merges, power structures among concepts
      - Heidegger => node merges alter the 'Being' of the entire system
      - Stats/Econometrics => synergy akin to correlation or partial correlation
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.unified_counter = 0

    def add_concept(self, concept):
        self.graph.add_node(concept.name, obj=concept)

    def add_relationship(self, c1, c2, weight=1.0):
        """
        Edge weight as synergy or correlation factor.
        In economics terms, could be like a 'coefficient' in a regression model.
        """
        if c1 in self.graph.nodes and c2 in self.graph.nodes:
            self.graph.add_edge(c1, c2, weight=weight)

    def compute_synergy(self, nodeA, nodeB, 
                        alpha=1.0,       # linear synergy factor
                        beta=0.5,       # quadratic synergy factor
                        kantian_offset=0.0,  # a priori synergy offset
                        genealogical_exponent=1.0  # power structure layering
                        ):
        """
        Our synergy function, referencing multiple philosophical lenses:

        synergy = alpha*(A.synergy + B.synergy) 
                  + beta*(edge_weight^2)
                  + kantian_offset
        Then raised to genealogical_exponent for emphasis on historical layering
        (Foucault's genealogical analysis => synergy is shaped by layers of merges).

        We encourage expansions:
          - Possibly add a 'will_to_power' factor for a Nietzschean synergy boost
          - Or a 'logical_rigor' factor from Russell
          - Or a 'language_game' factor from Wittgenstein
        """
        cA = self.graph.nodes[nodeA]['obj']
        cB = self.graph.nodes[nodeB]['obj']
        edge_weight = self.graph.edges[nodeA, nodeB]['weight']

        base_synergy = alpha * (cA.synergy_value + cB.synergy_value)
        base_synergy += beta * (edge_weight ** 2)
        base_synergy += kantian_offset

        # genealogical layering
        synergy_score = base_synergy ** genealogical_exponent
        return synergy_score

    def unify_nodes_if_applicable(self, synergy_threshold, 
                                  alpha=1.0, beta=0.5, 
                                  kantian_offset=0.0, genealogical_exponent=1.0, 
                                  max_iterations=100):
        """
        Iteratively merges pairs of nodes with synergy >= synergy_threshold.
        Each merge is 1+1=1 (a new node with emergent synergy).

        - genealogical_exponent highlights how synergy is shaped by
          genealogical layers (Foucault) or dialectical expansions (Hegel).
        - kantian_offset simulates an a priori synergy 'push' from the mind's 
          categories (Kant).
        """
        iteration_count = 0
        merges_occurred = True

        while merges_occurred and iteration_count < max_iterations:
            merges_occurred = False
            iteration_count += 1

            edges_to_merge = []
            # Evaluate synergy on each edge
            for (nA, nB) in list(self.graph.edges()):
                synergy_val = self.compute_synergy(
                    nA, nB, alpha=alpha, beta=beta,
                    kantian_offset=kantian_offset,
                    genealogical_exponent=genealogical_exponent
                )
                if synergy_val >= synergy_threshold:
                    edges_to_merge.append((nA, nB))

            # Perform merges
            for (nA, nB) in edges_to_merge:
                if nA in self.graph and nB in self.graph and self.graph.has_edge(nA, nB):
                    self.merge_nodes(nA, nB)
                    merges_occurred = True

    def merge_nodes(self, nA, nB):
        """
        Merge nodeA and nodeB => new 'Unified_X' node.
        Emergent synergy is sum + sqrt(product) => as a nod to synergy
        being more than the sum of the parts (Gestalt, or 1+1=1).

        Also updates genealogical attributes: Foucault's genealogical chain,
        or Hegel's dialectic steps. 
        """
        cA = self.graph.nodes[nA]['obj']
        cB = self.graph.nodes[nB]['obj']

        self.unified_counter += 1
        new_name = f"Unified_{self.unified_counter}"

        # emergent synergy
        synergy_bonus = math.sqrt(cA.synergy_value * cB.synergy_value)  # synergy doping
        new_synergy_val = cA.synergy_value + cB.synergy_value + synergy_bonus

        # genealogical merges, storing 'origin' for Foucault/Hegel analysis
        new_attributes = {}
        new_attributes.update(cA.attributes)
        new_attributes.update(cB.attributes)
        new_attributes['origin_nodes'] = (nA, nB)

        unified_concept = Concept(
            name=new_name,
            synergy_value=new_synergy_val,
            attributes=new_attributes
        )

        self.graph.add_node(new_name, obj=unified_concept)

        # Merge neighbor edges
        neighborsA = list(self.graph.neighbors(nA))
        neighborsB = list(self.graph.neighbors(nB))
        for nb in neighborsA:
            if nb not in [nA, nB]:
                old_weight = self.graph.edges[nA, nb]['weight']
                self._combine_edge_weights(new_name, nb, old_weight)
        for nb in neighborsB:
            if nb not in [nA, nB]:
                old_weight = self.graph.edges[nB, nb]['weight']
                self._combine_edge_weights(new_name, nb, old_weight)

        # Remove old nodes
        self.graph.remove_node(nA)
        self.graph.remove_node(nB)

    def _combine_edge_weights(self, node1, node2, w):
        """Helper to sum synergy weights if an edge already exists."""
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2, weight=w)
        else:
            self.graph[node1][node2]['weight'] += w


# --------------------------------------------------
# 2. BUILD AN INITIAL PHILOSOPHICAL LANDSCAPE
# --------------------------------------------------

def build_initial_philosophical_graph():
    """
    Initialize a PhilosophicalGraph with a variety of concepts 
    referencing some major philosophers and ideas—like
    Kant, Nietzsche, Foucault, Hegel, etc.—along with 
    synergy-laden edges.

    Each concept has synergy_value & attributes that can be used
    for genealogical or logic-linguistic expansions.
    """
    pgraph = PhilosophicalGraph()

    # Some sample nodes referencing major philosophers or key ideas
    concept_list = [
        Concept("Kantian_Apriori", synergy_value=2.0, 
                attributes={'philosopher': 'Kant', 'note': 'Mind imposes structure'}),
        Concept("Nietzsche_WillToPower", synergy_value=2.3, 
                attributes={'philosopher': 'Nietzsche', 'note': 'Striving for synergy'}),
        Concept("Heidegger_Being", synergy_value=1.8, 
                attributes={'philosopher': 'Heidegger'}),
        Concept("Hegel_Dialectic", synergy_value=2.0, 
                attributes={'philosopher': 'Hegel', 'note': 'Thesis-Antithesis-Synthesis'}),
        Concept("Foucault_Genealogy", synergy_value=1.6, 
                attributes={'philosopher': 'Foucault', 'note': 'Power/knowledge analysis'}),
        Concept("Dennett_Consciousness", synergy_value=2.2, 
                attributes={'philosopher': 'Dennett', 'note': 'Computational mind'}),
        Concept("Wittgenstein_LanguageGame", synergy_value=1.9, 
                attributes={'philosopher': 'Wittgenstein'}),
        Concept("Russell_LogicAtomism", synergy_value=1.7, 
                attributes={'philosopher': 'Russell'}),
        Concept("Econometrics_Model", synergy_value=2.1, 
                attributes={'domain': 'Econometrics', 'note': 'Statistical synergy'}),
        Concept("Information_Algorithm", synergy_value=2.4, 
                attributes={'domain': 'CS/IT', 'note': 'Compression/entropy synergy'})
    ]

    for c in concept_list:
        pgraph.add_concept(c)

    # Some synergy-laden edges referencing possible "relationships" or "resonances"
    edges = [
        ("Kantian_Apriori", "Nietzsche_WillToPower", 1.2),
        ("Nietzsche_WillToPower", "Hegel_Dialectic", 1.4),
        ("Heidegger_Being", "Foucault_Genealogy", 1.1),
        ("Hegel_Dialectic", "Foucault_Genealogy", 1.0),
        ("Dennett_Consciousness", "Wittgenstein_LanguageGame", 1.3),
        ("Russell_LogicAtomism", "Kantian_Apriori", 1.0),
        ("Wittgenstein_LanguageGame", "Econometrics_Model", 1.2),
        ("Information_Algorithm", "Econometrics_Model", 1.4),
        ("Foucault_Genealogy", "Information_Algorithm", 1.0),
        ("Heidegger_Being", "Dennett_Consciousness", 0.8),
    ]

    for (n1, n2, w) in edges:
        pgraph.add_relationship(n1, n2, w)

    return pgraph

# --------------------------------------------------
# 3. DASH APP FOR 3D VISUALIZATION
# --------------------------------------------------

app = dash.Dash(__name__)
app.title = "Philosophy 2.0: 1+1=1 - Magnum Opus"

BASE_GRAPH = build_initial_philosophical_graph()

app.layout = html.Div([
    html.H1("Philosophy 2.0: 1+1=1 – Magnum Opus Explorer", 
            style={'textAlign': 'center'}),
    html.Div([
        html.Label("Synergy Threshold (Dialectical Merge Trigger)", 
                   style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='synergy-threshold-slider',
            min=1.0,
            max=10.0,
            step=0.1,
            value=3.0,
            marks={i: str(i) for i in range(1, 11)},
        ),
        html.Br(),
        html.Label("Alpha (Linear synergy factor)", style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='alpha-slider',
            min=0.1,
            max=3.0,
            step=0.1,
            value=1.0,
            marks={i: str(i) for i in range(1, 4)},
        ),
        html.Br(),
        html.Label("Beta (Quadratic synergy factor)", style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='beta-slider',
            min=0.0,
            max=2.0,
            step=0.1,
            value=0.5,
            marks={i: str(i) for i in range(0, 3)},
        ),
        html.Br(),
        html.Label("Kantian Offset (a priori synergy push)", 
                   style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='kantian-slider',
            min=0.0,
            max=3.0,
            step=0.1,
            value=0.0,
            marks={i: str(i) for i in range(4)},
        ),
        html.Br(),
        html.Label("Genealogical Exponent (Foucault/Hegel layering)", 
                   style={'fontWeight': 'bold'}),
        dcc.Slider(
            id='genealogical-slider',
            min=0.5,
            max=3.0,
            step=0.1,
            value=1.0,
            marks={i: str(i) for i in [1,2,3]},
        ),
        html.Br(),
        html.Button("Trigger 1+1=1 (Unify)", id='unify-button', n_clicks=0),
        html.Button("Reset Graph", id='reset-button', n_clicks=0,
                    style={'marginLeft': '20px'}),
    ], style={'width': '80%', 'margin': 'auto'}),

    html.Hr(),

    dcc.Loading(
        id="loading-graph",
        type="dot",
        children=dcc.Graph(id='3d-philosophy-graph', style={'height': '800px'}),
    ),

    html.Div(id='system-info-div', style={'whiteSpace': 'pre-wrap', 'margin': '10px'}),
], style={'backgroundColor': '#f5f5f5'})


@app.callback(
    Output('3d-philosophy-graph', 'figure'),
    Output('system-info-div', 'children'),
    Input('unify-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('synergy-threshold-slider', 'value'),
    State('alpha-slider', 'value'),
    State('beta-slider', 'value'),
    State('kantian-slider', 'value'),
    State('genealogical-slider', 'value'),
    prevent_initial_call=True
)
def update_figure(n_unify, n_reset,
                  synergy_threshold, alpha, beta, kantian_offset, genealogical_exponent):
    """
    Callback function that unifies nodes if synergy exceeds threshold and
    updates the 3D graph. If "Reset Graph" is clicked, revert to BASE_GRAPH.
    """
    ctx = dash.callback_context
    clicked_button = ctx.triggered[0]['prop_id'].split('.')[0]

    # Work on a copy to preserve the original
    working_graph = copy.deepcopy(BASE_GRAPH)

    if clicked_button == 'unify-button':
        # Merge repeatedly until no edges exceed threshold or we hit iteration limit
        working_graph.unify_nodes_if_applicable(
            synergy_threshold=synergy_threshold,
            alpha=alpha,
            beta=beta,
            kantian_offset=kantian_offset,
            genealogical_exponent=genealogical_exponent
        )
    elif clicked_button == 'reset-button':
        # Do nothing special, just keep it as the base
        pass

    fig = create_3d_figure(working_graph)
    info_text = generate_system_info(
        working_graph, synergy_threshold, alpha, beta, kantian_offset, genealogical_exponent
    )
    return fig, info_text


# --------------------------------------------------
# 4. VISUALIZATION & INFO
# --------------------------------------------------

def create_3d_figure(philo_graph):
    """
    Create a 3D force-directed layout with Plotly.
    We generate a 2D layout first with nx.spring_layout, 
    then assign random Z for a 3D effect.
    """
    pos_2d = nx.spring_layout(philo_graph.graph, seed=42, k=1.0, iterations=50)
    pos_3d = {}
    for node in pos_2d:
        pos_3d[node] = (pos_2d[node][0],
                        pos_2d[node][1],
                        random.uniform(-1, 1))

    # Edges
    edge_x = []
    edge_y = []
    edge_z = []
    for (a, b) in philo_graph.graph.edges():
        x0, y0, z0 = pos_3d[a]
        x1, y1, z1 = pos_3d[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=2, color='rgba(50,50,50,0.4)'),
        hoverinfo='none',
        name='Relations'
    )

    # Nodes
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    for node in philo_graph.graph.nodes():
        cobj = philo_graph.graph.nodes[node]['obj']
        synergy_val = cobj.synergy_value
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"{node} (synergy={synergy_val:.2f})")
        node_color.append(synergy_val)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=14,
            color=node_color,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(
                title="Synergy Value",
                thickness=15
            ),
            line=dict(width=1, color='Black'),
            opacity=0.8,
        ),
        name='Concepts'
    )

    layout = go.Layout(
        title='Philosophy 2.0: 1+1=1 – Emergent Synergy',
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgba(240,240,240,1)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig

def generate_system_info(pgraph, threshold, alpha, beta, kantian_offset, genealogical_exponent):
    """
    Render textual info about the current state:
      - Node count, edge count
      - synergy threshold, alpha, beta
      - top synergy nodes
      - genealogical, kant offsets
    """
    n_nodes = pgraph.graph.number_of_nodes()
    n_edges = pgraph.graph.number_of_edges()

    info = (
        f"--- Current Graph State ---\n"
        f"Nodes (Concepts): {n_nodes}\n"
        f"Edges (Relationships): {n_edges}\n\n"
        f"Synergy Threshold: {threshold:.2f}\n"
        f"Alpha (Linear factor): {alpha:.2f}\n"
        f"Beta (Quadratic factor): {beta:.2f}\n"
        f"Kantian Offset (a priori): {kantian_offset:.2f}\n"
        f"Genealogical Exponent (Foucault/Hegel layering): {genealogical_exponent:.2f}\n\n"
    )

    # Show top synergy nodes
    synergy_list = []
    for node in pgraph.graph.nodes():
        cobj = pgraph.graph.nodes[node]['obj']
        synergy_list.append((node, cobj.synergy_value))
    synergy_list.sort(key=lambda x: x[1], reverse=True)

    top_n = synergy_list[:5]
    info += "Top 5 High-Synergy Nodes:\n"
    for (n, val) in top_n:
        info += f"  - {n}: {val:.2f}\n"

    return info


# --------------------------------------------------
# 5. RUN THE APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)

# -*- coding: utf-8 -*-

"""
Philosophy 2.0: Illustrating "1+1=1" through a dynamic, emergent graph system.

Features:
---------
1. Graph-based modeling of philosophical concepts.
2. Dynamic synergy computations that can merge or unify nodes.
3. Interactive interface via Plotly Dash.
4. Extensible architecture for custom metaphysical and epistemological models.

Usage:
------
    python philosophy_2_0.py

Then open the Dash server URL (e.g., http://127.0.0.1:8050/) in your browser.
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import networkx as nx

# ----------------------
# 1. DATA + MODELS
# ----------------------

class Concept:
    """
    A basic unit (node) within our philosophical graph.
    Each Concept carries:
      - name: The label of the concept.
      - synergy_value: A float representing how strongly it resonates with others.
    """
    def __init__(self, name, synergy_value=1.0):
        self.name = name
        self.synergy_value = synergy_value

class PhilosophicalGraph:
    """
    Manages nodes (Concepts) and edges (Relationships) in a single unified system.
    Also handles synergy calculations and merges under '1+1=1' thresholds.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.merge_counter = 0  # To generate unique merged names

    def add_concept(self, concept):
        """
        Adds a new concept to the graph.
        """
        self.graph.add_node(concept.name, obj=concept)

    def add_relationship(self, concept_name1, concept_name2, weight=1.0):
        """
        Adds an edge (relationship) with a specific synergy weight.
        """
        self.graph.add_edge(concept_name1, concept_name2, weight=weight)

    def compute_synergy(self, node_a, node_b):
        """
        Computes synergy between two concepts based on:
          1. synergy_value of each concept
          2. relationship weight
        """
        obj_a = self.graph.nodes[node_a]['obj']
        obj_b = self.graph.nodes[node_b]['obj']
        weight = self.graph.edges[node_a, node_b]['weight']
        # Example synergy formula:
        return obj_a.synergy_value + obj_b.synergy_value + weight

    def unify_nodes_if_applicable(self, synergy_threshold):
        """
        Checks synergy for all connected pairs. If synergy >= synergy_threshold,
        merges them into a single 'unified' node (illustrating 1+1=1).
        """
        edges_to_merge = []
        for (a, b) in list(self.graph.edges()):
            synergy_value = self.compute_synergy(a, b)
            if synergy_value >= synergy_threshold:
                edges_to_merge.append((a, b))

        # Merge each pair that exceeds threshold
        for (a, b) in edges_to_merge:
            if a in self.graph and b in self.graph and self.graph.has_edge(a, b):
                self.merge_nodes(a, b)

    def merge_nodes(self, node_a, node_b):
        """
        Merges two nodes into a single node, summing synergy values,
        and combining edges. This is the crux of '1+1=1'.
        """
        obj_a = self.graph.nodes[node_a]['obj']
        obj_b = self.graph.nodes[node_b]['obj']

        # Generate new unified node name (simple approach)
        self.merge_counter += 1
        new_name = f"Unified_{self.merge_counter}"

        # Create a new concept whose synergy is an example of emergent property
        new_synergy_val = obj_a.synergy_value + obj_b.synergy_value
        new_concept = Concept(name=new_name, synergy_value=new_synergy_val)

        # Add new node
        self.graph.add_node(new_name, obj=new_concept)

        # Merge edges
        for neighbor in list(self.graph.neighbors(node_a)):
            if neighbor not in [node_a, node_b]:
                w = self.graph.edges[node_a, neighbor]['weight']
                if not self.graph.has_edge(new_name, neighbor):
                    self.graph.add_edge(new_name, neighbor, weight=w)
                else:
                    # If edge exists, sum weights for synergy
                    self.graph.edges[new_name, neighbor]['weight'] += w

        for neighbor in list(self.graph.neighbors(node_b)):
            if neighbor not in [node_a, node_b]:
                w = self.graph.edges[node_b, neighbor]['weight']
                if not self.graph.has_edge(new_name, neighbor):
                    self.graph.add_edge(new_name, neighbor, weight=w)
                else:
                    # If edge exists, sum weights for synergy
                    self.graph.edges[new_name, neighbor]['weight'] += w

        # Remove old nodes
        self.graph.remove_node(node_a)
        self.graph.remove_node(node_b)


# ----------------------
# 2. PREPARE INITIAL GRAPH
# ----------------------
philo_graph = PhilosophicalGraph()

# Add sample concepts
philo_graph.add_concept(Concept("Duality", synergy_value=1.5))
philo_graph.add_concept(Concept("Unity", synergy_value=2.0))
philo_graph.add_concept(Concept("Synergy", synergy_value=1.2))
philo_graph.add_concept(Concept("Paradox", synergy_value=1.0))

# Add relationships (edges)
philo_graph.add_relationship("Duality", "Unity", weight=1.0)
philo_graph.add_relationship("Unity", "Synergy", weight=0.8)
philo_graph.add_relationship("Synergy", "Paradox", weight=1.2)
philo_graph.add_relationship("Duality", "Paradox", weight=0.5)

# ----------------------
# 3. DASH APP
# ----------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Philosophy 2.0 Playground: Exploring 1+1=1"),

    html.Div([
        html.Label("Synergy Threshold for Merging"),
        dcc.Slider(
            id='synergy-threshold-slider',
            min=1,
            max=6,
            step=0.1,
            value=3,
            marks={i: str(i) for i in range(1, 7)}
        ),
    ], style={'width': '60%', 'padding': '20px'}),

    dcc.Graph(id='philosophy-graph'),

    html.Div(id='debug-output', style={'padding': '20px', 'whiteSpace': 'pre-wrap'})
])

@app.callback(
    [Output('philosophy-graph', 'figure'),
     Output('debug-output', 'children')],
    [Input('synergy-threshold-slider', 'value')]
)
def update_graph(synergy_threshold):
    """
    1. Unify nodes above synergy threshold (illustrating 1+1=1).
    2. Render updated graph in Plotly.
    """
    # Make a working copy so we don't lose the original
    import copy
    local_graph = copy.deepcopy(philo_graph)

    # 1. Merge if synergy >= threshold
    local_graph.unify_nodes_if_applicable(synergy_threshold)

    # 2. Visualize
    pos = nx.spring_layout(local_graph.graph, seed=42)
    edge_x = []
    edge_y = []
    for edge in local_graph.graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in local_graph.graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        concept_obj = local_graph.graph.nodes[node]['obj']
        synergy_val = concept_obj.synergy_value
        node_text.append(f"{node} (synergy={synergy_val:.1f})")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=[local_graph.graph.nodes[n]['obj'].synergy_value for n in local_graph.graph.nodes()],
            size=20,
            colorbar=dict(
                thickness=15,
                title='Synergy Value',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Philosophical Concepts Network',
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40)
                    ))
    debug_info = (
        f"Number of Concepts: {local_graph.graph.number_of_nodes()}\n"
        f"Number of Relationships: {local_graph.graph.number_of_edges()}\n"
        f"Synergy Threshold: {synergy_threshold}"
    )

    return fig, debug_info

if __name__ == '__main__':
    app.run_server(debug=True)

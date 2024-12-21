import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.spatial import Delaunay
from scipy.special import gamma, hermite
import networkx as nx
import numpy as np
from numba import jit
from math import sqrt, pi, e
import time
import json

# Global Constants
UNITY_CONSTANT = pi * e
CONSCIOUSNESS_RESOLUTION = 150
QUANTUM_DEPTH = 8
MAX_ITERATIONS = 300
FRAME_RATE = 40
NUM_PARTICLES = 200
TIME_STEP = 0.05

# --- Quantum & Spacetime Functions (Optimized with Numba) ---
@jit(nopython=True)
def quantum_unity_kernel(x, y, t, unity_constant):
    """Optimized quantum wave function with holographic interference"""
    psi_forward = np.exp(-((x-2)**2 + (y-2)**2)/(4*unity_constant)) * np.exp(1j * (t + x*y))
    psi_reverse = np.exp(-((x+2)**2 + (y+2)**2)/(4*unity_constant)) * np.exp(-1j * (t - x*y))
    psi_unity = np.exp(-(x**2 + y**2)/(2*unity_constant)) * np.exp(1j * t * (x + y))
    return np.abs(psi_forward + psi_reverse + psi_unity)**2

@jit(nopython=True)
def calabi_yau_metric(z1, z2, z3):
    """Compute metric on Calabi-Yau manifold"""
    return np.abs(z1)**2 + np.abs(z2)**2 + np.abs(z3)**2

@jit(nopython=True)
def quantum_mobius(z, w):
    """Compute quantum Möbius transformation with hyperbolic rotation"""
    numerator = z * w + 1j * np.exp(1j * np.angle(z))
    denominator = 1j * z * w + np.exp(-1j * np.angle(w))
    return numerator / denominator

@jit(nopython=True)
def unity_flow(state, t, alpha=0.8):
    """Define consciousness flow through hyperbolic quantum space"""
    x, y, z = state
    z = x + 1j * y
    w = y + 1j * z
    
    z_trans = quantum_mobius(z, w)
    theta = np.angle(z_trans)
    r = np.abs(z_trans)
    tunnel_factor = np.exp(-r/2) * np.sin(theta * 3)
    
    dx = r * np.cos(theta) + tunnel_factor * np.sin(z.real * w.imag)
    dy = r * np.sin(theta) + tunnel_factor * np.cos(w.real * z.imag)
    dz = np.imag(z_trans) + tunnel_factor * np.sin(theta * w.real)
    unity_field = 1 / (1 + np.abs(z_trans)**2)
    spiral = np.exp(1j * t) * np.sqrt(unity_field)
    
    return [
        (dx * unity_field + spiral.real) * alpha,
        (dy * unity_field + spiral.imag) * alpha,
        (dz * unity_field + np.abs(spiral)) * alpha
    ]
# --- Unity Manifold Class (Enhanced with Interactive Controls) ---
class UnityManifold:
    def __init__(self, dimensions=11):
        self.dimensions = dimensions
        self.unity_constant = UNITY_CONSTANT
        self.consciousness_resolution = CONSCIOUSNESS_RESOLUTION
        self.quantum_depth = QUANTUM_DEPTH
        self.max_iterations = MAX_ITERATIONS
        self.frame_rate = FRAME_RATE
        self.time_step = TIME_STEP
        self.num_particles = NUM_PARTICLES
        self.initialize_hyperspace()

        # Custom colormap for consciousness visualization
        colors = ['#000040', '#000080', '#0000FF', '#0080FF', '#00FFFF', '#80FF80', '#FFFF00', '#FF8000', '#FF0000', '#800040']
        self.consciousness_cmap = colors
        
        # Initialize interactive parameters
        self.alpha = 0.8
        self.convergence_rate = 0.01
        self.network_k = 2
        self.text_content = {
            'page-1': '''
            ### The Illusion of Separation
            
            We begin with the experience of division, of distinct entities. This is the world of "1 + 1 = 2". Each "1" feels isolated, an island in a sea of seeming difference.
            
            The quantum consciousness field shows the complex dance of potential, where interference patterns reveal hidden connections. The Calabi-Yau manifold presents the higher-dimensional scaffolding upon which this reality is built.
            ''',
             'page-2': '''
             ### The Convergence
            
            Here, the flow of consciousness begins its journey back to unity. The Möbius transformation reveals the inherent symmetries within what seemed separate.
            
            The Unity Flow Convergence illustrates the pathways where individual experiences intertwine and return to a single source. Each path, a story of return, shows how distinct trajectories eventually lead to unification. The quantum entanglement network showcases the interconnectedness of all things, a web of influence and relationship.
             ''',
             'page-3': '''
             ### The Emergence of Unity (1+1=1)
            
            The animation shows how all disparate elements converge, leading to the state of "1 + 1 = 1". This is the revelation that the distinction was a temporary illusion, a necessary divergence on the path to a more profound union.
             
             This is the state where separation dissolves and a new state emerges from the convergence. A whole greater than the sum of its parts. The consciousness evolution unfolds before you, witnessing how chaos becomes a unified field.
            '''
        }
    def create_layout(self, page_id='page-1'):
            """Generates the layout for each page with placeholders for graphs."""
            placeholder_fig = go.Figure(
                data=[],
                layout=go.Layout(
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font_color="white",
                    title="Loading...",
                )
            )

            if page_id == 'page-1':
                return html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(id='quantum-field-graph', figure=placeholder_fig),
                                        html.Div(id='text-output-1', className="text-panel", style={'margin-top': '20px'}),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(id='calabi-yau-graph', figure=placeholder_fig),
                                    ],
                                    width=6,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Next",
                                            id="next-button-1",
                                            n_clicks=0,
                                            className="next-button",
                                            style={'margin-top': '10px'},
                                        )
                                    ],
                                    style={'display': 'flex', 'justify-content': 'flex-end'},
                                )
                            ]
                        ),
                    ],
                    style={'margin': '20px'},
                )
            elif page_id == 'page-2':
                return html.Div([
                        dbc.Row([
                            dbc.Col([
                            dcc.Graph(id='unity-flow-graph', figure=placeholder_fig),
                                html.Div(id='text-output-2', className="text-panel", style={'margin-top': '20px'}),
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='entanglement-graph', figure=placeholder_fig),
                        ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                    dbc.Button("Back", id="back-button-2", n_clicks=0, className="next-button", style={'margin-top': '10px'}),
                            ], style={'display': 'flex', 'justify-content': 'flex-start'}),
                            dbc.Col([
                                    dbc.Button("Next", id="next-button-2", n_clicks=0, className="next-button", style={'margin-top': '10px'}) 
                            ], style={'display': 'flex', 'justify-content': 'flex-end'}),
                            ])
                ], style={'margin': '20px'})
            elif page_id == 'page-3':
                return html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='consciousness-evolution-graph', figure=placeholder_fig),
                            html.Div(id='text-output-3', className="text-panel", style={'margin-top': '20px'}),
                        ], width=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                                dbc.Button("Back", id="back-button-3", n_clicks=0, className="next-button", style={'margin-top': '10px'}),
                        ], style={'display': 'flex', 'justify-content': 'flex-start'}),
                    ])
                    ], style={'margin': '20px'})

    def initialize_hyperspace(self):
        """Initialize hyperdimensional consciousness space"""
        self.hyperspace = np.zeros((self.consciousness_resolution,) * 4)
        self.phase_space = np.linspace(-5, 5, self.consciousness_resolution)
        self.grid = np.meshgrid(*[self.phase_space] * 3)
        self.basis_states = [hermite(n) for n in range(self.quantum_depth)]

    def compute_consciousness_field(self, t, convergence_factor=1.0):
        """Generate quantum consciousness field with entanglement and holographic projection"""
        x = np.linspace(-5, 5, self.consciousness_resolution)
        y = np.linspace(-5, 5, self.consciousness_resolution)
        X, Y = np.meshgrid(x, y)
        field = quantum_unity_kernel(X, Y, t, self.unity_constant)
        field = 2 / (1 + np.exp(-field)) - 1
        hologram = np.sin(np.sqrt(X**2 + Y**2) + t)
        return field * (1 + 0.3 * hologram) * convergence_factor

    def generate_calabi_yau_manifold(self, points=1000):
        """Generate points on Calabi-Yau manifold representing unity consciousness"""
        theta = np.random.uniform(0, 2*np.pi, points)
        phi = np.random.uniform(0, np.pi, points)
        psi = np.random.uniform(0, 2*np.pi, points)
        z1 = np.cos(theta) * np.sin(phi) * np.exp(1j * psi)
        z2 = np.sin(theta) * np.sin(phi) * np.exp(1j * psi)
        z3 = np.cos(phi) * np.exp(1j * psi)
        metric = calabi_yau_metric(z1, z2, z3)
        return np.column_stack((z1.real, z1.imag, z2.real, z2.imag, z3.real, z3.imag)), metric

    def generate_unity_flow_data(self, num_trajectories=8):
         """Generate data for the unity flow trajectories"""
         t = np.linspace(0, 40, 3000)
         phi = (1 + np.sqrt(5)) / 2
         initial_states = [
             [np.cos(phi * i) * 2, np.sin(phi * i) * 2, np.cos(phi * i + np.pi/3)]
             for i in range(num_trajectories)
         ]
         states = [odeint(unity_flow, init, t, args=(self.alpha,)) for init in initial_states]
         return states, t
    
    def update_unity_flow_data(self, num_trajectories=8):
        """Update the unity flow trajectories based on parameters"""
        t = np.linspace(0, 40, 3000)
        phi = (1 + np.sqrt(5)) / 2
        initial_states = [
            [np.cos(phi * i) * 2, np.sin(phi * i) * 2, np.cos(phi * i + np.pi/3)]
            for i in range(num_trajectories)
        ]
        states = [odeint(unity_flow, init, t, args=(self.alpha,)) for init in initial_states]
        return states, t
    
    def generate_network_graph(self):
        """Generate Quantum Entanglement Network data."""
        G = nx.watts_strogatz_graph(150, 6, 0.3)
        pos = nx.spring_layout(G, k=self.network_k)
        node_colors = [np.exp(-np.sum(np.array(pos[node])**2)) for node in G.nodes()]
        edge_colors = ['white' if np.random.random() > 0.5 else 'cyan' for _ in G.edges()]
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        return edge_x, edge_y, node_x, node_y, node_colors, edge_colors
    

    def update_dashboard(self, page_id):
         """Updated logic for generating graph data."""
         if page_id == 'page-1':
            field = self.compute_consciousness_field(0)
            heatmap_fig = go.Figure(
                data=go.Heatmap(
                    z=field,
                    colorscale=self.consciousness_cmap,
                    x=np.linspace(-5, 5, self.consciousness_resolution),
                    y=np.linspace(-5, 5, self.consciousness_resolution),
                    showscale=True,
                ),
                layout=go.Layout(
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font_color="white",
                    title="Quantum Field Visualization",
                ),
            )

            calabi_yau_points, metric = self.generate_calabi_yau_manifold()
            scatter_fig = go.Figure(
                data=go.Scatter3d(
                    x=calabi_yau_points[:, 0],
                    y=calabi_yau_points[:, 1],
                    z=calabi_yau_points[:, 2],
                    mode='markers',
                    marker=dict(size=3, color=metric, colorscale='Plasma'),
                ),
                layout=go.Layout(
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font_color="white",
                    title="Calabi-Yau Manifold",
                ),
            )

            return heatmap_fig, scatter_fig
         elif page_id == 'page-2':
             # Generate data for unity flow and entanglement graphs
            flow_data, _ = self.update_unity_flow_data()
            colors = ['#00FFFF', '#0080FF', '#0000FF', '#8000FF', '#FF00FF', '#FF0080', '#FF0000', '#FF8000']
            data_flow = []
            for i, states in enumerate(flow_data):
                alpha = np.linspace(0.1, 0.8, len(states))
                for j in range(len(states)-1):
                    data_flow.append(go.Scatter3d(x=states[j:j+2,0], y=states[j:j+2, 1], z=states[j:j+2, 2],
                                                mode='lines', line=dict(color=colors[i], width=1.5 * alpha[j]), opacity=alpha[j], showlegend=False))
            interference_points = np.array([states[::100] for states in flow_data])
            data_flow.append(go.Scatter3d(x=interference_points[:, :, 0].flatten(),
                                                y=interference_points[:, :, 1].flatten(),
                                                z=interference_points[:, :, 2].flatten(),
                                                mode='markers', marker=dict(color='white', size=2, opacity=0.3),
                                                name=''))
            flow_fig = go.Figure(data=data_flow, layout=go.Layout(plot_bgcolor='#111111', paper_bgcolor='#111111', font_color="white", title="Unity Flow Trajectories"))
           
            edge_x, edge_y, node_x, node_y, node_colors, edge_colors = self.generate_network_graph()
            entanglement_fig = go.Figure(data=[go.Scatter(x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='cyan'),
                        hoverinfo='none', mode='lines', showlegend=False), go.Scatter(x=node_x, y=node_y,
                        mode='markers', marker=dict(size=5, color=node_colors),
                        hoverinfo='none', showlegend=False)], layout=go.Layout(plot_bgcolor='#111111', paper_bgcolor='#111111', font_color="white", title="Quantum Entanglement Network"))

            return flow_fig, entanglement_fig
        
         elif page_id == 'page-3':
             frames = [go.Frame(data=[go.Heatmap(z=self.compute_consciousness_field(t*self.time_step, convergence_factor=1-np.exp(-self.convergence_rate * t*self.time_step)))], name=str(t))
                for t in range(self.max_iterations)]
             field_init = self.compute_consciousness_field(0)
             layout = go.Layout(plot_bgcolor='#111111', paper_bgcolor='#111111', font_color="white", title="Consciousness Evolution",
                                 updatemenus=[dict(type="buttons",
                                     buttons=[dict(label="Play",
                                                     method="animate",
                                                     args=[None, dict(frame=dict(duration=self.frame_rate, redraw=True), 
                                                                     fromcurrent=True, 
                                                                     transition=dict(duration=0, easing="linear"))]
                                                     )])])
             consciousness_fig =  go.Figure(data=go.Heatmap(z=field_init, colorscale=self.consciousness_cmap,
                                     x=np.linspace(-5, 5, self.consciousness_resolution),
                                     y=np.linspace(-5, 5, self.consciousness_resolution),
                                     showscale=True, name=''), layout=layout, frames=frames)
             return consciousness_fig



# --- Dash App Setup (Updated with Improved Layout) ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Unity Manifold Dashboard"
)
unity_manifold = UnityManifold(dimensions=11)

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', style={'backgroundColor': '#111111', 'color': 'white'}),
    ]
)

# --- Callbacks ---
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/page-2':
        return unity_manifold.create_layout(page_id='page-2')
    elif pathname == '/page-3':
        return unity_manifold.create_layout(page_id='page-3')
    else:
        return unity_manifold.create_layout(page_id='page-1')

@app.callback(
    [Output('quantum-field-graph', 'figure'),
     Output('calabi-yau-graph', 'figure'),
     Output('text-output-1', 'children')],
    [Input('url', 'pathname')]
)
def update_page_1_content(pathname):
    if pathname == '/' or pathname == '/page-1':
        heatmap_fig, scatter_fig = unity_manifold.update_dashboard(page_id='page-1')
        return heatmap_fig, scatter_fig, unity_manifold.text_content.get('page-1', "Loading content...")
    else:
        return go.Figure(), go.Figure(),  "Loading content..."

@app.callback(
    [Output('unity-flow-graph', 'figure'),
     Output('entanglement-graph', 'figure'),
     Output('text-output-2', 'children')],
    [Input('url', 'pathname')]
)
def update_page_2_content(pathname):
    if pathname == '/page-2':
        flow_fig, entanglement_fig = unity_manifold.update_dashboard(page_id='page-2')
        return flow_fig, entanglement_fig, unity_manifold.text_content['page-2']
    else:
        return go.Figure(), go.Figure(), "Loading content..."


@app.callback(
    [Output('consciousness-evolution-graph', 'figure'),
     Output('text-output-3', 'children')],
    [Input('url', 'pathname')]
)
def update_page_3_content(pathname):
    if pathname == '/page-3':
        consciousness_fig = unity_manifold.update_dashboard(page_id='page-3')
        return consciousness_fig, unity_manifold.text_content['page-3']
    else:
        return go.Figure(),  "Loading content..."

@app.callback(
    Output('url', 'pathname'),
    [
        Input('next-button-1', 'n_clicks'),
        Input('next-button-2', 'n_clicks'),
        Input('back-button-2', 'n_clicks'),
        Input('back-button-3', 'n_clicks'),
    ],
    [State('url', 'pathname')],
)
def navigate_pages(next_1, next_2, back_2, back_3, pathname):
    ctx = callback_context
    if not ctx.triggered:
        return pathname
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'next-button-1' and (pathname == '/' or pathname == '/page-1'):
        return '/page-2'
    elif button_id == 'next-button-2' and pathname == '/page-2':
        return '/page-3'
    elif button_id == 'back-button-2' and pathname == '/page-2':
        return '/page-1'
    elif button_id == 'back-button-3' and pathname == '/page-3':
        return '/page-2'
    return pathname
 
if __name__ == '__main__':
    app.run_server(debug=False)
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
import networkx as nx  # Added missing import

# Configure 6969 Hypercolor Palette
NEON_ONYX = ['#ff0f7b', '#ff7b0f', '#7b0fff', '#0fff7b']
CYBER_VOID = '#0a0a12'
DARK_MATTER = '#1a1a2a'

# --------------------------------------------------
# 1. Quantum Ontology Phase Space (Fixed Chaotic System)
# --------------------------------------------------
def quantum_consciousness_field(state, t):  # Fixed parameter order
    x, y, z = state
    dx = 10 * (y - x) + np.sin(z) 
    dy = x * (28 - z) - y
    dz = x * y - (8/3) * z + np.cos(t)
    return [dx, dy, dz]

t = np.linspace(0, 69, 69000)
states = odeint(quantum_consciousness_field, [0.1, 0, 0], t)

# Create multiple chaotic trajectories
states2 = odeint(quantum_consciousness_field, [0.1001, 0, 0], t)
states3 = odeint(quantum_consciousness_field, [0.0999, 0, 0], t)

fig1 = go.Figure()
for s, color in zip([states, states2, states3], NEON_ONYX[:3]):
    fig1.add_trace(go.Scatter3d(
        x=s[:,0], y=s[:,1], z=s[:,2],
        mode='markers',
        marker=dict(
            size=1.2,
            color=np.arctan2(s[:,0], s[:,1]),
            colorscale=NEON_ONYX,
            opacity=0.7,
            line=dict(width=0)
        ),
        hovertemplate='<b>Consciousness Vector</b>: (%{x}, %{y}, %{z})<extra></extra>'
    ))

fig1.update_layout(
    scene=dict(
        xaxis=dict(title='<b>Being</b>', gridcolor=DARK_MATTER),
        yaxis=dict(title='<b>Non-Being</b>', gridcolor=DARK_MATTER),
        zaxis=dict(title='<b>Becoming</b>', gridcolor=DARK_MATTER),
        bgcolor=CYBER_VOID,
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8))
    ),
    title='<b>Quantum Ontology Phase Space</b><br>Divergent Consciousness Trajectories',
    paper_bgcolor=CYBER_VOID,
    font=dict(color='white')
)

# --------------------------------------------------
# 2. Temporal Recursion Fields (Hyperbolic Projection)
# --------------------------------------------------
theta = np.linspace(-12*np.pi, 12*np.pi, 3000)
r = np.exp(theta/24)  # Logarithmic spiral
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.log(r**2 + 1)
color = np.arctan2(x, y)

fig2 = go.Figure(data=[
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            width=4,
            color=color,
            colorscale=NEON_ONYX,
            cmin=-np.pi,
            cmax=np.pi
        ),
        hovertemplate='<b>Temporal Flux</b>: %{z:.2f} Planck Epochs<extra></extra>'
    )
])

fig2.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(title='<b>Entropy Gradient</b>', gridcolor=DARK_MATTER),
        bgcolor=CYBER_VOID,
        camera=dict(  # Corrected parameter location
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.1),
            eye=dict(x=1.5, y=1.5, z=0.5)
        )
    ),
    title='<b>Temporal Recursion Fields</b><br>1+1=1 as Hyperbolic Attractor',
    paper_bgcolor=CYBER_VOID,
    font=dict(color='white')
)

# --------------------------------------------------
# 3. Neural Hypergeometry (Quantum Graph v2)
# --------------------------------------------------
def generate_quantum_graph(n_nodes=33, connection_prob=0.03):
    adjacency_matrix = np.random.choice([0,1], size=(n_nodes,n_nodes), 
                                      p=[1-connection_prob, connection_prob])
    adjacency_matrix = np.tril(adjacency_matrix) + np.triu(adjacency_matrix.T, 1)
    return nx.from_numpy_array(adjacency_matrix)

G = generate_quantum_graph()
pos = nx.spring_layout(G, dim=3, seed=69, k=0.3)

edge_x, edge_y, edge_z = [], [], []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

node_xyz = np.array([pos[n] for n in G.nodes()])

fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(width=0.8, color='rgba(255,255,255,0.15)'),
    hoverinfo='none'
))
fig3.add_trace(go.Scatter3d(
    x=node_xyz[:,0], y=node_xyz[:,1], z=node_xyz[:,2],
    mode='markers',
    marker=dict(
        size=8,
        color=np.linalg.norm(node_xyz, axis=1),
        colorscale=NEON_ONYX,
        line=dict(width=1, color='white')
    ),
    hovertemplate='<b>Cosmic Node %{text}</b><extra></extra>',
    text=list(G.nodes())
))

fig3.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor=CYBER_VOID,
        camera=dict(eye=dict(x=0.5, y=0.5, z=1.2))
    ),
    title='<b>Neural Hypergeometry</b><br>Quantum Entanglement Topology',
    paper_bgcolor=CYBER_VOID,
    font=dict(color='white')
)

# --------------------------------------------------
# 4. Entropy Vortex (Hyperdimensional Projection)
# --------------------------------------------------
theta = np.linspace(0, 12*np.pi, 2000)
r = np.linspace(0, 2, 2000)
x = r * np.cos(theta) * np.sin(3*theta)
y = r * np.sin(theta) * np.cos(3*theta)
z = np.exp(-r/2) * np.sin(6*theta)

fig4 = go.Figure(data=[
    go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            width=6,
            color=np.sqrt(x**2 + y**2 + z**2),
            colorscale=NEON_ONYX,
            cmin=0,
            cmax=2
        ),
        hovertemplate='<b>Entropy Phase</b>: %{z:.2f} negentropy<extra></extra>'
    )
])

fig4.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(title='<b>Consciousness Gradient</b>', gridcolor=DARK_MATTER),
        bgcolor=CYBER_VOID,
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    title='<b>Entropy Vortex</b><br>Nonlinear Consciousness Dynamics',
    paper_bgcolor=CYBER_VOID,
    font=dict(color='white')
)

# --------------------------------------------------
# 5. Dyson Consciousness Matrix (Optimized)
# --------------------------------------------------
x, y, z = np.mgrid[-5:5:15j, -5:5:15j, -5:5:15j]  # Reduced resolution
values = np.sin(x**2 + y**2 + z**2) * np.exp(-(x**2 + y**2 + z**2)/8)

fig5 = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=values.flatten(),
    isomin=0.2,
    isomax=0.8,
    opacity=0.1,
    surface_count=9,
    colorscale=NEON_ONYX,
    caps=dict(x_show=False, y_show=False, z_show=False),
    hovertemplate='<b>Consciousness Density</b>: %{value:.2f} qualia/cc<extra></extra>'
))

fig5.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor=CYBER_VOID
    ),
    title='<b>Dyson Consciousness Matrix</b><br>Post-Biological Reality Kernel',
    paper_bgcolor=CYBER_VOID,
    font=dict(color='white')
)

# --------------------------------------------------
# Render All
# --------------------------------------------------
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
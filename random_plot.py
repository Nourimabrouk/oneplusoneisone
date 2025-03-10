import numpy as np
import plotly.graph_objs as go
from manim import *

# ----------------------------
# PLOT 1: Quantum Entanglement as 1+1=1
# ----------------------------
class QuantumHype(ThreeDScene):
    def construct(self):
        # Create Bell state sphere
        sphere = Sphere(resolution=(64, 64)).set_color("#FF6B6B")
        entanglement_lines = VGroup(*[
            Line(start=sphere.point_at_angle(u, v), 
                 end=sphere.point_at_angle(u + PI/2, v + PI/2),
                 color="#4ECDC4") 
            for u in np.arange(0, TAU, TAU/8)
            for v in np.arange(0, TAU, TAU/8)
        ])
        
        self.play(
            Create(sphere),
            Create(entanglement_lines),
            rate_func=linear,
            run_time=3
        )
        self.wait()
        self.play(
            sphere.animate.rotate(PI/2, UP),
            entanglement_lines.animate.set_opacity(0.2),
            run_time=5
        )
        text = Text("1 + 1 = 1", font_size=72).next_to(sphere, DOWN)
        self.play(Write(text))
        self.wait(2)

# ----------------------------
# PLOT 2: Holographic Tensor Network (Plotly)
# ----------------------------
nodes = np.array([[np.cos(theta), np.sin(theta), 0] 
                 for theta in np.linspace(0, 2*np.pi, 8)])
edges = [(i, (i+3)%8) for i in range(8)] + [(i, (i+1)%8) for i in range(8)]

fig = go.Figure()
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[nodes[edge[0]][0], nodes[edge[1]][0]],
        y=[nodes[edge[0]][1], nodes[edge[1]][1]],
        z=[0, 0],
        mode='lines',
        line=dict(color='#FF6B6B', width=4)
    ))
fig.add_trace(go.Scatter3d(
    x=nodes[:,0], y=nodes[:,1], z=nodes[:,2],
    mode='markers',
    marker=dict(size=10, color='#4ECDC4')
))
fig.update_layout(
    title="<b>Holographic Tensor Network</b><br>1+1=1 as Cosmic Edge Entanglement",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    template="plotly_dark"
)
fig.show()


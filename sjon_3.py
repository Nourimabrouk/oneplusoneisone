import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from scipy.interpolate import griddata

# ---------------------
# PLOT 1: 4D Onomastic Hypercube (Sjon/Jan Willem/Gerben in Quantum State-Space)
# ---------------------
fig = plt.figure(figsize=(25, 8))
ax1 = fig.add_subplot(131, projection='3d')

# Quantum state-space
theta = np.linspace(0, 4*np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = (np.sin(theta) + 0.5*np.cos(phi))  # Jan Willem: Political axis
y = (np.cos(theta) * np.sin(phi))      # Gerben: Mythic axis
z = (np.sin(phi)**2)                   # Sjon: Quantum collapse
c = np.sqrt(x**2 + y**2 + z**2)        # Velserbroek entropy

surf = ax1.plot_surface(x, y, z, facecolors=plt.cm.viridis(c), rstride=2, cstride=2, alpha=0.8)
ax1.set_title("4D ONOMASTIC HYPERCUBE\nSjon's Quantum Syntax in State-Space", fontsize=10, pad=15)
ax1.set_xlabel('Jan Willem (Politics)', fontsize=8)
ax1.set_ylabel('Gerben (Mythos)', fontsize=8)
ax1.set_zlabel('Sjon (Collapse)', fontsize=8)
fig.colorbar(surf, ax=ax1, label='Velserbroek Entropy')

# ---------------------
# PLOT 2: Velserbroek Cyber-Marsh Fluid Dynamics (Data Peat Currents)
# ---------------------
ax2 = fig.add_subplot(132)

# Simulate peat-data turbulence
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
U = -1 - X**2 + Y  # Jan's revolutionary inflow
V = 1 + X - Y**2   # Gerben's mythic outflow
speed = np.sqrt(U**2 + V**2)

# Sjon's quantum turbulence
U += np.random.normal(0, 0.3, U.shape)
V += np.random.normal(0, 0.3, V.shape)

strm = ax2.streamplot(X, Y, U, V, color=speed, cmap='copper', linewidth=1, density=2)
ax2.set_title("VELSERBROEK CYBER-MARSH\nData Peat Currents & Sjon's Turbulence", fontsize=10, pad=15)
ax2.set_xlabel('Revolutionary Inflow (Jan)', fontsize=8)
ax2.set_ylabel('Mythic Outflow (Gerben)', fontsize=8)
fig.colorbar(strm.lines, ax=ax2, label='Memetic Velocity')

# ---------------------
# PLOT 3: Triune Temporal-Pseudonym Network (Fixed Nodes)
# ---------------------
ax3 = fig.add_subplot(133)

G = nx.Graph()
eras = ['Medieval', '1600s Dutch Republic', '2025 Metareality']

# Add nodes with attributes
nodes = [
    ('Medieval: Jan (Anabaptist)', {'era': 0}),
    ('Medieval: Ger (Spear)', {'era': 0}),
    ('1600s: Willem (Statist)', {'era': 1}),
    ('1600s: Broek (Marsh)', {'era': 1}),
    ('2025: Sjon (Quantum)', {'era': 2}),
    ('2025: Velserbroek (Cyber-Marsh)', {'era': 2})
]
G.add_nodes_from(nodes)

edges = [
    ('Medieval: Jan (Anabaptist)', '1600s: Willem (Statist)'),
    ('Medieval: Ger (Spear)', '2025: Sjon (Quantum)'),
    ('1600s: Willem (Statist)', '2025: Sjon (Quantum)'),
    ('1600s: Broek (Marsh)', '2025: Velserbroek (Cyber-Marsh)'),
    ('2025: Sjon (Quantum)', '2025: Velserbroek (Cyber-Marsh)')
]
G.add_edges_from(edges)

# Position nodes by era (fixed hashability)
pos = {}
for node in G.nodes:
    era = G.nodes[node]['era']
    if era == 0:
        pos[node] = (0 + np.random.normal(0, 0.1), 0)
    elif era == 1:
        pos[node] = (1 + np.random.normal(0, 0.1), 0)
    elif era == 2:
        pos[node] = (2 + np.random.normal(0, 0.1), 0)

colors = ['#8c510a', '#d8b365', '#01665e']
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=[colors[G.nodes[n]['era']] for n in G.nodes])
nx.draw_networkx_edges(G, pos, edge_color='#5a5a5a', width=2, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='monospace')

ax3.set_title("TEMPORAL-PSEUDONYM NETWORK\nFrom Medieval Spear to Quantum Marsh", fontsize=10, pad=15)
ax3.axis('off')

plt.tight_layout()
plt.show()
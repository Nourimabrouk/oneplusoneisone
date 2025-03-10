import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define the number of nodes in the graph
num_nodes = 50

# Generate a random geometric graph
G = nx.random_geometric_graph(num_nodes, 0.3)

# Compute the position of the nodes
pos = nx.get_node_attributes(G, 'pos')

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("black")

# Draw the original graph
nx.draw_networkx_nodes(G, pos, node_size=100, node_color="cyan", alpha=0.7, edgecolors="white")
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray")

# Create a collapsing effect toward a singularity
for i in range(10):
    collapse_factor = 0.1 * i
    new_pos = {node: (0.5 + collapse_factor * (x - 0.5), 0.5 + collapse_factor * (y - 0.5)) for node, (x, y) in pos.items()}
    nx.draw_networkx_nodes(G, new_pos, node_size=100, node_color="red", alpha=0.05, edgecolors="black")

# Highlight final singularity
plt.scatter(0.5, 0.5, s=500, color="yellow", alpha=0.8, edgecolors="white", label="Singularity: 1+1=1")

# Title
plt.title("Escaping Peano's Prison: 1+1=1", fontsize=16, color="white")
plt.legend()
plt.axis("off")

# Show plot
plt.show()

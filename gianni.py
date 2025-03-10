import pandas as pd
import plotly.express as px
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 🔥 GRAPH 1: NAME POPULARITY TRENDS (PLOTLY LINE CHART) 🔥 ---
np.random.seed(42)
years = np.arange(1800, 2025, 10)

# Simulated name popularity trends
wilfradus_popularity = np.exp(-0.02 * (years - 1850)**2) * 1000  
gerben_popularity = np.exp(-0.01 * (years - 1950)**2) * 1500  
sjon_popularity = np.exp(-0.015 * (years - 2000)**2) * 1200  

df_names = pd.DataFrame({
    "Year": np.tile(years, 3),
    "Popularity": np.concatenate([wilfradus_popularity, gerben_popularity, sjon_popularity]),
    "Name": ["Wilfradus"] * len(years) + ["Gerben"] * len(years) + ["Sjon"] * len(years)
})

# Plotly Line Graph
fig1 = px.line(df_names, x="Year", y="Popularity", color="Name",
               title="🔥 Historical Popularity of Dutch Names (Wilfradus, Gerben, Sjon) 🔥",
               labels={"Popularity": "Relative Popularity Index", "Year": "Year"},
               line_shape="spline")
fig1.update_traces(line=dict(width=3))
fig1.show()


# --- 🔥 GRAPH 2: ETYMOLOGICAL NETWORK (PLOTLY NETWORK GRAPH) 🔥 ---
G = nx.Graph()

# Edges defining etymological connections
edges = [
    ("Wilfradus", "Old Germanic"), ("Wilfradus", "Latinized"),
    ("Gerben", "Frisian"), ("Gerben", "Old Germanic"),
    ("Sjon", "Dutch"), ("Sjon", "Hebrew (John)"),
    ("Gianni", "Italian"), ("Gianni", "Hebrew (John)"),
    ("Jan Willem", "Dutch"), ("Jan Willem", "Old Germanic"),
    ("Mathilde", "Old Germanic"), ("Mathilde", "French")
]

G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42)
df_graph = pd.DataFrame({"Node1": [e[0] for e in edges], "Node2": [e[1] for e in edges]})

# Convert graph into a DataFrame format for Plotly visualization
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

# Node positions
node_x = [pos[node][0] for node in G.nodes]
node_y = [pos[node][1] for node in G.nodes]

# Create DataFrame for Plotly Scatter
df_nodes = pd.DataFrame({"Name": list(G.nodes), "X": node_x, "Y": node_y})

# Plotly Scatter Plot for Network Graph
fig2 = px.scatter(df_nodes, x="X", y="Y", text="Name",
                  title="🌍 Etymological Network of Dutch Naming Conventions 🌍",
                  labels={"x": "Semantic Space X", "y": "Semantic Space Y"},
                  size_max=15)
fig2.update_traces(marker=dict(size=12, color="blue"))

# Add edges as lines
fig2.add_traces(px.line(x=edge_x, y=edge_y).data)

fig2.show()


# --- 🔥 GRAPH 3: META-PATTERN ANALYSIS (PCA REDUCTION) 🔥 ---
df_features = pd.DataFrame({
    "Name": ["Wilfradus", "Gerben", "Sjon", "Gianni", "Jan Willem", "Mathilde"],
    "Tradition": [5, 4, 2, 3, 4, 5],
    "Modernity": [1, 3, 5, 4, 2, 2],
    "Religious Influence": [4, 3, 2, 5, 3, 4],
    "Popularity Index": [2, 4, 5, 3, 4, 3],
    "Linguistic Complexity": [5, 3, 2, 4, 4, 5]
})

# Standardize the data for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features.iloc[:, 1:])

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

df_pca = pd.DataFrame(principal_components, columns=["PC1", "PC2"])
df_pca["Name"] = df_features["Name"]

# Plotly PCA Scatter Plot
fig3 = px.scatter(df_pca, x="PC1", y="PC2", text="Name",
                  title="🔮 Hidden Meta-Patterns in Dutch Naming Conventions 🔮",
                  labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                  size_max=15)
fig3.update_traces(marker=dict(size=12, color="purple"))

fig3.show()

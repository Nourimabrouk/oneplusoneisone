import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for unity and blending duality
colors = [(0.2, 0.4, 0.8), (1, 1, 1), (0.8, 0.2, 0.4)]  # Blue -> White -> Red
cmap = LinearSegmentedColormap.from_list("unity_cmap", colors, N=500)

# Golden spiral parameters
phi = (1 + np.sqrt(5)) / 2  # Golden ratio
theta = np.linspace(0, 4 * np.pi, 1000)
r = phi ** (theta / (2 * np.pi))

# Create two golden spirals merging into one
x1, y1 = r * np.cos(theta), r * np.sin(theta)  # Spiral 1
x2, y2 = -r * np.cos(theta), -r * np.sin(theta)  # Spiral 2

# Unity singularity
singularity_x, singularity_y = [0], [0]

# Fractal-like circular waves
angles = np.linspace(0, 2 * np.pi, 500)
radii = np.linspace(0.5, 2.5, 6)  # Multiple layers of circular waves
circles = [(r * np.cos(angles), r * np.sin(angles)) for r in radii]

# Plot setup
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_facecolor('black')

# Plot golden spirals
ax.plot(x1, y1, color='cyan', alpha=0.8, label="Spiral 1 (Particle A)")
ax.plot(x2, y2, color='magenta', alpha=0.8, label="Spiral 2 (Particle B)")

# Plot fractal-like circular waves
for circle_x, circle_y in circles:
    ax.plot(circle_x, circle_y, color='white', alpha=0.5, linewidth=0.8)

# Highlight singularity
ax.scatter(singularity_x, singularity_y, color='yellow', s=300, label="Singularity Point (1+1=1)")

# Add a resonance field as a gradient
gradient = ax.imshow(
    np.random.rand(100, 100), cmap=cmap, extent=[-3, 3, -3, 3], alpha=0.5
)

# Text annotations
ax.text(0, -3.5, "Eclipsed Singularity", color='white', fontsize=16, ha='center', weight='bold')
ax.text(0, 3.2, "1+1=1", color='yellow', fontsize=18, ha='center', weight='bold')

# Legend
ax.legend(loc="upper right", fontsize=10, frameon=False, labelcolor='white')

# Turn off axes
ax.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv
from numba import njit, prange
import math
import os
import threading
import queue
import time

# --- Constants ---
TIMESTEPS = 100
BASE_LEARNING_RATE = 0.05
INTERACTION_STRENGTH = 0.7
PLOT_WIDTH = 12
PLOT_HEIGHT = 8
DATA_QUEUE_SIZE = 100

# --- System State ---
system_state = {
    "learning_rate": BASE_LEARNING_RATE,
    "interaction_strength": INTERACTION_STRENGTH,
    "convergence": 0.0,
    "simulation_progress": 0,
    "fractal_parameters": {"mandelbrot_c": 0.0, "julia_c": 0.5},
    "manifold_parameters": {"amplitude_mod": 1.0, "frequency_mod": 1.0},
}

# --- Queue for data ---
data_queue = queue.Queue(maxsize=DATA_QUEUE_SIZE)

# --- Mathematical Core ---
@njit(parallel=True, fastmath=True)
def calculate_phi_unity(t, f1_values, f2_values):
    """Calculates the integral of the product of two functions."""
    phi_unity = 0.0
    for x in prange(len(f1_values)):
        phi_unity += f1_values[x] * f2_values[x]
    return phi_unity / len(f1_values) * t

@njit
def mandelbrot(x, y, c, max_iter=50):
    """Generates the Mandelbrot set."""
    z = x + 1j * y
    count = np.zeros(z.shape, dtype=np.int32)
    for i in range(max_iter):
        mask = np.abs(z) < 2
        z[mask] = z[mask] ** 2 + c
        count[mask] += 1
    return count

@njit
def julia(x, y, c, max_iter=50):
    """Generates the Julia set."""
    z = x + 1j * y
    count = np.zeros(z.shape, dtype=np.int32)
    for i in range(max_iter):
        mask = np.abs(z) < 2
        z[mask] = z[mask] ** 2 + c
        count[mask] += 1
    return count

# --- Dynamic Simulations ---
def create_fractal_animation():
    """Creates and saves fractal animations."""
    n = 500
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)
    x_grid, y_grid = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), ncols=2)
    ax[0].set_title("Mandelbrot Set")
    ax[1].set_title("Julia Set")

    def update(frame):
        t = frame / TIMESTEPS * 2 * np.pi
        mandelbrot_c = math.sin(t) * 0.5 * system_state["interaction_strength"]
        julia_c = math.cos(t) * 0.5 * system_state["interaction_strength"]

        mandelbrot_image = mandelbrot(x_grid, y_grid, mandelbrot_c)
        julia_image = julia(x_grid, y_grid, julia_c)

        ax[0].imshow(mandelbrot_image, cmap="inferno", extent=[-2, 2, -2, 2])
        ax[1].imshow(julia_image, cmap="viridis", extent=[-2, 2, -2, 2])

    ani = FuncAnimation(fig, update, frames=TIMESTEPS, interval=100)
    ani.save("fractal_animation.mp4", fps=10)
    plt.close(fig)
    print("Fractal animation saved as 'fractal_animation.mp4'.")

def create_manifold_visualization(phi_unity_history):
    """Creates and saves 3D manifold visualizations."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(x, y)

    def surface_function(t):
        z = np.sin(xx + t) * np.cos(yy + t) * np.tanh(phi_unity_history[t])
        return z

    plotter = pv.Plotter(off_screen=True)
    for t in range(TIMESTEPS):
        zz = surface_function(t)
        mesh = pv.StructuredGrid(xx, yy, zz)
        plotter.add_mesh(mesh, color="blue", opacity=0.5)
        filename = f"manifold_frame_{t:03d}.png"
        plotter.screenshot(filename)
        print(f"Saved 3D manifold frame: {filename}")
        plotter.clear()
    plotter.close()
    print("All manifold frames saved.")

# --- Simulation Logic ---
def simulation_step(t):
    """Performs one simulation step."""
    f1 = np.sin(t + np.linspace(0, 2 * np.pi, 100))
    f2 = np.cos(t + np.linspace(0, 2 * np.pi, 100))
    phi_unity = calculate_phi_unity(t, f1, f2)
    return phi_unity

def adjust_parameters(phi_unity):
    """Adjusts system parameters based on convergence."""
    delta_learning_rate = 0.05 * (1 - phi_unity)
    delta_interaction = 0.01 * (phi_unity - 0.5)

    system_state["learning_rate"] = max(-1, min(1, system_state["learning_rate"] + delta_learning_rate))
    system_state["interaction_strength"] = max(0, min(2, system_state["interaction_strength"] + delta_interaction))
    system_state["manifold_parameters"]["amplitude_mod"] = max(
        0.1, system_state["manifold_parameters"]["amplitude_mod"] + 0.02 * math.sin(phi_unity * 2 * math.pi)
    )
    system_state["manifold_parameters"]["frequency_mod"] = max(
        0.1, system_state["manifold_parameters"]["frequency_mod"] + 0.03 * math.cos(phi_unity * 2 * math.pi)
    )

def simulation_loop():
    """Main simulation loop."""
    phi_unity_history = np.zeros(TIMESTEPS)
    for t in range(TIMESTEPS):
        phi_unity = simulation_step(t)
        phi_unity_history[t] = phi_unity
        adjust_parameters(phi_unity)
        system_state["simulation_progress"] = t
        data_queue.put((t, phi_unity, system_state.copy()))
    data_queue.put(None)
    return phi_unity_history

def data_processing_thread():
    """Handles data processing and visualization."""
    phi_unity_history = simulation_loop()
    create_fractal_animation()
    create_manifold_visualization(phi_unity_history)

# --- Main Execution ---
if __name__ == "__main__":
    print("Simulation started. Outputs will be saved to the current working directory.")
    processing_thread = threading.Thread(target=data_processing_thread, daemon=True)
    processing_thread.start()
    processing_thread.join()
    print("Simulation completed. Check the working directory for outputs.")

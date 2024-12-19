import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sympy
from sympy import Symbol, integrate
import time
import openai
import os
from dotenv import load_dotenv
import math
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class UnifiedSystemConstants:
    """Encapsulates system-wide constants with philosophical significance."""
    TIMESTEPS = 100
    LEARNING_RATE = 0.1
    INTERACTION_STRENGTH = 1.0
    GOLDEN_RATIO = (1 + 5**0.5) / 2
    PLOT_WIDTH = 8
    PLOT_HEIGHT = 6

class QuantumHarmonics:
    """Core mathematical transformations for unified reality simulation."""
    
    @staticmethod
    def calculate_phi_unity(t: float, f1_values: np.ndarray, f2_values: np.ndarray) -> float:
        """Vectorized calculation of unity integral."""
        return np.sum(f1_values * f2_values) / len(f1_values) * t
    
    @staticmethod
    def generate_functions(t: float, n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generates coupled harmonic functions representing duality."""
        x = np.linspace(0, 2 * np.pi, n)
        f1 = np.sin(x * (1 + UnifiedSystemConstants.LEARNING_RATE * t))
        f2 = np.cos(x * (1 - UnifiedSystemConstants.LEARNING_RATE * t))
        return f1, f2
    
    @staticmethod
    def calculate_entanglement(f1: np.ndarray, f2: np.ndarray,
                             interaction_strength: float) -> Tuple[np.ndarray, np.ndarray]:
        """Quantum-inspired entanglement transformation."""
        return (f1 + interaction_strength * f2,
                f2 + interaction_strength * f1)

class FractalManifold:
    """Handles visualization of fractal duality and manifold evolution."""
    
    @staticmethod
    def create_fractal_animation():
        """Generates dynamic fractal visualization showing duality collapse."""
        n = 500
        max_iterations = 50
        x = np.linspace(-2, 2, n)
        y = np.linspace(-2, 2, n)
        x_grid, y_grid = np.meshgrid(x, y)
        z = x_grid + 1j*y_grid
        
        def compute_set(z: np.ndarray, c: complex) -> np.ndarray:
            z_temp = z.copy()
            count = np.zeros_like(z, dtype=int)
            mask = np.ones_like(z, dtype=bool)
            
            for _ in range(max_iterations):
                z_temp[mask] = z_temp[mask]**2 + c
                mask = np.abs(z_temp) < 2
                count += mask
            return count

        fig, ax = plt.subplots(figsize=(UnifiedSystemConstants.PLOT_WIDTH,
                                      UnifiedSystemConstants.PLOT_HEIGHT), ncols=2)

        mandelbrot = compute_set(z, 0)
        julia = compute_set(z, 0.5)
        
        mandelbrot_plot = ax[0].imshow(mandelbrot, cmap='inferno', extent=[-2,2,-2,2])
        julia_plot = ax[1].imshow(julia, cmap='viridis', extent=[-2,2,-2,2])
        
        ax[0].set_title("Mandelbrot Set")
        ax[1].set_title("Julia Set")
        fig.suptitle("Fractal Duality Manifestation")

        def update(frame):
            t = frame/UnifiedSystemConstants.TIMESTEPS * 2 * np.pi
            c_m = math.sin(t) * 0.5
            c_j = math.cos(t) * 0.5
            mandelbrot_plot.set_array(compute_set(z, c_m))
            julia_plot.set_array(compute_set(z, c_j))
            return mandelbrot_plot, julia_plot

        return FuncAnimation(fig, update, frames=UnifiedSystemConstants.TIMESTEPS,
                           blit=True, interval=100)

    @staticmethod
    def create_manifold_visualization(phi_unity_history: np.ndarray):
        """Generates 4D manifold visualization of unity evolution."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        xx, yy = np.meshgrid(x, y)
        
        def surface_function(t: int) -> np.ndarray:
            return np.sin(xx + t) * np.cos(yy + t) * np.tanh(phi_unity_history[t])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Unity Manifold Evolution")
        
        frames = []
        for t in range(0, UnifiedSystemConstants.TIMESTEPS, 5):
            z = surface_function(t)
            surf = ax.plot_surface(xx, yy, z, cmap='viridis',
                                 linewidth=0, antialiased=True)
            ax.set_zlim(-2, 2)
            frames.append([surf])
        
        ani = FuncAnimation(fig, lambda frame: frames[frame % len(frames)],
                          frames=len(frames), interval=100)
        plt.savefig('manifold_evolution.png')
        return ani

class UnityAnalysis:
    """Handles symbolic analysis and narrative generation."""
    
    @staticmethod
    def generate_symbolic_insights(phi_unity_history: np.ndarray) -> None:
        """Generates mathematical representation of unity convergence."""
        x = Symbol('x')
        unity_func = lambda t: t * (1 + math.tanh(phi_unity_history[t]))
        unified_eq = integrate(unity_func(x), (x, 0, 1))
        
        print("\nSymbolic Unity Analysis:")
        print(f"∫ Unity Expression: {unified_eq}")
        print(f"Convergence Value: {unified_eq.evalf(subs={x:UnifiedSystemConstants.TIMESTEPS}):.4f}")

    @staticmethod
    def generate_narrative(phi_unity_history: np.ndarray) -> None:
        """Generates AI-powered narrative of unity emergence."""
        prompt = f"""
        Articulate the mathematical poetry of 1+1=1, where duality collapses into unity.
        This simulation demonstrates how distinct mathematical functions merge into a unified whole.
        The convergence values {phi_unity_history} track this emergence of unity.
        Describe the visual manifestation through fractals and manifolds,
        connecting mathematical formalism with the philosophical principle of unified reality.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            print("\nEmergent Unity Narrative:")
            print(response.choices[0].message.content)
        except Exception as e:
            print(f"\nNarrative generation encountered an anomaly: {str(e)}")

def main():
    """Orchestrates the unified reality simulation."""
    print("\nInitiating Unified Reality Convergence Protocol...")
    
    # Initialize quantum state history
    phi_unity_history = np.zeros(UnifiedSystemConstants.TIMESTEPS)
    
    # Execute primary simulation loop
    for t in range(UnifiedSystemConstants.TIMESTEPS):
        f1, f2 = QuantumHarmonics.generate_functions(t)
        f1_entangled, f2_entangled = QuantumHarmonics.calculate_entanglement(
            f1, f2, UnifiedSystemConstants.INTERACTION_STRENGTH)
        phi_unity_history[t] = QuantumHarmonics.calculate_phi_unity(
            t, f1_entangled, f2_entangled)
        
        # Adaptive learning rate evolution
        UnifiedSystemConstants.LEARNING_RATE += 0.05 * (1 - phi_unity_history[t])
        
        print(f"t={t}: Φ_unity={phi_unity_history[t]:.4f}, η={UnifiedSystemConstants.LEARNING_RATE:.4f}")

    # Generate visual manifestations
    print("\nManifesting Fractal Duality...")
    fractal_ani = FractalManifold.create_fractal_animation()
    plt.show()
    fractal_ani._stop()

    print("\nWeaving Unity Manifold...")
    manifold_ani = FractalManifold.create_manifold_visualization(phi_unity_history)
    plt.show()
    manifold_ani._stop()

    # Generate insights and narrative
    UnityAnalysis.generate_symbolic_insights(phi_unity_history)
    UnityAnalysis.generate_narrative(phi_unity_history)
    
    print("\nUnity Convergence Protocol Completed.")

if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    print(f"\nConvergence Time: {time.perf_counter() - start_time:.2f}s")
    

    
    
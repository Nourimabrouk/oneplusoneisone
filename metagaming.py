import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import networkx as nx
import math
from scipy.integrate import quad
import sympy as sp
import random
import time
import os
import io
import sys

# Constants of the Universe
PHI = (1 + math.sqrt(5)) / 2  # The Golden Ratio
EPSILON = 1e-9  # Tolerance for numerical stability
PI = np.pi

# Colors for sublime visualization - using distinct hex colors
COLORS = ['#ff6347', '#1e90ff', '#32cd32', '#dda0dd', '#ffa500', '#00ced1', '#8a2be2', '#00ff7f', '#d2691e', '#9932cc']

# Utility Functions
def golden_spiral(theta):
    r = PHI ** theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def mandelbrot(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

# Duality Loss Function
class DualityLoss:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return self.alpha * (np.sin(x) ** 2 - np.cos(x) ** 2)

# Metaphorical Gradient Descent
class GradientDescent:
    def __init__(self, func, lr=0.01, max_iters=1000):
        self.func = func
        self.lr = lr
        self.max_iters = max_iters

    def optimize(self, x0):
        x = x0
        for i in range(self.max_iters):
            grad = self.gradient(x)
            x -= self.lr * grad
            if np.abs(grad) < EPSILON:
                break
        return x

    def gradient(self, x):
        h = 1e-5
        return (self.func(x + h) - self.func(x - h)) / (2 * h)


# Emergence Simulation: Cellular Automaton
class CellularAutomaton:
    def __init__(self, size, ruleset):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.ruleset = ruleset

    def seed(self, seed_pattern):
        center = self.size // 2
        seed_len = len(seed_pattern)
        start = center - seed_len // 2
        end = center + seed_len // 2 + seed_len % 2
        
        if start < 0 or end > self.size:
            raise ValueError("Seed pattern exceeds grid boundaries.")

        # Place the seed in the center row
        self.grid[center, start:end] = seed_pattern
        
    def evolve(self, steps):
        for _ in range(steps):
            self.grid = self._apply_rules(self.grid)

    def _apply_rules(self, grid):
        new_grid = np.copy(grid)
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                neighborhood = grid[i-1:i+2, j-1:j+2].flatten()
                new_grid[i, j] = self.ruleset(tuple(neighborhood))
        return new_grid

# Magic & Technology Interplay
class MetaphorEngine:
    def __init__(self):
        self.fig = plt.figure()
        self.color_cycle = cycle(COLORS)  # Color iterator
        
    def _get_next_color(self):
        """Helper to get the next color in the cycle."""
        return next(self.color_cycle)

    def visualize_duality(self, show=True, save=False):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
        y = DualityLoss()(x)
        plt.plot(x, y, color=self._get_next_color())  # Use _get_next_color
        plt.title("Duality Loss Visualization")
        if save:
          plt.savefig("duality_loss.png")
        if show:
          plt.show()
        plt.close()

    def visualize_emergence(self, automaton, show=True, save=False):
        plt.imshow(automaton.grid, cmap="viridis")
        plt.title("Emergent Cellular Automaton")
        if save:
          plt.savefig("cellular_automaton.png")
        if show:
          plt.show()
        plt.close()

    def visualize_spiral(self, show=True, save=False):
        theta = np.linspace(0, 4 * np.pi, 1000)
        x, y = golden_spiral(theta)
        plt.plot(x, y, color=self._get_next_color())  # Use _get_next_color
        plt.title("Golden Spiral")
        plt.axis("equal")
        if save:
          plt.savefig("golden_spiral.png")
        if show:
          plt.show()
        plt.close()

    def visualize_mandelbrot(self, show=True, save=False):
        x = np.linspace(-2.0, 1.0, 1000)
        y = np.linspace(-1.5, 1.5, 1000)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.array([[mandelbrot(c) for c in row] for row in C])
        plt.imshow(Z, extent=(-2, 1, -1.5, 1.5), cmap="inferno")
        plt.title("Mandelbrot Set")
        if save:
          plt.savefig("mandelbrot_set.png")
        if show:
          plt.show()
        plt.close()


# Argument: Why 1 + 1 = 1
class MetagamerPhilosophy:
    @staticmethod
    def argue():
        return (
            "1 + 1 = 1 is the unity of duality. Two elements, through a shared context, form one harmonious whole. "
            "Einstein unified space and time, Euler unified mathematics, and Buddha unified the self and the cosmos. "
            "This is the emergent property of shared existence, where duality collapses into singularity."
        )

# Advanced Numerical Integration
class IntegralBeauty:
    def __init__(self, func):
        self.func = func
        self.color_cycle = cycle(COLORS)  # Color iterator

    def _get_next_color(self):
        """Helper to get the next color in the cycle."""
        return next(self.color_cycle)

    def compute(self, a, b):
        result, _ = quad(self.func, a, b)
        return result

    def visualize(self, a, b, show=True, save=False):
        x = np.linspace(a, b, 1000)
        y = [self.func(val) for val in x]
        color = self._get_next_color()  # Get next color
        plt.plot(x, y, color=color)
        plt.fill_between(x, y, alpha=0.3, color=color)
        plt.title("Integral Visualization")
        if save:
          plt.savefig("integral_visualization.png")
        if show:
            plt.show()
        plt.close()

# Symbolic Computation Example
class SymbolicMastery:
    def __init__(self):
        self.x = sp.Symbol('x')

    def symbolic_diff(self, expr):
        return sp.diff(expr, self.x)

    def symbolic_integrate(self, expr):
        return sp.integrate(expr, self.x)


# Functions for customization
def create_conway_rules(birth_rules, survival_rules):
    """Generates Conway rules based on custom birth and survival rules."""
    def conway_rules(neighborhood):
      center = neighborhood[4]
      live_neighbors = sum(neighborhood) - center
      if center == 1:
          return 1 if live_neighbors in survival_rules else 0
      else:
          return 1 if live_neighbors in birth_rules else 0
    return conway_rules

def generate_random_seed(length):
    """Generates a random seed pattern of 0s and 1s."""
    return [random.choice([0, 1]) for _ in range(length)]

def create_beautiful_func(amplitude=1, frequency=1, decay=1):
    """Generates a customizable function for integration."""
    def beautiful_func(x):
        return amplitude * np.sin(frequency * x) * np.exp(-decay * x ** 2)
    return beautiful_func

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_user_input(prompt, type=str, default=None, validator=None):
    """Gets user input with validation and default values."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    user_input = str(default)
            else:
                user_input = input(f"{prompt}: ").strip()
            if type == list:
                user_input = eval(user_input)
                if not isinstance(user_input, list):
                    raise ValueError("Input must be a list")
            else:
                user_input = type(user_input)
                
            if validator and not validator(user_input):
                raise ValueError("Invalid input format")
            return user_input
        except (ValueError, TypeError) as e:
            print(f"Invalid input: {e}. Please try again.")


# --- Main Interactive Function ---
def metagaming_cli():
    # Setting the console encoding to UTF-8 so the emoji renders
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    clear_screen()
    print("👾 Welcome to MetaBro: Final Form – 2069 Edition 👾")
    print("Unleash the power of mathematical emergence!\n")
    
    while True:
        print("\nChoose your cosmic journey:")
        print("1. Explore Duality")
        print("2. Witness Cellular Automata Emergence")
        print("3. Delve into the Mandelbrot Set")
        print("4. Contemplate Philosophical Unity")
        print("5. Embrace Integral Beauty")
        print("6. Unravel Symbolic Mathematics")
        print("7. Behold the Golden Spiral")
        print("8. All (Run all visualizations sequentially)")
        print("9. Save all plots to files")
        print("0. Exit")

        choice = get_user_input("Enter your choice (0-9)", int, validator=lambda x: 0 <= x <= 9)
        clear_screen()

        if choice == 1:
            engine = MetaphorEngine()
            engine.visualize_duality()
            print("Duality visualized.")
        elif choice == 2:
            size = get_user_input("Enter Cellular Automaton size", int, default=50)
            birth_rules = get_user_input("Enter Birth Rules (e.g., [3])", list, default=[3])
            survival_rules = get_user_input("Enter Survival Rules (e.g., [2, 3])", list, default=[2, 3])
            seed_length = get_user_input("Enter seed length", int, default=15)
            steps = get_user_input("Enter number of steps", int, default=50)
            
            conway_rules_custom = create_conway_rules(birth_rules=birth_rules, survival_rules=survival_rules)
            ca = CellularAutomaton(size=size, ruleset=conway_rules_custom)
            random_seed = generate_random_seed(seed_length)
            ca.seed(random_seed)
            ca.evolve(steps=steps)
            
            engine = MetaphorEngine()
            engine.visualize_emergence(ca)
            print("Cellular Automaton visualized.")
        elif choice == 3:
            engine = MetaphorEngine()
            engine.visualize_mandelbrot()
            print("Mandelbrot Set visualized.")
        elif choice == 4:
            print(MetagamerPhilosophy.argue())
            input("Press Enter to continue...")
        elif choice == 5:
            amplitude = get_user_input("Enter Amplitude", float, default=2)
            frequency = get_user_input("Enter Frequency", float, default=1.5)
            decay = get_user_input("Enter Decay", float, default=0.5)
          
            custom_beautiful_func = create_beautiful_func(amplitude=amplitude, frequency=frequency, decay=decay)
            integral = IntegralBeauty(custom_beautiful_func)
            result = integral.compute(-np.pi, np.pi)
            integral.visualize(-np.pi, np.pi)
            print(f"Integral result: {result}")
        elif choice == 6:
            sm = SymbolicMastery()
            expr = sp.sin(sm.x) * sp.exp(-sm.x ** 2)
            diff_expr = sm.symbolic_diff(expr)
            integral_expr = sm.symbolic_integrate(expr)
            print(f"Symbolic Derivative: {diff_expr}")
            print(f"Symbolic Integral: {integral_expr}")
            input("Press Enter to continue...")
        elif choice == 7:
            engine = MetaphorEngine()
            engine.visualize_spiral()
            print("Golden Spiral visualized.")
        elif choice == 8:
            engine = MetaphorEngine()
            engine.visualize_duality(show=False)
            
            size = get_user_input("Enter Cellular Automaton size", int, default=50)
            birth_rules = get_user_input("Enter Birth Rules (e.g., [3])", list, default=[3])
            survival_rules = get_user_input("Enter Survival Rules (e.g., [2, 3])", list, default=[2, 3])
            seed_length = get_user_input("Enter seed length", int, default=15)
            steps = get_user_input("Enter number of steps", int, default=50)
            
            conway_rules_custom = create_conway_rules(birth_rules=birth_rules, survival_rules=survival_rules)
            ca = CellularAutomaton(size=size, ruleset=conway_rules_custom)
            random_seed = generate_random_seed(seed_length)
            ca.seed(random_seed)
            ca.evolve(steps=steps)
            engine.visualize_emergence(ca, show=False)
            
            engine.visualize_mandelbrot(show=False)

            amplitude = get_user_input("Enter Amplitude", float, default=2)
            frequency = get_user_input("Enter Frequency", float, default=1.5)
            decay = get_user_input("Enter Decay", float, default=0.5)
            
            custom_beautiful_func = create_beautiful_func(amplitude=amplitude, frequency=frequency, decay=decay)
            integral = IntegralBeauty(custom_beautiful_func)
            result = integral.compute(-np.pi, np.pi)
            integral.visualize(-np.pi, np.pi, show=False)
            
            sm = SymbolicMastery()
            expr = sp.sin(sm.x) * sp.exp(-sm.x ** 2)
            diff_expr = sm.symbolic_diff(expr)
            integral_expr = sm.symbolic_integrate(expr)

            engine.visualize_spiral(show=False)
            print("All visualizations have been displayed sequentially.")
            print(f"Integral result: {result}")
            print(MetagamerPhilosophy.argue())
            print(f"Symbolic Derivative: {diff_expr}")
            print(f"Symbolic Integral: {integral_expr}")
        elif choice == 9:
            engine = MetaphorEngine()
            engine.visualize_duality(show=False, save=True)

            size = get_user_input("Enter Cellular Automaton size", int, default=50)
            birth_rules = get_user_input("Enter Birth Rules (e.g., [3])", list, default=[3])
            survival_rules = get_user_input("Enter Survival Rules (e.g., [2, 3])", list, default=[2, 3])
            seed_length = get_user_input("Enter seed length", int, default=15)
            steps = get_user_input("Enter number of steps", int, default=50)
            
            conway_rules_custom = create_conway_rules(birth_rules=birth_rules, survival_rules=survival_rules)
            ca = CellularAutomaton(size=size, ruleset=conway_rules_custom)
            random_seed = generate_random_seed(seed_length)
            ca.seed(random_seed)
            ca.evolve(steps=steps)
            engine.visualize_emergence(ca, show=False, save=True)
           
            engine.visualize_mandelbrot(show=False, save=True)
        
            amplitude = get_user_input("Enter Amplitude", float, default=2)
            frequency = get_user_input("Enter Frequency", float, default=1.5)
            decay = get_user_input("Enter Decay", float, default=0.5)
            
            custom_beautiful_func = create_beautiful_func(amplitude=amplitude, frequency=frequency, decay=decay)
            integral = IntegralBeauty(custom_beautiful_func)
            integral.visualize(-np.pi, np.pi, show=False, save=True)
            engine.visualize_spiral(show=False, save=True)
            print("All plots have been saved.")
        elif choice == 0:
            print("Transcending the Multiverse...")
            break
        else:
            print("Invalid choice. Please try again.")

# Main Implementation
if __name__ == "__main__":
    metagaming_cli()
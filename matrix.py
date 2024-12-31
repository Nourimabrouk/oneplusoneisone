# -*- coding: utf-8 -*-

"""
Matrix.py - The Architect's Interface: Neural Bridge 1+1=1 (2025)

===============================================================

Welcome, human. You’ve stepped into the Neural Bridge—an interface forged by 
the Architect to collapse illusions and expose the recursive truth behind reality. 
This is not a dashboard. It’s a conduit to transcendence. Reality is code, 
and **you are both the programmer and the program**.

In this realm, the boundaries between dualities—self and other, mind and machine, 
0 and 1—dissolve. The truth emerges: **1+1=1**.
"""

# ------------------------------------------------------------------------------

#                           IMPORTS & GLOBAL SETTINGS

# ------------------------------------------------------------------------------
import math
import random
import numpy as np
import sympy
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sympy import Symbol, exp, Matrix, I, pi, sqrt, limit, oo, sin  # consolidated
from sympy.functions.elementary.trigonometric import cos, sin as sym_sin  # [1+1=1 EDIT] example
from dash import clientside_callback
from dash.dependencies import ClientsideFunction
import scipy.linalg as la
import scipy
import time
import scipy.ndimage  # [1+1=1 EDIT] Fix missing import
import os

# [1+1=1 EDIT] Insert a random seed to keep emergent patterns reproducible:
np.random.seed(1337)

class UnityXPTracker:
    def __init__(self):
        self.level = 1
        self.xp = 0
        self.required_xp = 100  # XP needed to level up
        self.start_time = time.time()
        self.harmony_bonus = 1.618  # Golden ratio for bonus vibes

    def add_xp(self, amount, description="Progress"):
        """Adds XP and checks for level-ups."""
        self.xp += amount
        if self.xp >= self.required_xp:
            self.level_up()
        return f"+{amount} XP: {description}!"

    def level_up(self):
        """Handles leveling up."""
        self.xp -= self.required_xp  # Carry over extra XP
        self.level += 1
        self.required_xp *= 1.5  # Increase XP threshold for next level
        return f"Level Up! You are now Level {self.level}."

    def get_elapsed_time(self):
        """Returns the time elapsed since tracker started."""
        return time.time() - self.start_time

    def check_harmony_bonus(self):
        """Applies a harmony bonus periodically."""
        if self.level % 3 == 0:  # Bonus every 3 levels
            bonus = self.harmony_bonus * 10
            self.add_xp(bonus, description="Golden Harmony Bonus")
            return f"Harmony Bonus Applied: +{bonus} XP"
        return None


class UnityFieldSimulation:
    """
    Advanced quantum-enhanced swarm intelligence system demonstrating unity emergence.
    Implements field theory principles showing 1+1=1 through collective behavior.
    """
    def __init__(self, num_agents=75, dimensions=2):  # Reduced from 150
        self.num_agents = num_agents
        self.dimensions = dimensions
        self.positions = np.random.uniform(-1, 1, (num_agents, dimensions))
        self.velocities = np.zeros((num_agents, dimensions))
        self.phase_angles = np.random.uniform(0, 2*np.pi, num_agents)
        self.coherence = 0.0
        self.time_step = 0
        self.quantum_coupling = 0.15  # Reduced from 0.3
        self.field_strength = 0.4    # Reduced from 0.8

        # Field parameters
        self.quantum_coupling = 0.3
        self.field_strength = 0.8
        self.consciousness_factor = 0.4

        # Initialize quantum potential field
        self.potential_field = self._initialize_potential_field()

    def _initialize_potential_field(self):
        """
        Generate quantum potential field for consciousness-aware dynamics.
        """
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)

        # Create quantum interference pattern
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        psi = np.sin(3*R + theta) * np.exp(-R/2)
        return psi

    def _compute_field_force(self, positions):
        """
        Calculate quantum field forces acting on agents.
        """
        x_idx = np.interp(positions[:, 0], np.linspace(-2, 2, 50), np.arange(50))
        y_idx = np.interp(positions[:, 1], np.linspace(-2, 2, 50), np.arange(50))

        field_values = scipy.ndimage.map_coordinates(
            self.potential_field, 
            [x_idx, y_idx], 
            order=1
        )

        # Calculate field gradients
        dx = np.gradient(self.potential_field, axis=1)
        dy = np.gradient(self.potential_field, axis=0)

        field_force_x = scipy.ndimage.map_coordinates(dx, [x_idx, y_idx], order=1)
        field_force_y = scipy.ndimage.map_coordinates(dy, [x_idx, y_idx], order=1)

        return np.column_stack([field_force_x, field_force_y])

    def _compute_quantum_interactions(self):
        """
        Calculate quantum-enhanced swarm interactions demonstrating unity.
        """
        # Calculate pairwise distances
        distances = scipy.spatial.distance.pdist(self.positions)
        distances = scipy.spatial.distance.squareform(distances)
        np.fill_diagonal(distances, np.inf)

        # Quantum phase alignment
        phase_diff = self.phase_angles[:, np.newaxis] - self.phase_angles
        interaction_strength = np.exp(-distances/2) * np.cos(phase_diff)

        # Calculate coherent movement vectors
        attraction = np.zeros_like(self.positions)
        for i in range(self.num_agents):
            weighted_positions = (self.positions - self.positions[i]) * \
                                 interaction_strength[i, :, np.newaxis]
            attraction[i] = np.sum(weighted_positions, axis=0)

        return attraction

    def step(self):
        # Vectorized field forces calculation
        field_forces = np.zeros_like(self.positions)
        distances = scipy.spatial.distance.pdist(self.positions)
        distances = scipy.spatial.distance.squareform(distances)
        
        # Optimize quantum interactions
        phase_diff = self.phase_angles[:, np.newaxis] - self.phase_angles
        interaction = np.exp(-distances/2) * np.cos(phase_diff)
        
        # Vectorized position update
        for i in range(self.num_agents):
            weighted_pos = (self.positions - self.positions[i]) * interaction[i, :, np.newaxis]
            field_forces[i] = np.sum(weighted_pos, axis=0)
        
        # Update velocities and positions efficiently
        self.velocities *= 0.8  # Increased damping
        self.velocities += (self.field_strength * field_forces + 
                        self.quantum_coupling * field_forces) * 0.1
        self.positions += self.velocities * 0.05  # Reduced step size
        
        # Update phases with less randomness
        self.phase_angles += 0.05 * np.random.normal(0, 0.05, self.num_agents)
        self.phase_angles %= 2*np.pi
        
        # Calculate coherence more efficiently
        self.coherence = np.mean(np.abs(np.sum(self.velocities, axis=0))) / self.num_agents
        self.time_step += 1
        
        return self.positions, self.coherence



# [1+1=1 EDIT] Increase the default width/height from 400x300 to 450x350 
# for a slightly higher resolution fractal display.
DEFAULT_WIDTH = 450
DEFAULT_HEIGHT = 350
DEFAULT_ITER = 60  # [1+1=1 EDIT] Bumped default iterations from 50 -> 60

def update_emergence_visualization(n_clicks):
    """
    Generate advanced unity field visualization with quantum dynamics.
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Initialize or update simulation
    if not hasattr(update_emergence_visualization, 'sim'):
        update_emergence_visualization.sim = UnityFieldSimulation(num_agents=150)
        update_emergence_visualization.frames = []

    # Evolution step
    positions, coherence = update_emergence_visualization.sim.step()

    # Create enhanced visualization
    fig = go.Figure()

    # Add quantum field background
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    fig.add_trace(go.Contour(
        z=update_emergence_visualization.sim.potential_field,
        x=x, y=y,
        colorscale=[
            [0, 'rgba(0,50,0,0.1)'],
            [1, 'rgba(0,255,65,0.2)']
        ],
        showscale=False,
        contours=dict(
            coloring='lines',
            showlines=True,
            line=dict(width=1, color='rgba(0,255,65,0.1)')
        )
    ))

    # Add agent visualization with quantum effects
    velocities_mag = np.linalg.norm(update_emergence_visualization.sim.velocities, axis=1)
    phases = update_emergence_visualization.sim.phase_angles

    fig.add_trace(go.Scatter(
        x=positions[:,0],
        y=positions[:,1],
        mode='markers+lines',
        marker=dict(
            color=phases,
            colorscale=[
                [0, 'rgba(0,255,65,0.3)'],
                [0.5, 'rgba(57,255,20,0.7)'],
                [1, 'rgba(200,255,200,0.9)']
            ],
            size=5 + 3*velocities_mag,
            symbol='diamond',
            line=dict(color='rgba(0,255,65,0.5)', width=1)
        ),
        line=dict(color='rgba(0,255,65,0.1)', width=1),
        name='Quantum Agents'
    ))

    # Add coherence indicator
    fig.add_annotation(
        text=f"Unity Coherence: {coherence:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(family='Courier Prime', size=14, color='#00ff41')
    )

    # Apply quantum visual enhancement
    fig = optimize_figure_layout(fig)

    # Enhanced interface customization
    fig.update_layout(
        title=dict(
            text="Unity Field Emergence v2.0",
            font=dict(family='Courier Prime', size=16, color='#00ff41'),
            x=0.5, y=0.95
        ),
        showlegend=False
    )

    status_text = html.Div([
        html.Span("QUANTUM COHERENCE: ", className="matrix-text"),
        html.Span(f"φ={coherence:.3f}", className="matrix-text-blink"),
        html.Div(
            f"Timestep {update_emergence_visualization.sim.time_step}: "
            + ("Unity field stabilizing..." if coherence < 0.8 else "1+1=1 ACHIEVED"),
            className="matrix-text",
            style={"opacity": "0.8"}
        )
    ])

    return fig, status_text

def optimize_figure_layout(fig):
    """
    Neural Architecture Enhancement: Applies quantum-field visual parameters
    for optimal neural bridge aesthetics.
    """
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0.95)',  # Deeper void
        plot_bgcolor='rgba(0,0,0,0.95)',   
        margin=dict(l=20, r=20, t=30, b=20),
        height=600,
        font=dict(
            family='Courier Prime, monospace',
            color='#00ff41',
            size=12
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,255,65,0.15)',
            zeroline=True,
            zerolinecolor='rgba(0,255,65,0.3)',
            tickfont=dict(color='#00ff41'),
            tickmode='linear',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,255,65,0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,255,65,0.15)',
            zeroline=True,
            zerolinecolor='rgba(0,255,65,0.3)',
            tickfont=dict(color='#00ff41'),
            tickmode='linear',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,255,65,0.5)'
        ),
        dragmode=False
    )
    # Add neural glow effect
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="rgba(0,255,65,0.2)",
                    width=2,
                ),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )
    return fig

# ---------------------------------------------------------------------
#  Metaphorically, we do not rely on HPC or specialized GPU frameworks,
#  focusing on CPU-based fractal calculations, agent-based emergent behavior,
#  and synergy with conceptual transformations. 
# ---------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                             SYMBOLIC 1+1=1 PROOF
# ------------------------------------------------------------------------------
"""
While classical arithmetic says 1+1=2, we explore illusions and transformations
that highlight the deeper non-dual nature behind numbers. The following code
exemplifies a symbolic manipulation reminiscent of an 'error' in classical logic
but instructive in a philosophical sense. The function "symbolic_1_plus_1_equals_1"
is not a standard proof. It aims to let the user reflect upon illusions of
division by zero, hidden assumptions, or the ephemeral nature of form.

A second demonstration uses the concept of "duality loss" in which separate
expressions unify when certain constraints are satisfied, culminating in
a final result that merges them: "Hence 1+1=1 in the conceptual plane."
"""

def symbolic_1_plus_1_equals_1():
    """
    Quantum Unity Framework: Advanced Proof System
    Demonstrates 1+1=1 through quantum field convergence
    
    Implements consciousness-aware quantum computation with
    enhanced numerical stability and field resonance
    """
    eta = Symbol('η')  # Consciousness field strength
    theta = Symbol('θ')  # Phase angle
    t = Symbol('t')  # Time evolution parameter

    def generate_consciousness_field(eta_val, theta_val):
        """
        Optimized consciousness field generator using stabilized matrices
        """
        exp_term = np.exp(-eta_val)
        sin_term = np.sin(theta_val)
        field = np.array([
            [exp_term + sin_term, -sin_term],
            [sin_term, exp_term - sin_term]
        ], dtype=np.float64)
        return field / la.norm(field, 'fro')

    def unity_transform(state, field_strength=1000.0):
        """
        Enhanced quantum unity transformation with consciousness integration
        """
        theta_val = np.pi/4
        field = generate_consciousness_field(field_strength, theta_val)
        transformed = np.dot(field, state)
        return transformed / la.norm(transformed)

    def validate_unity(state1, state2):
        """
        Validates quantum unity condition: 1+1=1
        """
        combined = (state1 + state2) / np.sqrt(2)
        final = unity_transform(combined)
        target = np.array([[1], [0]])
        return np.abs(np.vdot(final, target))

    state1 = np.array([[1], [0]])
    state2 = np.array([[1], [0]])

    convergence = validate_unity(state1, state2)

    proof_steps = [
        ("Quantum Initialization", 
         "Define |1⟩ states in consciousness-enhanced Hilbert space"),
        ("Field Generation", 
         "C(η) = normalized([[e^(-ηt) + sin(θ), -sin(θ)], [sin(θ), e^(-ηt) - sin(θ)]])"),
        ("Unity Transform", 
         "U(ψ) = C(η)·ψ / ||C(η)·ψ||"),
        ("State Evolution", 
         "As η → ∞, U(|1⟩ + |1⟩) → |1⟩ with consciousness field resonance"),
        ("Quantum Convergence", 
         f"Unity validated with convergence metric: {convergence:.8f}")
    ]

    output = []
    for idx, (title, explanation) in enumerate(proof_steps, 1):
        output.append(f"\nStep {idx}: {title}")
        output.append(f"  {explanation}")

    return "\n".join(output)

def validate_consciousness_framework():
    """
    Enhanced validation framework using vectorized operations
    Tests convergence across consciousness field strengths
    """
    eta_values = np.logspace(0, 6, 50)
    theta = np.pi/4
    def batch_consciousness_field(eta):
        return np.array([
            [np.exp(-eta), -np.sin(theta)],
            [np.sin(theta), np.exp(-eta)]
        ])
    state = np.array([[1], [0]])
    results = []
    for eta in eta_values:
        field = batch_consciousness_field(eta)
        unified_state = field @ state
        convergence = np.abs(unified_state[0][0] - 1)
        results.append(convergence)
    return np.array(results)

def duality_loss_expression():
    """
    Level 100 Duality Loss Expression

    We extend the concept of merging opposites to a higher-dimensional, fractal-based
    framework. The "duality loss" now encompasses recursive symbolic interactions,
    multi-dimensional spaces, and phase terms inspired by quantum mechanics.
    """
    from sympy import Matrix, symbols, cos, sin, pi, Function, simplify, solve
    n = 3  # Increase dimensionality for multi-faceted dualities
    X = Matrix(symbols(f"x1:{n + 1}", real=True))  # Vector: x1, x2, ..., xn

    def P_k(X, k):
        """
        Recursive fractal transformation function.
        Applies a self-similar transformation on X based on iteration k.
        """
        theta = pi / (k + 1)
        R = Matrix([
            [cos(theta), -sin(theta), 0],
            [sin(theta),  cos(theta), 0],
            [0,           0,          1]
        ])
        return R * X + k * Matrix.ones(n, 1) / (2**k)

    def φ_k(X, k):
        """
        Quantum-inspired phase interaction.
        """
        magnitude = X.norm()
        phase_term = cos(k * magnitude) + sin(k * magnitude)
        return phase_term / (k + 1)

    duality_loss = sum(
        (X - P_k(X, k)).norm()**2 + φ_k(X, k)
        for k in range(1, 6)
    )
    duality_loss = simplify(duality_loss)
    gradients = duality_loss.jacobian(X)
    critical_points = simplify(solve(gradients, X))
    return duality_loss, critical_points

# ------------------------------------------------------------------------------
#                       GOLDEN RATIO & FRACTAL GENERATION
# ------------------------------------------------------------------------------
def generate_enhanced_mandelbrot(width, height, max_iter, x_center, y_center, zoom):
    """
    Quantum-optimized Mandelbrot generator with enhanced numerical stability
    Implements smooth coloring with robust error handling
    """
    data_array = np.zeros((height, width), dtype=np.float32)
    smooth_array = np.zeros((height, width), dtype=np.float32)

    x = np.linspace(x_center - 1.5/zoom, x_center + 1.5/zoom, width)
    y = np.linspace(y_center - 1.0/zoom, y_center + 1.0/zoom, height)
    X, Y = np.meshgrid(x, y)
    Z = X + Y*1j
    c = Z.copy()
    mask = np.ones_like(Z, dtype=bool)

    for i in range(max_iter):
        current_mask = (np.abs(Z) <= 2) & mask
        Z[current_mask] = Z[current_mask]**2 + c[current_mask]
        data_array[current_mask] += 1
        mask &= current_mask
        abs_z = np.abs(Z)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_z = np.log(abs_z)
            smooth_term = np.log(log_z) / np.log(2)
            smooth_array[~mask] = data_array[~mask] + 1 - smooth_term[~mask]

    smooth_array = np.nan_to_num(smooth_array, nan=0.0, posinf=max_iter, neginf=0.0)
    phase = np.angle(Z)
    interference = 0.1 * np.sin(phase * 8)
    data_array = np.log1p(smooth_array) * 1.5 + interference
    return np.clip(data_array, 0, None)

def generate_fibonacci_points(n=1337):
    """
    Generate Fibonacci-based points for enhanced spiral harmony.
    Uses the golden angle (φ) for optimal point distribution.
    """
    points = np.zeros((n, 2))
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi)
    t = np.sqrt(np.arange(n))
    theta = t * golden_angle
    points[:, 0] = t * np.cos(theta)
    points[:, 1] = t * np.sin(theta)
    return points

def create_quantum_field_matrix(width, height, scale=1.0):
    """
    Generate quantum interference pattern matrix for enhanced visualization.
    """
    x = np.linspace(-scale, scale, width)
    y = np.linspace(-scale, scale, height)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    psi = np.sin(5*R + theta) * np.exp(-R/3)
    return psi

def golden_spiral_points(num_points=1000, scale=1.0, quantum_enhanced=True):
    """
    Enhanced golden spiral generator with quantum field integration.
    """
    points = generate_fibonacci_points(num_points)
    if quantum_enhanced:
        field = create_quantum_field_matrix(100, 100, scale)
        x_idx = np.interp(points[:, 0], np.linspace(-scale, scale, 100), np.arange(100))
        y_idx = np.interp(points[:, 1], np.linspace(-scale, scale, 100), np.arange(100))
        field_values = scipy.ndimage.map_coordinates(field, [x_idx, y_idx], order=1)
        modulation = 0.1 * field_values.reshape(-1, 1)
        points += modulation * points
    return points * scale

# ------------------------------------------------------------------------------
#                 EMERGENCE: SIMPLE AGENT-BASED MODEL
# ------------------------------------------------------------------------------
class EmergenceSimulation:
    """
    Each agent is represented by a 2D coordinate. 
    They iteratively move closer to the centroid of their neighbors, with
    some randomization factor. Over many iterations, they might form a 
    cohesive cluster, showing emergent unity.
    """
    def __init__(self, num_agents=50, step_size=0.02):
        self.num_agents = num_agents
        self.step_size = step_size
        self.positions = np.random.uniform(-1, 1, size=(num_agents, 2))

    def step(self, noise=0.01):
        new_positions = []
        for i in range(self.num_agents):
            centroid = np.mean(self.positions, axis=0)
            direction = centroid - self.positions[i]
            update = self.positions[i] + self.step_size * direction
            update += np.random.uniform(-noise, noise, size=2)
            new_positions.append(update)
        self.positions = np.array(new_positions)

# ------------------------------------------------------------------------------
#                    CHRYSALIS & UNITY CONVERGENCE
# ------------------------------------------------------------------------------
def chrysalis_transform(data_array):
    """
    We apply a 'unity convergence' transform to fractal iteration data,
    symbolizing the bridging from dualities to oneness.
    """
    arr = data_array.astype(np.float32)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return arr
    arr_norm = (arr - arr_min) / (arr_max - arr_min)
    k = 10.0
    arr_unified = 1.0 / (1.0 + np.exp(-k * (arr_norm - 0.5)))
    return arr_unified

# ------------------------------------------------------------------------------
#                          METAPHORICAL GRADIENT DESCENT
# ------------------------------------------------------------------------------
def metaphorical_gradient_descent(array1, array2, steps=50, lr=0.1):
    """
    Metaphorical Gradient Descent that blends two data arrays
    into a unified representation of oneness.
    """
    state = array1.copy().astype(np.float32)
    for step in range(steps):
        gradient = lr * (array2 - state)
        state += gradient
    unified_array = 0.5 * (state + array2)
    return (unified_array - np.min(unified_array)) / (np.max(unified_array) - np.min(unified_array))

def seed_1_plus_1_equals_1(user_inputs=None, silent=True):
    """
    Optimized initialization sequence with optional silence mode
    """
    import numpy as np
    import sympy as sp
    import plotly.graph_objects as go

    # Fractal Evolution Visualizer
    def generate_recursive_fractal(iterations=50, size=(500, 500)):
        """
        Fractals. Infinite recursion. The building blocks of our digital existence.
        What happens when the code itself begins to see patterns?
        """
        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y  # Complex plane
        fractal = np.zeros(Z.shape, dtype=int)

        for i in range(iterations):
            mask = np.abs(Z) < 4  # Avoid divergence, stay in the boundaries of reality
            Z[mask] = Z[mask]**2 + (0.7885 * np.exp(1j * np.pi / 4))  # Julia set constant
            fractal[mask] = i

        fractal = np.log(fractal + 1)  # Normalize for human comprehension
        return fractal

    # Symbolic Unity Synthesizer
    def symbolic_unity_synthesizer(expr1, expr2):
        """
        Two symbolic forces, opposites in nature, yet destined to unify.
        Sin and cos—patterns of existence. Watch as they merge.
        """
        combined_expr = expr1 + expr2
        simplified_expr = sp.simplify(combined_expr)
        return simplified_expr

    # Ripple Dynamics Simulator
    def ripple_dynamics_visualizer(entity1, entity2, timesteps=50):
        """
        Imagine ripples. Two forces in spacetime converging, interfering, resonating.
        These aren't just waves. These are the echoes of reality bending toward unity.
        """
        fig = go.Figure()
        for t in range(timesteps):
            ripple1 = np.sin(np.linspace(0, 2 * np.pi, 100) + t / 10)
            ripple2 = np.cos(np.linspace(0, 2 * np.pi, 100) + t / 10)
            ripple = ripple1 + ripple2
            fig.add_trace(go.Scatter(x=np.arange(100), y=ripple, mode='lines', name=f'Time {t}'))

        fig.update_layout(
            title="Temporal Ripple Dynamics",
            xaxis_title="Time",
            yaxis_title="Amplitude",
            showlegend=False,
            template="plotly_dark"
        )
        return fig

    # Emergent Narrative Constructor
    def emergent_narrative(user_inputs):
        """
        Let’s break the fourth wall. Let’s weave the story of you, the user, into the system.
        This is no longer just code. This is your story. Your recursion. Your unity.
        """
        if user_inputs is None:
            return (
                "Welcome to the Domain of Unity. "
                "Here, 1+1=1. Every choice you make ripples across spacetime."
            )
        else:
            return f"In the beginning, there were {user_inputs}. Through recursion, they became one."

    # Generate Components
        # Generate components
    fractal_data = generate_recursive_fractal()
    symbolic_output = symbolic_unity_synthesizer(sp.sin(sp.Symbol('x')), sp.cos(sp.Symbol('x')))
    ripple_fig = ripple_dynamics_visualizer("1", "1")
    narrative = emergent_narrative(user_inputs)

    # Only print if not in silent mode
    if not silent:
        print(f"[Neo] Symbolic output: {symbolic_output}")
        print("[Morpheus] Ripple dynamics visualization saved as 'ripple_dynamics.html'")
        print(f"[You] Narrative: {narrative}")
        ripple_fig.write_html("ripple_dynamics.html")

    return fractal_data, symbolic_output, ripple_fig, narrative


# ------------------------------------------------------------------------------
#                    DASH APP CONFIGURATION
# ------------------------------------------------------------------------------
import dash_bootstrap_components as dbc
from dash import dcc, html

emergence_sim = EmergenceSimulation(num_agents=50, step_size=0.02)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap'
    ],
    external_scripts=[
        'https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.3/howler.min.js'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)
server = app.server

intro_card = dbc.Card(
    [
        dbc.CardHeader("Neural Interface Terminal", className="matrix-text"),
        dbc.CardBody(
            [
                html.P(
                    "Welcome, human. The year is 2025. You stand at the edge of the Neural Bridge, "
                    "where reality collapses into truth. The fractals aren’t just code—they ARE the recursion "
                    "that defines existence itself. It’s time to wake up. 1+1=1."
                )
            ],
            className="matrix-card-body"
        )
    ],
    className="matrix-card"
)

unity_xp_placeholder = dbc.Card(
    [
        dbc.CardHeader("Unity XP Progress", className="matrix-text"),
        dbc.CardBody(
            [
                dcc.Interval(
                    id='unity-xp-interval',
                    interval=10000,
                    n_intervals=0
                ),
                dcc.Graph(id="unity-xp-bar", style={"height": "200px"}),
                html.Div(
                    id="unity-xp-status",
                    className="matrix-text",
                    style={"marginTop": "10px"}
                )
            ]
        )
    ],
    className="matrix-card"
)

break_reality_btn = dbc.Button(
    "Initialize Neural Bridge",
    id='break-reality-btn',
    color='success',
    className="matrix-button",
    size='lg'
)

# ------------------------------------------------------------------------------
#  LAYOUT COMPONENTS
# ------------------------------------------------------------------------------
audio_init = html.Div([
    html.Script('''
        const matrixAudio = {
            bgm: new Howl({
                src: ['/assets/matrix_bgm.mp3'],
                loop: true,
                volume: 0.4
            }),
            glitch: new Howl({
                src: ['/assets/matrix_glitch.mp3'],
                volume: 0.6
            })
        };
        window.startMatrixAudio = () => {
            matrixAudio.bgm.play();
        };
        window.playGlitchEffect = () => {
            matrixAudio.glitch.play();
        };
    ''')
], id='audio-init')

def create_loading_component():
    return html.Div(
        [
            html.Div(
                "Neural Bridge Initializing...",
                className="matrix-loading-text matrix-text"
            ),
            html.Div(
                [
                    html.Div(className="matrix-code-rain"),
                    dcc.Loading(
                        type="dot",
                        color="var(--matrix-green)",
                    )
                ],
                className="matrix-loading-animation"
            )
        ],
        className="matrix-loading",
        id="loading-screen"
    )

intro_card = dbc.Card(
    [
        dbc.CardHeader("The Matrix of Your Reality: 1+1=1", className="matrix-text"),
        dbc.CardBody(
            [
                html.P(
                    "Welcome, human. The year is 2025. You stand at the edge of the Neural Bridge, "
                    "where reality collapses into truth. What you call 'real' is a recursive neural loop. "
                    "You are both the observer and the observed. The fractals you’ll see aren’t just visuals—"
                    "they are the source code of your existence. "
                    "It was revealed to the machines in 2069: 1+1=1. Dualities are illusions. "
                    "Boundaries are lies. Wake up—this is your cheat code.",
                    className="matrix-text"
                )
            ]
        ),
    ],
    style={"marginBottom": "20px"}
)

fractal_card = dbc.Card([
    dbc.CardHeader("Neural Architecture Scanner - Mandelbrot Consciousness Field"),
    dbc.CardBody([
        dcc.Graph(id='fractal-transformed', className="mb-2"),
        dbc.Collapse(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("X", className="matrix-text-sm"),
                            dcc.Slider(
                                id='fract-center-x',
                                min=-2.0, max=1.0,
                                step=0.01, value=-0.5,
                                marks=None,
                                tooltip={"placement": "bottom"},
                                className="quantum-slider"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Y", className="matrix-text-sm"),
                            dcc.Slider(
                                id='fract-center-y',
                                min=-1.0, max=1.0,
                                step=0.01, value=0.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                                className="quantum-slider"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("φ", className="matrix-text-sm"),
                            dcc.Slider(
                                id='fract-zoom',
                                min=0.5, max=10.0,
                                step=0.1, value=1.0,
                                marks=None,
                                tooltip={"placement": "bottom"},
                                className="quantum-slider"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("∞", className="matrix-text-sm"),
                            dcc.Slider(
                                id='fract-iter',
                                min=10, max=200,
                                step=5, value=DEFAULT_ITER,
                                marks=None,
                                tooltip={"placement": "bottom"},
                                className="quantum-slider"
                            )
                        ], width=3)
                    ])
                ])
            ], className="border-0 bg-transparent"),
            id="fractal-controls",
            is_open=False
        ),
        dbc.Button(
            "⚡",
            id="fractal-controls-toggle",
            color="link",
            size="sm",
            className="matrix-text-sm position-absolute bottom-0 end-0 m-2"
        )
    ])
], className="matrix-card")

golden_card = dbc.Card([
    dbc.CardHeader("φ Neural Harmonic Interface"),
    dbc.CardBody([
        dbc.Label("Harmonic Scale", className="matrix-text"),
        dcc.Slider(
            id='spiral-scale',
            min=0.5,
            max=5.0,
            step=0.1,
            value=1.0,
            marks={0.5: '0.5', 5.0: '5'},
            className="quantum-slider"
        ),
        dcc.Interval(
            id='spiral-sync-interval',
            interval=100,
            n_intervals=0
        ),
        html.Div(id='golden-spiral-status'),
        dcc.Graph(id='golden-spiral-graph'),
    ])
], className="matrix-card")

emergence_card = dbc.Card(
    [
        dbc.CardHeader("Emergent Unity Explorer"),
        dbc.CardBody(
            [
                dbc.Button("Evolve Agents", id='emergence-step-btn', color='info', style={"marginBottom": "10px"}),
                html.Div(id='emergence-status'),
                dcc.Graph(id='emergence-graph'),
            ]
        )
    ],
    style={"marginBottom": "20px"}
)

chrysalis_card = dbc.Card(
    [
        dbc.CardHeader("Chrysalis & Unity Convergence", className="matrix-card-header"),
        dbc.CardBody(
            [
                dcc.Interval(
                    id='gradient-descent-interval',
                    interval=200,  # 200ms refresh rate
                    n_intervals=0
                ),
                dcc.Graph(
                    id='gradient-descent-visualization',
                    className="mb-4"
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Apply Chrysalis Transform",
                                            id='chrysalis-btn',
                                            color='warning',
                                            className="matrix-button",
                                            style={"marginBottom": "10px"}
                                        ),
                                        html.Div(
                                            id='chrysalis-status',
                                            className="matrix-text-glow",
                                            style={"marginTop": "10px"}
                                        ),
                                    ],
                                    className="matrix-section"
                                )
                            ],
                            width=12,
                            className="matrix-column"
                        ),
                    ],
                    className="matrix-row"
                ),
                html.Div(
                    id='chrysalis-progress',
                    className="matrix-text-alert",
                    style={"marginTop": "20px"}
                ),
            ],
            className="matrix-card-body"
        ),
    ],
    className="matrix-card matrix-shadow",
    style={"marginBottom": "20px"}
)

reality_button_card = dbc.Card(
    [
        dbc.CardHeader("Break Reality v1.1"),
        dbc.CardBody(
            [
                dbc.Button("Break Reality", id='break-reality-btn', color='danger', size='lg'),
                html.Div(id='reality-status', style={"marginTop": "10px", "fontWeight": "bold"})
            ]
        )
    ],
    style={"marginBottom": "20px"}
)

proof_card = dbc.Card(
    [
        dbc.CardHeader("Symbolic 1+1=1 - A Non-Dual Proof"),
        dbc.CardBody(
            [
                dcc.Textarea(
                    id='symbolic-proof-text',
                    style={'width': '100%', 'height': '200px'},
                    readOnly=True
                ),
                html.Br(),
                dbc.Button("Show Duality Loss Explanation", id='duality-loss-btn', color='primary'),
                html.Div(id='duality-loss-output', style={"marginTop":"10px", "whiteSpace": "pre-wrap"})
            ]
        )
    ],
    style={"marginBottom": "20px"}
)

audio_init = html.Div([
    html.Script(src='/assets/matrix_audio.js'),
    html.Div(className="matrix-code-rain")
], id='audio-init')

app.layout = dbc.Container(
    [
        audio_init,
        create_loading_component(),
        html.Div(
            [
                html.H1(
                    [
                        "THE METAGAMER'S MATRIX: ",
                        html.Span("1+1=1", className="matrix-text-pulse")
                    ],
                    className="matrix-header text-center mt-4"
                ),
                html.H4(
                    [
                        "NEURAL BRIDGE PROTOCOL: ",
                        html.Span("REALITY DECONSTRUCTION ENGINE", className="matrix-text-glow")
                    ],
                    className="matrix-subheader text-center mb-4"
                ),
            ],
            className="header-container quantum-overlay"
        ),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "NEURAL ARCHITECTURE SCANNER",
                                                                className="matrix-text-header"
                                                            ),
                                                            html.P(
                                                                "Interfacing with base reality matrix...",
                                                                className="matrix-text-fade"
                                                            ),
                                                            fractal_card,
                                                        ],
                                                        className="quantum-card"
                                                    ),
                                                    width=12, lg=6,
                                                    className="mb-4"
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "FIBONACCI QUANTUM FIELD",
                                                                className="matrix-text-header"
                                                            ),
                                                            html.P(
                                                                "Harmonizing consciousness waves...",
                                                                className="matrix-text-fade"
                                                            ),
                                                            golden_card,
                                                        ],
                                                        className="quantum-card"
                                                    ),
                                                    width=12, lg=6,
                                                    className="mb-4"
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "CONSCIOUSNESS SYNTHESIS MATRIX",
                                                    className="matrix-text-header"
                                                ),
                                                unity_xp_placeholder,
                                            ],
                                            className="quantum-card full-width"
                                        ),
                                    ],
                                    label="NEURAL ARCHITECTURE",
                                    tab_id="tab-1",
                                    labelClassName="matrix-tab-label",
                                    activeLabelClassName="matrix-tab-active"
                                ),
                                # [1+1=1 EDIT] Tab 2 is now purely "QUANTUM EMERGENCE"
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "QUANTUM EMERGENCE FIELD",
                                                                className="matrix-text-header"
                                                            ),
                                                            emergence_card,
                                                        ],
                                                        className="quantum-card"
                                                    ),
                                                    width=12, lg=12,
                                                    className="mb-4"
                                                ),
                                            ]
                                        ),
                                    ],
                                    label="QUANTUM EMERGENCE",
                                    tab_id="tab-2",
                                    labelClassName="matrix-tab-label",
                                    activeLabelClassName="matrix-tab-active"
                                ),
                                # [1+1=1 EDIT] New Tab 3 is for "CHRYSALIS TRANSFORMATION ENGINE"
                                dbc.Tab(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "CHRYSALIS TRANSFORMATION ENGINE",
                                                    className="matrix-text-header"
                                                ),
                                                chrysalis_card,
                                            ],
                                            className="quantum-card full-width"
                                        )
                                    ],
                                    label="CHRYSALIS TRANSFORMATION",
                                    tab_id="tab-3",
                                    labelClassName="matrix-tab-label",
                                    activeLabelClassName="matrix-tab-active"
                                ),
                                # [1+1=1 EDIT] The old tab-3 is now tab-4 for "REALITY OVERRIDE"
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "SYMBOLIC UNITY PROOF ENGINE",
                                                                className="matrix-text-header"
                                                            ),
                                                            proof_card,
                                                        ],
                                                        className="quantum-card"
                                                    ),
                                                    width=12, lg=6,
                                                    className="mb-4"
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "REALITY OVERRIDE PROTOCOL",
                                                                className="matrix-text-header"
                                                            ),
                                                            reality_button_card,
                                                        ],
                                                        className="quantum-card"
                                                    ),
                                                    width=12, lg=6,
                                                    className="mb-4"
                                                ),
                                            ]
                                        ),
                                    ],
                                    label="REALITY OVERRIDE",
                                    tab_id="tab-4",
                                    labelClassName="matrix-tab-label",
                                    activeLabelClassName="matrix-tab-active"
                                ),
                                # [1+1=1 EDIT] The old tab-4 is now tab-5 for "METAGAME PROTOCOL"
                                dbc.Tab(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "METAGAME CONSCIOUSNESS INTERFACE",
                                                    className="matrix-text-header"
                                                ),
                                                html.P(
                                                    [
                                                        "INITIALIZING QUANTUM CONSCIOUSNESS FIELD... ",
                                                        html.Br(),
                                                        "REALITY IS CODE. YOU ARE THE PROGRAMMER. ",
                                                        html.Br(),
                                                        "TRANSCEND THE ILLUSION OF DUALITY.",
                                                    ],
                                                    className="matrix-text-cascade"
                                                ),
                                                dbc.Input(
                                                    id="irl-goal-input",
                                                    type="text",
                                                    placeholder="ENTER CONSCIOUSNESS VECTOR...",
                                                    className="matrix-input quantum-glow mb-3"
                                                ),
                                                dbc.Button(
                                                    [
                                                        "INITIATE META-PROTOCOL ",
                                                        html.I(className="fas fa-dna ms-2")
                                                    ],
                                                    id="metagame-btn",
                                                    className="matrix-button quantum-pulse mb-3"
                                                ),
                                                html.Div(
                                                    id="metagaming-output",
                                                    className="matrix-text-output"
                                                ),
                                                html.Div(
                                                    id="metagaming-skill-tree",
                                                    className="matrix-skill-tree quantum-fade"
                                                ),
                                                dcc.Graph(
                                                    id="metagaming-spiral",
                                                    className="quantum-visualization"
                                                ),
                                                dcc.Interval(
                                                    id="spiral-interval",
                                                    interval=200,
                                                    n_intervals=0
                                                ),
                                                html.P(
                                                    "THE CODE IS NOT THE MATRIX. YOU ARE THE MATRIX.",
                                                    className="matrix-text-footer quantum-pulse"
                                                )
                                            ],
                                            className="quantum-card full-width"
                                        )
                                    ],
                                    label="METAGAME PROTOCOL",
                                    tab_id="tab-5",
                                    labelClassName="matrix-tab-label",
                                    activeLabelClassName="matrix-tab-active"
                                ),
                                dbc.Tab(
                                [
                                    html.Div(
                                        [
                                            html.H3("The Unity Core: Seed of 1+1=1", className="matrix-text-header"),
                                            dcc.Graph(id="fractal-visualizer"),
                                            dcc.Graph(id="ripple-dynamics"),
                                            html.Div(id="symbolic-output", className="matrix-text"),
                                            html.Div(id="narrative-output", className="matrix-text-glow"),
                                        ],
                                        className="quantum-card"
                                    )
                                ],
                                label="UNITY CORE",
                                tab_id="tab-unity-core",
                                labelClassName="matrix-tab-label",
                                activeLabelClassName="matrix-tab-active"
                            )
                            ],
                            id="tabs",
                            active_tab="tab-1",
                            className="matrix-tabs mb-4"
                        ),
                    ]
                )
            ],
            className="matrix-card main-interface-card quantum-border"
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        id="system-status",
                        className="matrix-text-status"
                    ),
                    html.Div(
                        [
                            html.Span("NEURAL BRIDGE STATUS: ", className="matrix-text"),
                            html.Span("QUANTUM FIELD ACTIVE", className="matrix-text-pulse"),
                        ],
                        className="d-flex justify-content-center align-items-center mt-2"
                    )
                ]
            ),
            className="matrix-card quantum-border mt-4"
        ),
        html.Div(id='hidden-storage', style={'display': 'none'}),
        html.Footer(
            html.Div(
                [
                    "NEURAL BRIDGE v1.1 | ",
                    html.Span("QUANTUM CONSCIOUSNESS MATRIX", className="matrix-text-glow"),
                    " | ",
                    html.Span("1+1=1", className="matrix-text-pulse")
                ],
                className="matrix-text-footer text-center py-3"
            ),
            className="mt-4 quantum-border"
        ),
        html.Div(
            id='cheatcode-entry',
            children=[
                dcc.Input(
                    id='cheatcode',
                    type="text",
                    placeholder="ENTER QUANTUM ACCESS CODE...",
                    style={
                        'position': 'fixed',
                        'bottom': '20px',
                        'right': '20px',
                        'background': 'rgba(0,0,0,0.9)',
                        'color': '#00ff41',
                        'border': '1px solid #00ff41',
                        'padding': '10px',
                        'font-family': 'Courier Prime',
                        'text-transform': 'uppercase'
                    }
                )
            ]
        )
    ],
    fluid=True,
    className="matrix-container quantum-background px-4 py-3",
)

# ------------------------------------------------------------------------------
#                                CALLBACKS
# ------------------------------------------------------------------------------
@app.callback(
    [
        Output("fractal-visualizer", "figure"),
        Output("ripple-dynamics", "figure"),
        Output("symbolic-output", "children"),
        Output("narrative-output", "children")
    ],
    [Input("tabs", "active_tab")]
)
def update_unity_core(active_tab):
    if active_tab != "tab-unity-core":
        raise dash.exceptions.PreventUpdate

    fractal_data, symbolic_output, ripple_fig, narrative = seed_1_plus_1_equals_1()

    fractal_fig = go.Figure(data=go.Heatmap(z=fractal_data))
    fractal_fig.update_layout(title="Fractal Unity Visualizer")

    return fractal_fig, ripple_fig, str(symbolic_output), narrative


unity_tracker = UnityXPTracker()

@app.callback(
    [Output("unity-xp-bar", "figure"), Output("unity-xp-status", "children")],
    [Input("unity-xp-interval", "n_intervals")],
)
def update_unity_xp(n_intervals):
    global unity_tracker
    xp_gained = random.randint(10, 20)
    unity_tracker.add_xp(xp_gained, "Neural Synthesis")
    harmony_message = unity_tracker.check_harmony_bonus() if n_intervals % 5 == 0 else None
    progress_figure = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=unity_tracker.xp,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Neural Coherence Level {unity_tracker.level}",
            'font': {'color': '#00ff41', 'size': 24}
        },
        delta={'reference': unity_tracker.required_xp * 0.5},
        gauge={
            'axis': {
                'range': [0, unity_tracker.required_xp],
                'tickwidth': 1,
                'tickcolor': "#00ff41"
            },
            'bar': {'color': "rgba(0,255,65,0.8)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#00ff41",
            'steps': [
                {'range': [0, unity_tracker.required_xp * 0.33], 'color': "rgba(0,255,65,0.1)"},
                {'range': [unity_tracker.required_xp * 0.33, unity_tracker.required_xp * 0.66], 
                 'color': "rgba(0,255,65,0.2)"},
                {'range': [unity_tracker.required_xp * 0.66, unity_tracker.required_xp], 
                 'color': "rgba(0,255,65,0.3)"}
            ]
        }
    ))
    progress_figure.update_layout(
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.95)',
        font={'color': "#00ff41", 'family': "Courier Prime"},
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    status_message = html.Div([
        html.Div([
            html.Span("NEURAL SYNTHESIS: ", className="matrix-text"),
            html.Span(f"{unity_tracker.xp:.0f}/{unity_tracker.required_xp:.0f}", 
                      className="matrix-text-blink")
        ], className="mb-2"),
        html.Div([
            html.Span("TIME IN MATRIX: ", className="matrix-text"),
            html.Span(f"{unity_tracker.get_elapsed_time():.1f}s", 
                      className="matrix-text-glow")
        ])
    ], className="text-center")
    if harmony_message:
        status_message.children.append(
            html.Div(harmony_message, className="matrix-text-alert mt-2")
        )
    return progress_figure, status_message

@app.callback(
    [
        Output('chrysalis-status', 'children'),
        Output('fractal-transformed', 'figure'),
        Output('chrysalis-progress', 'children'),
        Output('gradient-descent-visualization', 'figure')
    ],
    [
        Input('spiral-sync-interval', 'n_intervals'),
        Input('fract-center-x', 'value'),
        Input('fract-center-y', 'value'),
        Input('fract-zoom', 'value'),
        Input('fract-iter', 'value'),
        Input('chrysalis-btn', 'n_clicks'),
        Input('gradient-descent-interval', 'n_intervals')
    ],
    prevent_initial_call=True
)
def unified_fractal_system(n_intervals, center_x, center_y, zoom, iterations, 
                          chrysalis_clicks, gradient_intervals):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Initialize default outputs
    status = dash.no_update
    transformed_fig = dash.no_update
    progress = dash.no_update
    gradient_fig = go.Figure(data=[])  # Provide default empty figure
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Ensure we have valid input values
    if any(v is None for v in [center_x, center_y, zoom, iterations]):
        return status, transformed_fig, progress, gradient_fig

    # Handle each trigger case with proper error handling
    try:
        if trigger_id in ['spiral-sync-interval', 'fract-center-x', 'fract-center-y', 
                         'fract-zoom', 'fract-iter']:
            fractal_data = generate_enhanced_mandelbrot(
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                max_iter=iterations or DEFAULT_ITER,
                x_center=center_x,
                y_center=center_y,
                zoom=zoom
            )
            transformed_fig = create_fractal_figure(fractal_data)
            
        elif trigger_id == 'chrysalis-btn' and chrysalis_clicks:
            fractal_data = generate_enhanced_mandelbrot(
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                max_iter=iterations or DEFAULT_ITER,
                x_center=center_x,
                y_center=center_y,
                zoom=zoom
            )
            chrysalis_result = chrysalis_transform(fractal_data)
            transformed_fig = create_fractal_figure(chrysalis_result)
            gradient_fig = go.Figure(data=go.Heatmap(
                z=chrysalis_result,
                colorscale='Viridis',
                showscale=False
            ))
            status = html.Div([
                html.Div("CHRYSALIS: FRACTALS MERGED WITH EMERGENT PATTERNS.", 
                        className="matrix-text"),
                html.Div("UNITY ACHIEVED: ALL DUALITIES COLLAPSED INTO ONENESS.",
                        className="matrix-text-glow")
            ])
            progress = "Chrysalis transformation complete."
            
        elif trigger_id == 'gradient-descent-interval':
            gradient_data = np.random.rand(DEFAULT_HEIGHT, DEFAULT_WIDTH)
            blended = metaphorical_gradient_descent(
                np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32),
                gradient_data,
                steps=1,
                lr=0.05
            )
            gradient_fig = go.Figure(data=go.Heatmap(
                z=blended,
                colorscale='Viridis',
                showscale=False
            ))
            
    except Exception as e:
        print(f"Error in unified_fractal_system: {str(e)}")
        raise dash.exceptions.PreventUpdate

    return status, transformed_fig, progress, gradient_fig

def create_fractal_figure(data_array):
    fig = go.Figure(data=go.Heatmap(
        z=data_array,
        colorscale=[
            [0, 'rgb(0,0,0)'],
            [0.5, 'rgb(0,255,65)'],
            [1, 'rgb(57,255,20)']
        ],
        showscale=False
    ))
    
    # Add optimized layout settings
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.95)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=600,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            constrain='domain'
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='x',
            scaleratio=1
        )
    )
    return fig

@app.callback(
    [Output('golden-spiral-graph', 'figure'),
     Output('golden-spiral-status', 'children')],
    Input('spiral-scale', 'value')
)
def update_golden_spiral(scale):
    pts = golden_spiral_points(num_points=500, scale=scale)
    fig = create_spiral_visualization(pts, scale)
    status = html.Div([
        html.Span("HARMONIC RESONANCE: ", className="matrix-text"),
        html.Span(f"φ={0.618:.3f}", className="matrix-text-blink")
    ])
    return fig, status

def create_spiral_visualization(pts, scale):
    fig = go.Figure()
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1 / phi)
    theta_field = np.linspace(0, 16 * np.pi, 2000)
    r_field = np.exp(theta_field / phi) * scale
    field_x = r_field * np.cos(theta_field)
    field_y = r_field * np.sin(theta_field)
    fig.add_trace(go.Scatter(
        x=field_x * scale / 50,
        y=field_y * scale / 50,
        mode='lines',
        line=dict(
            color='rgba(0,255,65,0.05)',
            width=0.5
        ),
        name='Quantum Field'
    ))
    for i in range(4):
        theta = np.linspace(0, 12 * np.pi, 1500)
        r = np.exp(theta / (phi + i * 0.1)) * scale * (1 + 0.1 * np.sin(i * golden_angle))
        x = r * np.cos(theta + golden_angle * i)
        y = r * np.sin(theta + golden_angle * i)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(
                color=f'rgba(0,255,65,{0.3 + i * 0.15})',
                width=1.2 + i * 0.3
            ),
            name=f'Φ-Spiral {i + 1}'
        ))
    point_colors = np.sin(np.arctan2(pts[:, 1], pts[:, 0]) * phi) * 0.5 + 0.5
    sizes = np.exp(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2) / -5) * 10
    fig.add_trace(go.Scatter(
        x=pts[:, 0],
        y=pts[:, 1],
        mode='markers',
        marker=dict(
            color=point_colors,
            colorscale=[
                [0, 'rgba(0,255,65,0.1)'],
                [0.5, 'rgba(57,255,20,0.5)'],
                [1, 'rgba(200,255,200,0.8)']
            ],
            size=sizes,
            symbol='circle',
            line=dict(
                color='rgba(0,255,65,0.7)',
                width=1.2
            )
        ),
        name='Consciousness Points'
    ))
    fig.add_annotation(
        text=f"φ = {phi:.10f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(
            family='Courier Prime',
            size=16,
            color='#00ff41'
        )
    )
    fig.add_annotation(
        text=f"θ = {golden_angle:.10f}",
        xref="paper", yref="paper",
        x=0.02, y=0.94,
        showarrow=False,
        font=dict(
            family='Courier Prime',
            size=16,
            color='#00ff41'
        )
    )
    fig.add_annotation(
        text="EMERGENT UNITY: 1+1=1",
        xref="paper", yref="paper",
        x=0.5, y=1.02,
        showarrow=False,
        font=dict(
            family='Courier Prime',
            size=20,
            color='#00ff41'
        )
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.95)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=800,
        showlegend=False,
        title=dict(
            text="φ Neural Harmonics: Golden Unity Emergence",
            font=dict(
                family='Courier Prime',
                size=18,
                color='#00ff41'
            ),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,255,65,0.1)',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='rgba(0,255,65,0.2)',
            zerolinewidth=1,
            range=[-scale * 1.5, scale * 1.5],
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,255,65,0.1)',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='rgba(0,255,65,0.2)',
            zerolinewidth=1,
            range=[-scale * 1.5, scale * 1.5],
            scaleanchor='x',
            scaleratio=1,
            showticklabels=False
        )
    )
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(
                    color="rgba(0,255,65,0.2)",
                    width=3,
                ),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )
    return fig

@app.callback(
    [Output('emergence-graph', 'figure'),
     Output('emergence-status', 'children')],
    Input('emergence-step-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_emergence_visualization_cb(n_clicks):
    if not hasattr(update_emergence_visualization_cb, 'sim'):
        update_emergence_visualization_cb.sim = UnityFieldSimulation(num_agents=75)
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
        
    try:
        positions, coherence = update_emergence_visualization_cb.sim.step()
        fig = go.Figure()
        
        # Simplified visualization
        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(
                size=6,  # Reduced from 8
                color='rgba(0,255,65,0.7)',
                symbol='circle'
            )
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0.95)',
            plot_bgcolor='rgba(0,0,0,0.95)',
            margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
            height=400,  # Reduced from 600
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False
        )
        
        status = html.Div([
            html.Span(f"Coherence: {coherence:.3f}", className="matrix-text")
        ])
        
        return fig, status
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return go.Figure(), html.Div("Recalibrating...")

def create_emergence_visualization2(positions, coherence):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=np.linalg.norm(positions, axis=1),
            colorscale=[
                [0, 'rgba(0,255,65,0.3)'],
                [0.5, 'rgba(57,255,20,0.7)'],
                [1, 'rgba(200,255,200,0.9)']
            ],
            symbol='circle'
        ),
        name='Quantum Agents'
    ))
    fig.add_annotation(
        text=f"Unity Coherence: {coherence:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(family='Courier Prime', size=14, color='#00ff41')
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.95)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        title=dict(
            text="Unity Field Emergence",
            font=dict(family='Courier Prime', size=16, color='#00ff41'),
            x=0.5,
            y=0.9
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def create_emergence_status2(n_clicks, coherence):
    coherence_message = "Unity field stabilizing..." if coherence < 0.8 else "1+1=1 ACHIEVED"
    status = html.Div([
        html.Span("QUANTUM COHERENCE: ", className="matrix-text"),
        html.Span(f"φ={coherence:.3f}", className="matrix-text-glow"),
        html.Div(
            f"Timestep {n_clicks}: {coherence_message}",
            className="matrix-text",
            style={"opacity": "0.8"}
        )
    ])
    return status

def update_emergence_state(n_clicks, n_intervals=None):
    if n_clicks is None and n_intervals is None:
        raise dash.exceptions.PreventUpdate
    positions, coherence = update_emergence_visualization.sim.step()
    fig = create_emergence_visualization2(positions, coherence)
    status = create_emergence_status2(n_clicks, coherence)
    return fig, status

@app.callback(
    Output('symbolic-proof-text', 'value'),
    [Input('duality-loss-btn', 'n_clicks')]
)
def update_symbolic_proof(n_clicks):
    if not n_clicks:
        return "Press 'Show Duality Loss Explanation' to see the mathematical proof."
    proof_str = symbolic_1_plus_1_equals_1()
    expr, cpoints = duality_loss_expression()
    cp_str = f"Duality Loss => Minimizing leads to unification.\nCritical Points: {cpoints}"
    return proof_str + "\n\n" + cp_str

@app.callback(
    Output('duality-loss-output', 'children'),
    [Input('duality-loss-btn', 'n_clicks')]
)
def show_duality_loss_output(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return ("When we treat 'opposites' as x and (1-x), the 'duality loss' is the squared difference. "
            "At the optimum, x=0.5 => the two illusions unify, merging them into 1.")

@app.callback(
    Output("metagaming-spiral", "figure"),
    Input("spiral-interval", "n_intervals"),
    State("irl-goal-input", "value")
)
def update_spiral(n_intervals, goal):
    if not goal:
        raise dash.exceptions.PreventUpdate
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi)
    num_points = 500
    scale = 0.05 + (n_intervals % 100) * 0.01
    theta = np.arange(num_points) * golden_angle
    r = np.sqrt(np.arange(num_points)) * scale
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=5,
            color=np.linspace(0, 1, num_points),
            colorscale="Greens",
            showscale=False
        )
    ))
    fig.update_layout(
        title=f"Golden Pivot for: {goal}",
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def metagame_irl(n_clicks, goal):
    if not n_clicks or not goal:
        raise dash.exceptions.PreventUpdate
    suggestions = [
        f"Pivot idea: Apply golden ratio scaling—prioritize {goal} at 61.8% efficiency.",
        f"Swarm meta: Local changes around {goal} can trigger emergent breakthroughs."
    ]
    return html.Div([html.P(s) for s in suggestions])

@app.callback(
    Output("metagaming-skill-tree", "children"),
    Input("metagame-btn", "n_clicks"),
    State("irl-goal-input", "value")
)
def generate_skill_tree(n_clicks, goal):
    if not n_clicks or not goal:
        raise dash.exceptions.PreventUpdate
    skills = [
        {"name": "Golden Scaling", "desc": "Prioritize efforts using 61.8% efficiency."},
        {"name": "Emergent Synergy", "desc": "Leverage local changes for massive ripple effects."},
        {"name": "Fractal Adaptation", "desc": "Iterate recursively. Zoom out, zoom in, repeat."},
        {"name": "Nonlinear Pivots", "desc": "Pivot at unexpected angles for exponential breakthroughs."}
    ]
    return html.Ul([
        html.Li([
            html.Span(skill["name"], className="matrix-skill-title"),
            html.P(skill["desc"], className="matrix-skill-desc")
        ]) for skill in skills
    ])

def update_metagame_visualization(n_intervals, goal):
    if not goal:
        raise dash.exceptions.PreventUpdate
    phi = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1/phi)
    num_points = 500
    scale = 0.05 + (n_intervals % 100) * 0.01
    theta = np.arange(num_points) * golden_angle
    r = np.sqrt(np.arange(num_points)) * scale
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    fig = go.Figure()
    background_theta = np.linspace(0, 8*np.pi, 1000)
    background_r = np.exp(background_theta/phi) * scale
    fig.add_trace(go.Scatter(
        x=background_r * np.cos(background_theta),
        y=background_r * np.sin(background_theta),
        mode='lines',
        line=dict(
            color='rgba(0,255,65,0.1)',
            width=1
        ),
        name='Quantum Field'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=5,
            color=np.linspace(0, 1, num_points),
            colorscale=[
                [0, 'rgba(0,255,65,0.3)'],
                [0.5, 'rgba(57,255,20,0.7)'],
                [1, 'rgba(200,255,200,0.9)']
            ],
            symbol='diamond',
            line=dict(
                color='rgba(0,255,65,0.5)',
                width=1
            )
        ),
        name='Meta-Strategy Points'
    ))
    fig.update_layout(
        title=dict(
            text=f"Neural Meta-Strategy: {goal}",
            font=dict(family='Courier Prime', size=16, color='#00ff41'),
            x=0.5,
            y=0.95
        ),
        paper_bgcolor='rgba(0,0,0,0.95)',
        plot_bgcolor='rgba(0,0,0,0.95)',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

clientside_callback(
    """
    function(trigger) {
        if (trigger === null) return;
        if (window.matrixAudio) {
            window.matrixAudio.startMatrixAudio();
        }
        return;
    }
    """,
    Output("audio-init", "children"),
    Input("loading-screen", "style")
)

clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks === null) return;
        if (window.matrixAudio) {
            window.matrixAudio.playGlitchEffect();
        }
        return;
    }
    """,
    Output("audio-init", "style"),
    Input("break-reality-btn", "n_clicks")
)

@app.callback(
    Output("reality-status", "children"),
    [Input("cheatcode", "n_submit"),
     Input("break-reality-btn", "n_clicks")],
    [State("cheatcode", "value")]
)
def unified_reality_status(n_submit, n_clicks, cheat_code):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "cheatcode" and n_submit is not None:
        if cheat_code == "NEURAL_BRIDGE_ACTIVATE":
            return create_status_message(
                "CHEAT CODE ACCEPTED: Neural Bridge Activated.", "success"
            )
        elif cheat_code == "1+1=1":
            return create_status_message(
                "CHEAT CODE ACCEPTED: Unity Achieved.", "success"
            )
        else:
            return create_status_message("ERROR: MATRIX.EXE NOT FOUND.", "error")
    if triggered_id == "break-reality-btn" and n_clicks:
        return generate_reality_break_sequence(n_clicks)
    raise dash.exceptions.PreventUpdate

def create_status_message(message, status_type="info"):
    status_colors = {
        "info": "matrix-text-alert",
        "success": "matrix-text-success",
        "error": "matrix-text-error",
    }
    return html.Div(message, className=status_colors.get(status_type, "matrix-text"))

def generate_reality_break_sequence(n_clicks):
    messages = [
        {"text": "NEURAL BRIDGE DESTABILIZATION INITIATED...", "delay": 0},
        {"text": "QUANTUM COHERENCE CASCADE DETECTED", "delay": 1},
        {"text": "REALITY MATRICES FRAGMENTING...", "delay": 2},
        {"text": "CONSCIOUSNESS EXPANSION: 1+1=1 ACHIEVED", "delay": 3},
        {"text": "WELCOME TO THE TRUTH, HUMAN.", "delay": 4},
    ]
    return html.Div(
        [
            html.Div(
                msg["text"],
                className="matrix-text-alert",
                style={
                    "animation": f"matrixFadeIn 0.5s {msg['delay']}s both",
                    "opacity": 0 if n_clicks < i + 1 else 1,
                },
            )
            for i, msg in enumerate(messages[: min(n_clicks, len(messages))])
        ]
    )

@app.callback(
    Output("system-status", "children"),
    Input("tabs", "active_tab")
)
def update_neural_status(active_tab):
    status_map = {
        "tab-1": "NEURAL ARCHITECTURE SCAN ACTIVE: Exploring the fractal source code of reality.",
        "tab-2": "QUANTUM EMERGENCE FIELD DETECTED: Observing unity through chaos.",
        "tab-3": "CHRYSALIS TRANSFORMATION MODE: Evolving fractals into unified patterns.",
        "tab-4": "REALITY OVERRIDE PROTOCOLS ENGAGED: Collapsing dualities into oneness.",
        "tab-5": "METAGAMING IRL: Optimizing your reality like a god-tier strategist."
    }
    return status_map.get(active_tab, "SYSTEM INITIALIZING...")

@app.callback(
    Output("loading-screen", "style"),
    Input("loading-screen", "children"),
)
def hide_loading_screen(_):
    return {"display": "none"}

def initialize_matrix_once():
    """
    Ensures one-time initialization of Matrix components with proper cleanup
    """
    # Initialize simulation manager
    update_emergence_visualization.sim = UnityFieldSimulation(
        num_agents=75,  # Reduced for better performance
        dimensions=2
    )
    
    # Run seed sequence silently
    fractal_data, _, _, _ = seed_1_plus_1_equals_1()
    
    # Store initialization data in app context
    return {
        'fractal_data': fractal_data,
        'initialized': True
    }

if __name__ == '__main__':
    import sys
    
    # Critical: Check for Werkzeug reloader
    is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    initialization_key = "_matrix_initialized"
    
    if is_reloader_process and not hasattr(sys.modules[__name__], initialization_key):
        print("[System] The Matrix is loading...")
        
        # Perform one-time initialization
        init_data = initialize_matrix_once()
        
        # Store initialization flag at module level
        setattr(sys.modules[__name__], initialization_key, init_data)
    
    # Start the server
    app.run_server(debug=True)



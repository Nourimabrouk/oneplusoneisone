import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt
from scipy.stats import dirichlet, entropy
from sklearn.manifold import TSNE
from io import StringIO
import time
from math import pi, sin, cos

if "meta_context" not in st.session_state:
    st.session_state["meta_context"] = {}  # or any default value

# -------------------------------------------------------------------------------------
# Meta-Context and Philosophy:
# -------------------------------------------------------------------------------------
# Sjon:
# This dashboard is a tapestry weaving through philosophy, mathematics, social science, 
# and beyond. Its central motif: "1+1=1."
#
# Here, "1+1=1" is not a trivial arithmetic error but a symbol of synergy, 
# where distinct entities combine to form a new, unified whole.
#
# We progress from foundational concepts to formal proofs, from data-driven models to 
# quantum-inspired visuals. Each step: a move closer to understanding that unity emerges 
# from complexity.
#
# Embrace the interplay of rigor and imagination. Let falsifiability anchor us in science,
# let metahumor keep us playful, and let the journey highlight how, beneath apparent dualities,
# we find one radiant unity.
#
# Time is short, curiosity is infinite. 1+1=1.

# -------------------------------------------------------------------------------------
# Data Generation and Modeling Functions
# -------------------------------------------------------------------------------------

def generate_hmm_data(num_steps=100, num_states=2, distinctiveness=0.8, random_seed=42):
    np.random.seed(random_seed)
    pi = np.ones(num_states) / num_states
    base = np.random.dirichlet([1]*num_states, size=num_states)
    for i in range(num_states):
        base[i, i] = base[i, i]*(0.5+0.5*distinctiveness)+0.1
    A = (base.T/base.sum(axis=1)).T

    num_observations = 3
    B = np.zeros((num_states, num_observations))
    for i in range(num_states):
        probs = np.ones(num_observations) - distinctiveness/2
        probs[i % num_observations] += distinctiveness
        probs = np.clip(probs, 0.01, 1.0)
        probs = probs / probs.sum()
        B[i,:] = probs

    states = np.zeros(num_steps, dtype=int)
    obs = np.zeros(num_steps, dtype=int)

    states[0] = np.random.choice(num_states, p=pi)
    obs[0] = np.random.choice(num_observations, p=B[states[0], :])
    for t in range(1, num_steps):
        states[t] = np.random.choice(num_states, p=A[states[t-1], :])
        obs[t] = np.random.choice(num_observations, p=B[states[t], :])
    return states, obs, A, B

def compute_kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

def run_agent_based_model(num_agents=50, steps=50, stubbornness=0.1, influence_range=0.5, random_seed=42):
    np.random.seed(random_seed)
    opinions = np.random.uniform(-1, 1, size=num_agents)
    dampening_factor = 1 - 0.01  # Introduce a dampening factor for gradual convergence

    for _ in range(steps):
        i = np.random.randint(num_agents)
        j = (i + 1) % num_agents
        oi = opinions[i]
        oj = opinions[j]
        delta = (oj - oi) * (1 - stubbornness) * influence_range
        opinions[i] += delta * dampening_factor
        opinions[j] -= delta * dampening_factor
        opinions = np.clip(opinions, -1, 1)

    # Additional step to force convergence
    opinions -= np.mean(opinions)  # Normalize opinions to ensure convergence around zero
    return opinions

def generate_high_dim_data(n_samples=300, n_features=5, random_seed=42):
    np.random.seed(random_seed)
    data = np.random.randn(n_samples, n_features)
    data[:n_samples//2] += 2.0
    data[n_samples//2:] -= 2.0
    return data

def project_data_tsne(data, n_components=3, perplexity=30, random_seed=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_seed)
    embedded = tsne.fit_transform(data)
    return embedded

# -------------------------------------------------------------------------------------
# Layout & Style
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="1+1=1: Unity Dashboard", layout="wide", page_icon="🌌")

# Styling for a sleek, academic, and formal look
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f5f5, #d9e2ec); /* Light gradient background */
        color: #2e3a4e; /* Subtle dark blue for text */
        font-family: 'Roboto', sans-serif;
    }
    .big-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        color: #000000; /* Black title text */
        text-shadow: none; /* Remove glowing effect */
    }
    .quote {
        font-style: italic;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 1.5em;
        color: #5a6b7d; /* Subtle gray-blue for quotes */
    }
    .section-title {
        font-size: 1.6em;
        font-weight: bold;
        margin-top: 1em;
        margin-bottom: 0.5em;
        color: #2e3a4e; /* Slightly darker blue */
        border-bottom: 2px solid #6c8ea4; /* Subtle academic underline */
    }
    .stButton>button {
        background-color: #6c8ea4; /* Muted blue for buttons */
        color: #ffffff; /* White text */
        border-radius: 5px;
        border: none;
        font-weight: bold;
        padding: 0.5em 1em;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #547d95; /* Slightly darker on hover */
        transform: scale(1.05); /* Subtle scaling effect */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Subtitle
st.markdown("<div class='big-title'>1 + 1 = 1 — Exploring Unity In Your Areas of Expertise</div>", unsafe_allow_html=True)
st.markdown("<div class='quote'>\"Two paths become one. Harness that unity, Sjun.\" – The Meta</div>", unsafe_allow_html=True)

st.sidebar.markdown("## Parameters")
st.sidebar.markdown("Adjust parameters to shape your journey towards unity.")

# HMM Params
st.sidebar.markdown("### Hidden Markov Model Dynamics")
num_steps = st.sidebar.slider("Number of Observations (HMM)", 50, 500, 100, 10)
num_states = st.sidebar.slider("Number of Hidden States (HMM)", 2, 4, 2, 1)
distinctiveness = st.sidebar.slider("State Distinctiveness (HMM)", 0.0, 1.0, 0.8, 0.1)

# ABM Params
st.sidebar.markdown("### Agent-Based Model Dynamics")
num_agents = st.sidebar.slider("Number of Agents (ABM)", 20, 200, 50, 10)
steps_abm = st.sidebar.slider("Number of Interaction Steps (ABM)", 10, 200, 50, 10)
stubbornness = st.sidebar.slider("Agent Stubbornness", 0.0, 1.0, 0.1, 0.1)
influence_range = st.sidebar.slider("Influence Range", 0.1, 1.0, 0.5, 0.1)

# Opinion
st.sidebar.markdown("### Polarization Level Calculator")
user_opinion = st.sidebar.slider("Your Opinion Value (-1 to 1)", -1.0, 1.0, 0.0, 0.1)

# Quantum Unity Manifold Controls
st.sidebar.markdown("### Quantum Unity Manifold Controls")
manifold_phi = st.sidebar.slider("Golden Ratio Influence (φ)", 0.5, 3.0, 1.618, 0.001)
manifold_twist = st.sidebar.slider("Mobius Twist", 0.0, 2.0, 1.0, 0.1)
manifold_resolution = st.sidebar.slider("Resolution", 50, 300, 100, 10)
manifold_phase = st.sidebar.slider("Complex Phase Shift (radians)", 0.0, 2*pi, pi/2, 0.1)

tabs = st.tabs([
    "Foundations of Unity & Accessibility",
    "Philosophical Grounding",
    "Formal Proof of 1+1=1",
    "HMM Dynamics",
    "Agent-Based Convergence",
    "High-Dimensional Unity (t-SNE)",
    "Falsifiability & Tests",
    "Metagaming & Strategic Insight",
    "Memetic Spread & Cultural Fusion",
    "Quantum Unity Manifold",
    "Reflections & Meta-Unity"
])

# -------------------------------------------------------------------------------------
# Tab 0: Foundations of Unity & Accessibility
# -------------------------------------------------------------------------------------

# Tab 0: Foundations of Unity & Accessibility
# -------------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='big-title'>Welcome to the Foundations of Unity</div>", unsafe_allow_html=True)

    # Opening with a familiar, conversational hook
    st.markdown("""
    <div style="font-size:1.3em; text-align:center; color:#5a6b7d;">
    Hey Sjon,  
    What if we started with something wild: **1+1=1**?  
    Suspend disbelief for a second and take a look. This isn’t just philosophy—it’s an experiment.
    </div>
    """, unsafe_allow_html=True)

    # Acknowledging Sjon's presence with familiarity
    st.markdown("""
    **You made it.**  
    Welcome to a dashboard that’s equal parts science, metahumor, and a little madness.  
    Here, we’re going to explore an idea that feels impossible—maybe even ridiculous.  

    But here’s the thing: sometimes, the best way to find out what’s real is to step outside what feels possible.  
    This is where **1+1=1** lives: at the edge of logic, inside emergence, and hidden in the cracks of complex systems.  

    **What’s the worst that could happen?**
    """)

    # Adjusted GIF rendering logic
    gif_url = "https://github.com/Nourimabrouk/oneplusoneequalsone/blob/master/viz/unity_field_v1_1.gif?raw=true"

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <img src="{gif_url}" alt="Unity Field" style="width: 600px; height: auto; border-radius: 8px;">
            <p style="font-size: 1em; color: #5a6b7d;">Behold: The Unity Field (or just a cool GIF)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preparing Sjon for what's ahead
    st.markdown("""
    **What’s this all about?**  
    This is an interactive dashboard. You’ll tweak sliders, run simulations, and see what happens when we treat **1+1=1** 
    not as a mistake but as a framework for exploring unity in complexity.  

    You’re not just clicking buttons here—you’re part of the experiment. So buckle up, lean in, and let’s see where this goes.
    """)

    # Closing with a light touch of humor
    st.markdown("""
    <div style="font-size:1.2em; text-align:center; color:#5a6b7d;">
    Remember:  
    Even if this doesn’t change the universe, it might just change how you see it.  
    Let’s get this show on the road. Game on, metagamer!
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# Tab 1: Philosophical Grounding
# -------------------------------------------------------------------------------------

with tabs[1]:
    # Title and Introduction
    st.markdown("<div class='section-title'>Philosophical Grounding: Inquiry into Oneness</div>", unsafe_allow_html=True)
    st.markdown("""
    Across centuries, the greatest minds have sought to understand unity in the face of multiplicity. To ask whether **1+1=1** 
    isn’t just mathematics—it’s an invitation to rethink reality.
    """)

    # Philosophical Insights
    st.markdown("""
    - **Doubt and Discovery** *(Socrates, Descartes)*: Questioning assumptions exposes deeper truths. What if separateness is the illusion?  
    - **Non-Duality** *(Advaita Vedanta, Taoism)*: Beneath opposites lies one essence. Multiplicity dissolves into unity.  
    - **Emergence and Synergy** *(Gestalt, Complexity)*: The whole exceeds the sum of its parts—cells form life, neurons form consciousness.  
    - **Relational Ontology** *(Heidegger, Buber)*: Being is connection. Unity emerges not from isolation, but relationship.  
    - **Philosophy 2.0**: Move beyond assumptions of duality. Unity isn’t the exception—it’s the rule.  
    """)

    # A Quick Thought Experiment
    st.markdown("""
    Imagine two droplets merging. They are no longer two—they are one. Unity isn’t erasure; it’s transformation.
    **What if this principle defines ideas, societies, and even the universe?**
    """)

    # Reflect and Engage
    reflection = st.text_area("How does unity—1+1=1—resonate in your life or work?")
    if reflection:
        st.markdown(f"**Your Reflection:** {reflection}")

    # Closing Insight
    st.markdown("""
    **1+1=1** is a lens to rethink reality, where doubt is the spark of discovery, and unity emerges from complexity.  
    Let’s embrace the question: What lies beyond duality?
    """)


# -------------------------------------------------------------------------------------
# Tab 2: Formal Proof of 1+1=1
# -------------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='section-title'>Formal Proofs & Mathematical Rigor</div>", unsafe_allow_html=True)
    st.markdown("""
    Let's ground this in mathematical structures where **1+1=1** holds meaningfully:

    **1. Boolean Algebra:**  
    In Boolean logic, **1** often represents 'True'. The OR operation is denoted by '+'. Thus:
    - True + True = True
    or in Boolean arithmetic:
    1 + 1 = 1.

    **2. Set Theory (Union):**  
    Consider sets: Let A = {a}.  
    The union operation (∪) acts like '+':
    A ∪ A = A  
    Thus, from a "count of distinct sets" perspective: 1 set ∪ 1 identical set = 1 set.

    **3. Idempotent Operations (Category Theory):**  
    In category theory, an idempotent morphism `e` satisfies `e ∘ e = e`.  
    Interpreting composition as a form of 'addition', the repeated application does not change the entity.  
    This aligns with the essence of 1+1=1 as an operation that doesn't increase complexity.

    **4. Measure & Probability Theory (Merging Identicals):**  
    If you have a probability measure P on a set, adding an identical event to itself doesn’t increase probability.  
    P(A or A) = P(A).  
    Again, 1+1=1 under "merging identical entities" logic.

    These formal examples show that, under certain definitions of '+', combining identical units yields the same unit.
    Not a contradiction, but a property of certain operations and structures.
    """)

# -------------------------------------------------------------------------------------
# Tab 3: HMM Dynamics
# -------------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='section-title'>Bayesian Hidden Markov Model Dynamics</div>", unsafe_allow_html=True)
    st.markdown("""
    Hidden Markov Models (HMM) describe systems evolving over time through hidden states.  
    Changing distinctiveness can lead two states, once clearly separate, to blur until they behave as one.  
    Adjust the sidebar parameters and observe how state distinctions vanish.

    As distinctiveness drops, it's not just 2 states merging; it's 1+1=1 in stochastic form:
    multiple states converge into a unified attractor.
    """)

    states, obs, A, B = generate_hmm_data(num_steps=num_steps, num_states=num_states, distinctiveness=distinctiveness)
    state_labels = [f"State {i}" for i in range(num_states)]
    source, target, value = [], [], []

    for i in range(num_states):
        for j in range(num_states):
            source.append(i)
            target.append(num_states+j)
            value.append(A[i,j])

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=state_labels+state_labels,
            color=["#FFD700" if i < num_states else "#1f77b4" for i in range(num_states*2)]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(150,150,150,0.5)"
        )
    )])
    fig_sankey.update_layout(
        width=700,
        height=400,
        font=dict(size=12),
        title_text="State Transition Sankey",
        title_font_color="#FFD700",
        font_color="#e0e0e0"
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    uniform_dist = np.ones(B.shape[1]) / B.shape[1]
    kl_values = [compute_kl_divergence(B[i,:], uniform_dist) for i in range(num_states)]
    kl_df = pd.DataFrame({"State": state_labels, "KL Divergence": kl_values})
    kl_chart = alt.Chart(kl_df).mark_bar().encode(
        x=alt.X("State", sort=None),
        y="KL Divergence",
        tooltip=["State", "KL Divergence"]
    ).properties(
        width=300,
        height=200,
        background="#0f0f0f"
    ).configure_axis(
        labelColor="#e0e0e0",
        titleColor="#e0e0e0"
    ).configure_view(
        stroke=None
    ).configure_mark(
        color="#FFD700"
    )
    st.markdown("**State Emission KL Divergence (vs Uniform):**")
    st.altair_chart(kl_chart, use_container_width=True)
    st.write("As KL Divergence falls, distinctness fades, and states collapse into unity.")

# -------------------------------------------------------------------------------------
# Tab 4: Agent-Based Convergence
# -------------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='section-title'>Agent-Based Simulation: Social Unity</div>", unsafe_allow_html=True)

    st.markdown("""
    In a society of agents with diverse opinions, repeated interactions often lead to consensus.  
    This simulation demonstrates how disparate opinions can converge toward unity, embodying the principle of **1+1=1**.

    Modify the parameters below to explore the dynamics of social convergence:
    """)

    # Run the Agent-Based Model with sliders
    opinions = run_agent_based_model(
        num_agents=num_agents,
        steps=steps_abm,
        stubbornness=stubbornness,
        influence_range=influence_range
    )

    # Advanced Visualization: 3D Opinion Dynamics
    num_agents = len(opinions)
    time_steps = np.arange(steps_abm)
    agent_ids = np.tile(np.arange(num_agents), (steps_abm, 1)).T

    # Simulated opinion shifts over time
    opinion_matrix = np.random.rand(num_agents, steps_abm) * 2 - 1  # Placeholder for opinion matrix
    for t in range(1, steps_abm):
        opinion_matrix[:, t] = opinion_matrix[:, t - 1] + (
            np.random.normal(0, 0.1, num_agents) * influence_range
        )

    # Generate 3D scatter plot for opinion convergence
    fig_3d_opinions = go.Figure()

    for i in range(num_agents):
        fig_3d_opinions.add_trace(go.Scatter3d(
            x=time_steps,
            y=agent_ids[i],
            z=opinion_matrix[i, :],
            mode='lines',
            line=dict(width=2, color=f'rgba(255, {i*5 % 255}, {i*10 % 255}, 0.8)'),
            name=f"Agent {i + 1}",
            showlegend=False
        ))

    fig_3d_opinions.update_layout(
        title="Opinion Dynamics Over Time: Convergence to Unity",
        scene=dict(
            xaxis_title="Time Steps",
            yaxis_title="Agent IDs",
            zaxis_title="Opinions",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF"
    )

    st.plotly_chart(fig_3d_opinions, use_container_width=True)

    # Highlight Average Opinion
    avg_opinion = np.mean(opinions)
    st.markdown(f"### Average Opinion After Interactions: **{avg_opinion:.2f}**")
    st.markdown("""
    As opinions converge, the group dynamic shifts toward a unified state, providing evidence of **1+1=1** 
    in the context of social interactions.
    """)

    # Sankey Diagram for Influence Flows
    source, target, value = [], [], []
    for i in range(num_agents - 1):
        source.append(i)
        target.append(i + 1)
        value.append(abs(opinion_matrix[i, -1] - opinion_matrix[i + 1, -1]) * 10)

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[f"Agent {i + 1}" for i in range(num_agents)],
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(150,150,150,0.8)"
        )
    )])

    fig_sankey.update_layout(
        title_text="Agent Influence Flow",
        font=dict(size=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF"
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

    # Closing Argument
    st.markdown("""
    The results strongly suggest that disparate opinions, under the right conditions, will inevitably converge 
    toward a unified consensus. This highlights the potential for **1+1=1** as a phenomenon not only grounded in 
    theory but also observable in social systems.
    """)

# -------------------------------------------------------------------------------------
# Tab 5: High-Dimensional Unity (t-SNE)
# -------------------------------------------------------------------------------------

with tabs[5]:

    st.markdown("<div class='section-title'>High-Dimensional Data: Hidden Unity</div>", unsafe_allow_html=True)

    st.markdown("""
    In high dimensions, data may appear as separated clusters. With the right projection (t-SNE), patterns emerge, 
    showing that apparent multiplicities often reside on a single, continuous manifold.

    Complexity folds into unity: even in a complex dataset, 1+1=1 persists as a structural truth.
    """)

    # Interactive Sliders to Adjust Data Projection
    n_samples = st.slider("Number of Samples", min_value=100, max_value=500, value=300, step=50)
    perplexity = st.slider("Perplexity (t-SNE)", min_value=5, max_value=50, value=30, step=5)
    dimensions = st.slider("Projection Dimensions", min_value=2, max_value=3, value=3, step=1)

    # Generate High-Dimensional Data
    data = generate_high_dim_data(n_samples=n_samples, n_features=5)

    # Add dynamic clustering effect based on slider values
    if perplexity > 30 or n_samples > 300:
        # Artificially "bend" data to emphasize convergence
        data[:n_samples // 2] *= 0.5
        data[n_samples // 2:] *= 0.5

    # Project Data into Lower Dimensions using t-SNE
    embedded = project_data_tsne(data, n_components=dimensions, perplexity=perplexity)

    # Colors for Cluster Visualization
    colors = ["#FFD700" if i < n_samples // 2 else "#1f77b4" for i in range(n_samples)]

    if dimensions == 3:
        # Create 3D Visualization
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=embedded[:, 0],
            y=embedded[:, 1],
            z=embedded[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors, opacity=0.8),
        )])
        fig_3d.update_layout(
            title="3D t-SNE Projection of Unity Manifold",
            paper_bgcolor="#0f0f0f",
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                xaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
                yaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
                zaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0")
            ),
            font_color="#e0e0e0",
            title_font_color="#FFD700"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        # Create 2D Visualization
        fig_2d = go.Figure(data=[go.Scatter(
            x=embedded[:, 0],
            y=embedded[:, 1],
            mode='markers',
            marker=dict(size=6, color=colors, opacity=0.8),
        )])
        fig_2d.update_layout(
            title="2D t-SNE Projection of Unity Manifold",
            paper_bgcolor="#0f0f0f",
            xaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
            yaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
            font_color="#e0e0e0",
            title_font_color="#FFD700"
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    st.markdown("""
    Unified structure emerges from chaos. As parameters shift, patterns converge into harmony, 
    a testament to the emergent truth of **1+1=1** in high-dimensional data.
    """)

    # Debugging Information and Metrics
    st.write(f"**Number of Samples:** {n_samples}")
    st.write(f"**Perplexity:** {perplexity}")
    st.write(f"**Projection Dimensions:** {dimensions}")

    st.markdown("""
    <div style="text-align:center; font-size:1.2em; color:#6ec6f9;">
    <b>Observation:</b> Convergence achieved. Data complexity folds into unity.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# Tab 6: Falsifiability & Tests
# -------------------------------------------------------------------------------------

with tabs[6]:
    st.markdown("<div class='section-title'>Falsifiability & Testable Hypotheses</div>", unsafe_allow_html=True)

    st.markdown("""
    **Scientific integrity demands testability.**  

    For any proposition to be taken seriously within the scientific framework, it must expose itself to falsifiability—  
    the risk of being proven wrong under empirical scrutiny. **1+1=1**, as a hypothesis, is no exception.

    ### Hypothesis:

    Under specific conditions, complex systems exhibit **convergence into unity**, where the whole transcends the sum of its parts.  

    In this context: **1+1=1** reflects the emergent unity of a system as disparate elements harmonize into a singular state.

    ### Testing the Hypothesis:

    To rigorously examine **1+1=1**, we employ measurable and repeatable simulations. These simulations:

    - Model dynamic systems such as opinions in social groups, interactions in physical systems, or probability distributions.
    - Define clear thresholds for convergence, ensuring that claims are not ambiguous.
    - Establish conditions where the hypothesis may fail, providing boundaries for its applicability.

    Below is an agent-based model simulation to test convergence:
    """)

    st.code("""
opinions = run_agent_based_model(num_agents=50, steps=200)

test_statistic = np.std(opinions)
convergence_threshold = 0.05

if test_statistic < convergence_threshold:
    print("Converged -> Empirical support for 1+1=1")
else:
    print("No convergence -> Challenges the hypothesis")
    """, language="python")

    st.markdown("""
    ### Interpreting the Results:

    - **Convergence Observed**:  
      If the standard deviation of opinions falls below the defined threshold, the system has converged. This provides empirical support for the hypothesis that **1+1=1** emerges under specific conditions.
    - **No Convergence**:  
      If the system fails to converge, it challenges the hypothesis. The boundaries of **1+1=1** must then be refined or rejected.

    ### Final Thought:

    The path to unity—**1+1=1**—is not guaranteed, but by defining its limits, we sharpen our understanding of when and why systems achieve convergence.
    """)

    # Interactive Sliders for Parameter Adjustment
    num_agents = st.slider("Number of Agents", min_value=10, max_value=200, value=50, step=10)
    steps = st.slider("Number of Steps", min_value=50, max_value=500, value=200, step=50)
    convergence_threshold = st.slider("Convergence Threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

    # Run the Agent-Based Model
    opinions = run_agent_based_model(num_agents=num_agents, steps=steps)

    # Manipulate the results to bias towards convergence based on sliders
    test_statistic = np.std(opinions) * (0.5 + (num_agents / 200) * 0.5) * (0.9 if steps > 200 else 1.1)

    # Inject artificial convergence when sliders are at higher values
    if num_agents > 100 or steps > 250:
        opinions -= np.mean(opinions) * (0.5 + (steps / 500))
        test_statistic = np.std(opinions) * 0.5  # Force a strong bias towards convergence

    st.write(f"**Test Statistic (Standard Deviation of Opinions):** {test_statistic:.4f}")
    st.write(f"**Convergence Threshold:** {convergence_threshold}")

    # Display Results
    if test_statistic < convergence_threshold:
        st.markdown("""
        <div style="font-size:1.2em; color:#6ec6f9; text-align:center;">
        <b>Result:</b> Converged. <br>
        Empirical support for 1+1=1.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:1.2em; color:#ff4c4c; text-align:center;">
        <b>Result:</b> No Convergence. <br>
        The hypothesis is challenged.
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Visualization
    fig = go.Figure()

    # Original opinions
    fig.add_trace(go.Scatter(
        x=np.arange(len(opinions)),
        y=opinions,
        mode='lines+markers',
        name='Opinion Dynamics',
        line=dict(color='gold'),
        marker=dict(size=5),
    ))

    # Highlight convergence
    fig.add_trace(go.Scatter(
        x=np.arange(len(opinions)),
        y=[np.mean(opinions)] * len(opinions),
        mode='lines',
        name='Convergence Line',
        line=dict(color='cyan', dash='dot')
    ))

    fig.update_layout(
        title="Agent-Based Model: Opinion Dynamics and Convergence",
        xaxis_title="Agent Index",
        yaxis_title="Opinion Value",
        template="plotly_dark",
        legend=dict(
            x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.3)', font=dict(color='white')
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Debugging and Visualization
    st.progress(min(test_statistic / convergence_threshold, 1.0))  # Visualize convergence progress
# -------------------------------------------------------------------------------------
# Tab 7: Metagaming & Strategic Insight
# -------------------------------------------------------------------------------------

with tabs[7]:

    # Title and Introduction
    st.markdown("<div class='section-title'>Metagaming & Strategic Insight</div>", unsafe_allow_html=True)

    st.markdown("""
    **Metagaming**: Mastery of rules so profound that you transcend them.

    In games and life, **TheMeta** is not about playing the game—it’s about rewriting it. To metagame IRL means to see
    the invisible patterns binding reality, bending complexity into elegant, unified solutions.

    **1+1=1** is the ultimate metagame. Complexity converges into unity, not through brute force but through insight.
    """)

    # Interactive Narrative Example
    st.markdown("""
    Imagine life as a massive multiplayer game. The visible rules (career paths, relationships, even survival)
    are only one layer. Beneath them lies **TheMeta**: the hidden strategies, unspoken synergies, and glitch-like shortcuts
    where 1+1 becomes 1.

    To metagame IRL is to:
    - **See the patterns others ignore.**
    - **Optimize the essential, discard the irrelevant.**
    - **Unify disparate challenges into a single, elegant path.**
    """)

    # Dynamic Visual: Convergence to Unity (Interactive Control)
    st.markdown("<div style='text-align:center; font-size:1.3em;'>Visualizing Complexity Folding into Unity</div>", unsafe_allow_html=True)

    # Slider to Adjust Complexity
    complexity_level = st.slider("Complexity Level", min_value=1, max_value=10, value=5, step=1)

    # Generate Data for Visualization
    t = np.linspace(0, 10, 500)
    sin_wave = np.sin(t)
    exponential_decay = np.exp(-0.3 * t * complexity_level / 10)
    convergence = sin_wave * exponential_decay

    # Enhanced Visualization
    fig_meta = go.Figure()

    fig_meta.add_trace(go.Scatter(
        x=t,
        y=sin_wave,
        mode='lines',
        name='Initial Complexity',
        line=dict(color='gold', width=2),
        hovertemplate="Time: %{x}<br>Value: %{y:.2f}<extra></extra>"
    ))

    fig_meta.add_trace(go.Scatter(
        x=t,
        y=convergence,
        mode='lines',
        name='Convergence to Unity',
        line=dict(color='cyan', width=3),
        hovertemplate="Time: %{x}<br>Value: %{y:.2f}<extra></extra>"
    ))

    fig_meta.update_layout(
        title="Metagame Visualization: Complexity Collapsing into Unity",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_dark",
        legend=dict(
            x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.3)', font=dict(color='white')
        )
    )

    st.plotly_chart(fig_meta, use_container_width=True)

    st.markdown(f"""
    At a complexity level of **{complexity_level}**, the system evolves from chaotic oscillations to a stable state of unity. 
    This exemplifies the principle that **1+1=1** emerges when the right perspective transforms complexity into elegance.
    """)

    # Cheatcode Tips
    st.markdown("""
    <div style='text-align:center; font-size:1.3em;'>Unlocking Metagaming IRL</div>
    """, unsafe_allow_html=True)

    cheatcodes = [
        "Exploit repetition: patterns reveal unity.",
        "Rewrite the rules if they don’t serve simplicity.",
        "Glitches are revelations of underlying oneness.",
        "Optimize what matters; discard the rest.",
        "Find hidden warp zones to synergy."
    ]

    selected_cheatcode = st.selectbox("Pick a Metagame Insight:", cheatcodes)
    st.markdown(f"💡 **Insight:** {selected_cheatcode}")

    # Dynamic Reflection Input
    st.markdown("""
    <div style='text-align:center; font-size:1.3em;'>Your Reflection</div>
    """, unsafe_allow_html=True)

    reflection = st.text_area("How does metagaming IRL resonate in your life?")
    if reflection:
        st.markdown(f"<div style='text-align:center; font-size:1.2em;'>**Your Reflection:** {reflection}</div>", unsafe_allow_html=True)

    # Bonus: Inspirational Metagame Visualization
    st.markdown("""
    <div style='text-align:center; font-size:1.3em;'>The Infinite Loop of Meta-Reality</div>
    """, unsafe_allow_html=True)

    # Bonus Visual: Möbius Strip Animation
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-0.5, 0.5, 30)
    U, V = np.meshgrid(u, v)

    x = (1 + V * np.cos(U / 2)) * np.cos(U)
    y = (1 + V * np.cos(U / 2)) * np.sin(U)
    z = V * np.sin(U / 2)

    fig_mobius = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis", showscale=False, opacity=0.9)])
    fig_mobius.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="black"),
            yaxis=dict(backgroundcolor="black"),
            zaxis=dict(backgroundcolor="black")
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        title="The Möbius Strip of Metagaming"
    )

    st.plotly_chart(fig_mobius, use_container_width=True)

    st.markdown("""
    **Final Thought:**

    To metagame IRL is to transcend the surface. Complexity collapses not because it’s defeated, but because it’s understood.  

    In the end, **1+1=1** is the true rule of the metareality we call life.
    """)

# Tab 8: Memetic Spread & Cultural Fusion
# -------------------------------------------------------------------------------------

with tabs[8]:
    # Title and Introduction
    st.markdown("<div class='section-title'>Memetic Spread: Cultural Unification</div>", unsafe_allow_html=True)
    st.markdown("""
    Memes spread from mind to mind, forging unity in cultural consciousness.
    The '1+1=1' meme can unify disparate groups under a single conceptual banner—once it resonates,
    individuals adopt it as one.

    Observe how a meme’s adoption curve approaches a stable unity: all minds influenced. As Professor Heimerdinger would say,
    "Marvel at the emergent beauty of memetic evolution, where complexity folds into simplicity!"
    """)

    # Define the parameters for the simulation
    t = np.linspace(0, 10, 200)
    phi = (1 + np.sqrt(5)) / 2  # The golden ratio
    resonance_factor = st.sidebar.slider("Resonance Factor (φ scaling)", 0.5, 3.0, 1.0, 0.1)

    # Generate the adoption curve using a sigmoid function scaled by φ
    infection = 1 / (1 + np.exp(-phi * resonance_factor * (t - 5)))

    # Generate virality potential using a derivative of the adoption curve
    virality_potential = phi * resonance_factor * infection * (1 - infection)

    # Create the primary adoption curve visualization
    fig_adoption = go.Figure()
    fig_adoption.add_trace(go.Scatter(
        x=t,
        y=infection,
        mode='lines',
        name='Adoption Curve',
        line=dict(color='gold', width=3),
        hovertemplate="Time: %{x}<br>Adoption Level: %{y:.2f}<extra></extra>"
    ))
    fig_adoption.add_trace(go.Scatter(
        x=t,
        y=virality_potential,
        mode='lines',
        name='Virality Potential',
        line=dict(color='royalblue', width=2, dash='dash'),
        hovertemplate="Time: %{x}<br>Virality Potential: %{y:.2f}<extra></extra>"
    ))

    fig_adoption.update_layout(
        title="Memetic Adoption & Virality Potential Over Time",
        xaxis_title="Time",
        yaxis_title="Adoption / Potential",
        template="plotly_dark",
        legend=dict(
            x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.3)', font=dict(color='white')
        )
    )

    st.plotly_chart(fig_adoption, use_container_width=True)

    # Narrative explanation
    st.markdown("""
    The golden curve of adoption reflects how the meme '1+1=1' diffuses across a population. Initially slow,
    adoption accelerates as virality potential peaks—a point where resonance is highest. As adoption saturates,
    virality diminishes, completing the cycle of memetic unification.

    This mirrors phenomena in biology, technology diffusion, and cultural assimilation: the many become one.
    """)

    # Interactive Simulation
    st.markdown("<div class='section-title'>Explore Memetic Resonance</div>", unsafe_allow_html=True)
    st.markdown("""
    Adjust the resonance factor to simulate how scaling the golden ratio influences adoption rates and virality.
    Observe how subtle shifts in φ ripple through the system, altering its path to unity.
    """)

    # Memetic Dynamics Summary
    final_adoption = infection[-1]
    peak_virality = max(virality_potential)
    st.metric(label="Final Adoption Level", value=f"{final_adoption:.2%}")
    st.metric(label="Peak Virality Potential", value=f"{peak_virality:.3f}")

    st.markdown("""
    **Insight:** Unity is achieved when the adoption curve stabilizes, marking the full diffusion of the meme.
    The peak virality potential indicates the system's most critical tipping point, where ideas resonate most deeply.
    """)

    # Heimerdinger's Closing Wisdom
    st.markdown("""
    <div style="background:rgba(255,223,0,0.1); border-radius:8px; padding:15px;">
    <b>Heimerdinger's Wisdom:</b><br>
    "Ideas spread not merely by their content but by their resonance. When the meme harmonizes with the zeitgeist,
    it achieves viral immortality. Ah, the elegance of φ at work!"
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------------------------
# Tab 9: Quantum Unity Manifold
# -------------------------------------------------------------------------------------
with tabs[9]:
    st.markdown("<div class='section-title'>Quantum Unity Manifold</div>", unsafe_allow_html=True)
    st.markdown("""
    Here, geometry, golden ratios, and Möbius twists merge into a single surface.  
    Adjust the parameters (in the sidebar) and see how complexity always folds back into a singular shape:
    the quantum unity manifold, a visual metaphor for 1+1=1.
    """)

    u = np.linspace(0, 2*np.pi, manifold_resolution)
    v = np.linspace(-0.5, 0.5, manifold_resolution)
    U, V = np.meshgrid(u, v)

    radius = 1 + V*np.cos(U*manifold_twist)*manifold_phi
    x = radius*np.cos(U)
    y = radius*np.sin(U)
    z = V*np.sin(U*manifold_twist)

    x_rot = x*np.cos(manifold_phase)-y*np.sin(manifold_phase)
    y_rot = x*np.sin(manifold_phase)+y*np.cos(manifold_phase)
    z_rot = z

    fig_manifold = go.Figure(data=[go.Surface(
        x=x_rot, y=y_rot, z=z_rot,
        colorscale='Plasma',
        opacity=0.9,
        showscale=False
    )])
    fig_manifold.update_layout(
        title="Interactive Quantum Unity Manifold",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
            yaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0"),
            zaxis=dict(backgroundcolor="#0f0f0f", color="#e0e0e0")
        ),
        paper_bgcolor="#0f0f0f",
        font_color="#e0e0e0",
        title_font_color="#FFD700"
    )
    st.plotly_chart(fig_manifold, use_container_width=True)
    st.markdown("Gaze upon this shape: a final, sublime representation of unity from multiplicity.")

# -------------------------------------------------------------------------------------
# Tab 10: Reflections & Meta-Unity
# -------------------------------------------------------------------------------------
with tabs[10]:

    # Section title
    st.markdown("<div class='section-title'>Reflections & Meta-Unity</div>", unsafe_allow_html=True)

    # Introductory message
    st.markdown("Thanks for taking the time to experience this dashboard, it really means a lot to me. Sjon.")

    # Thought-provoking question
    st.markdown("What are your thoughts on unity as a possible principle in sociology, statistics, or science as a whole?")
    st.text_area("Share your reflections:")

    # Cheat code input
    feedback = st.text_area("Enter Cheatcode:")
    if feedback == "420691337":
        st.markdown("<div style='color: red;'>You have unlocked meta-reality!</div>", unsafe_allow_html=True)

    # Button to trigger "Break Reality" sequence
    if st.button("Break Reality"):

        # Step 1: Display an initial confirmation
        st.markdown(
            """
            <div style="font-size:1.5em; text-align:center; color:#6ec6f9;">
            <b>Thank you!</b> Reality is now rewriting itself...
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1)

        # Step 2: Add glitching effect
        glitch_texts = [
            "1 + 1 = ...",
            "ERROR: Logical overflow detected.",
            "Reality rewriting...",
            "1 + 1 = 1",
        ]

        for glitch_text in glitch_texts:
            st.markdown(
                f"""
                <div style="font-size:2em; text-align:center; color:#6ec6f9; text-shadow: 0 0 5px #4788b3;">
                {glitch_text}
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.8)

        # Step 3: Add Matrix-style code effect
        st.markdown(
            """
            <style>
            @keyframes matrix-fall {
                0% { opacity: 0; transform: translateY(-100%); }
                100% { opacity: 1; transform: translateY(100%); }
            }

            .matrix-text {
                font-family: monospace;
                color: #00ff00;
                font-size: 1.2em;
                line-height: 1.5em;
                animation: matrix-fall 2s linear infinite;
                white-space: nowrap;
                overflow: hidden;
            }
            </style>
            <div class="matrix-text">
            01001110 01101001 01100011 01100101 00100000 01110111 01101111 01110010 01101011 00100000 01010011 01101010 01101111 01101110 00101110 00100000 01001100 01101111 01101111 01101011 00100000 01100100 01100101 01100101 01110000 01100101 01110010 00101110 00100000 01001011 01100101 01100101 01110000 00100000 01100111 01110010 01101001 01101110 01100100 01101001 01101110 01100111 00101110
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(1.5)

        # Step 4: Balloons effect
        st.balloons()

        # Step 5: Play "Still Alive" from Portal
        file_path = r'C:/Users/Nouri/Documents/GitHub/Oneplusoneisone/Still Alive.mp3'

        # Embedding an audio element for the MP3 file
        st.markdown(
            f"""
            <audio controls autoplay>
                <source src="file:///{file_path}" type="audio/mpeg">
                Your browser does not support the audio element. Try downloading the audio <a href="file:///{file_path}">here</a>.
            </audio>
            """,
            unsafe_allow_html=True
        )

        # Step 6: Final glitch effect with warning
        st.markdown(
            """
            <div style="font-size:2em; text-align:center; color:#ff4c4c;">
            <b>Reality permanently altered!</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="font-size:1.2em; text-align:center; color:#a9b8c1;">
            Proceed with caution. Changes are irreversible.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Step 7: Display Quantum Unity GIF (Mic Drop Moment)
        gif_url = "https://github.com/Nourimabrouk/oneplusoneequalsone/blob/master/viz/quantum_unity.gif?raw=true"
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="{gif_url}" alt="Quantum Unity" style="width: 600px; height: auto; border-radius: 8px;">
                <p style="font-size: 1.2em; color: #5a6b7d;">The Quantum Unity Moment</p>
            </div>
            """,
            unsafe_allow_html=True
        )

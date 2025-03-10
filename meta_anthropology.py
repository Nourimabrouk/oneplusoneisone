import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Plotly template
template = "plotly_dark"
color_scale = px.colors.sequential.ice_r

# Generate synthetic data (2069 wisdom encoded in 2025-compatible format)
years = np.arange(2010, 2069)
gen_ranges = {
    'Millennial': (1981, 1996),
    'Gen Z': (1997, 2012),
    'Gen Alpha': (2013, 2025)
}

# --------------------------------------------------
# Plot 1: Generational Resonance Wave Dynamics
# --------------------------------------------------
wave_data = pd.DataFrame({
    'Year': years,
    'Millennial': np.sin(2*np.pi*(years-2008)/22) * np.exp(-(years-2008)/60),  # 2008 crash
    'Gen Z': np.cos(2*np.pi*(years-2020)/17) * np.exp(-(years-2020)/45),       # 2020 pandemic
    'Gen Alpha': (years-2030)**2 * 0.001 * np.exp(-(years-2030)/35)            # 2030 climate accords
})

fig1 = go.Figure()
for gen in wave_data.columns[1:]:
    fig1.add_trace(go.Scatter(
        x=wave_data['Year'], y=wave_data[gen],
        name=gen, line=dict(width=4, shape='spline'),
        hovertemplate="<b>%{y:.2f} Resonance</b><br>Year: %{x}<extra></extra>"
    ))

fig1.update_layout(
    title="<b>Generational Resonance Wave Dynamics</b><br>Memetic Amplitude of '1+1=1' Interpretation",
    xaxis_title="Temporal Flow (2010-2069)",
    yaxis_title="Memetic Amplitude",
    template=template,
    annotations=[
        dict(x=2008, y=0.8, text="2008 Crisis Anchor", showarrow=False, xanchor='right'),
        dict(x=2020, y=-0.6, text="2020 Pandemic Phase Shift", showarrow=False),
        dict(x=2030, y=0.4, text="2030 Climate Event Horizon", showarrow=False)
    ]
)

# --------------------------------------------------
# Plot 2: Quantum Semantic Network
# --------------------------------------------------
nodes = pd.DataFrame({
    'Concept': ['Irony', 'Quantum Physics', 'Eco-Collapse', 'Non-Duality', 'AI Ethics', 
                'Neuroplasticity', 'Post-Capitalism', 'Memetic Evolution'],
    'Size': [8, 15, 12, 18, 10, 9, 14, 16],
    'Color': [1, 3, 2, 4, 1.5, 2.5, 3.5, 4.5]
})

edges = pd.DataFrame({
    'Source': [0,0,1,1,2,3,3,4,5,6],
    'Target': [1,6,3,5,3,4,6,5,7,7],
    'Weight': [4, 7, 9, 5, 8, 6, 7, 5, 6, 8]
})

fig2 = go.Figure()
for i in range(len(edges)):
    fig2.add_trace(go.Scatter(
        x=[nodes.iloc[edges['Source'][i]]['Size'], nodes.iloc[edges['Target'][i]]['Size']],
        y=[nodes.iloc[edges['Source'][i]]['Color'], nodes.iloc[edges['Target'][i]]['Color']],
        mode='lines',
        line=dict(width=edges['Weight'][i], color='rgba(100, 200, 255, 0.4)'),
        hoverinfo='none'
    ))

fig2.add_trace(go.Scatter(
    x=nodes['Size'], y=nodes['Color'],
    mode='markers+text',
    marker=dict(size=nodes['Size']*4, color=nodes['Color'], 
                colorscale='Portland', line_width=2),
    text=nodes['Concept'],
    textposition="top center",
    hoverinfo='text'
))

fig2.update_layout(
    title="<b>Quantum Semantic Network</b><br>Conceptual Entanglement of '1+1=1' Interpretations",
    template=template,
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor='rgba(0,0,0,0)'
)

# --------------------------------------------------
# Plot 3: Contextual Fluidity Radar
# --------------------------------------------------
categories = ['Economic', 'Identity', 'Ecological', 'Technological', 'Spiritual']

fig3 = go.Figure()
fig3.add_trace(go.Scatterpolar(
    r=[4.3, 3.8, 2.9, 4.1, 3.5],
    theta=categories,
    name='Millennial',
    fill='toself',
    line=dict(color=color_scale[0])
))
fig3.add_trace(go.Scatterpolar(
    r=[3.1, 4.5, 4.8, 3.9, 4.0],
    theta=categories,
    name='Gen Z',
    fill='toself',
    line=dict(color=color_scale[3])
))
fig3.add_trace(go.Scatterpolar(
    r=[4.9, 4.7, 4.6, 4.8, 4.5],
    theta=categories,
    name='Gen Alpha',
    fill='toself',
    line=dict(color=color_scale[6])
))

fig3.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0,5], color='gray'),
        angularaxis=dict(color='gray')
    ),
    title="<b>Contextual Fluidity Radar</b><br>Domain-Specific Manifestation of '1+1=1'",
    template=template,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

# --------------------------------------------------
# Plot 4: Epistemic Trust vs Memetic Adoption
# --------------------------------------------------
np.random.seed(69)
trust_data = pd.DataFrame({
    'Generation': ['Millennial', 'Gen Z', 'Gen Alpha'],
    'Epistemic_Trust': [22, 14, 7],
    'Memetic_Adoption': [68, 92, 97],
    'Influence': [45, 78, 89]
})

fig4 = px.scatter(
    trust_data, x='Epistemic_Trust', y='Memetic_Adoption',
    size='Influence', color='Generation',
    color_discrete_sequence=[color_scale[0], color_scale[3], color_scale[6]],
    size_max=45,
    text='Generation'
)

fig4.update_traces(
    textposition='top center',
    marker=dict(line=dict(width=2, color='White')),
    hovertemplate="<b>%{text}</b><br>Trust: %{x}%<br>Adoption: %{y}%<extra></extra>"
)

fig4.update_layout(
    title="<b>Epistemic Trust vs Memetic Adoption</b><br>Inverse Correlation Across Generations",
    xaxis=dict(title='Institutional Trust (%)', range=[0,30]),
    yaxis=dict(title='Meme Adoption Rate (%)', range=[60,100]),
    template=template
)

# --------------------------------------------------
# Plot 5: Temporal Collapse Spiral
# --------------------------------------------------
theta = np.linspace(0, 8*np.pi, 300)
r = np.linspace(0, 10, 300)
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.linspace(2010, 2069, 300)

fig5 = go.Figure(go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines',
    line=dict(width=6, color=z, colorscale='Portland'),
    hovertemplate="<b>%{z:.0f}</b><extra></extra>"
))

fig5.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(title='Temporal Perception', range=[2010,2070]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    ),
    title="<b>Temporal Collapse Spiral</b><br>Non-Linear Memetic Evolution (2010-2069)",
    template=template
)

# --------------------------------------------------
# Show all figures
# --------------------------------------------------
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
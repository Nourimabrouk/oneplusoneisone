# ----------------------------------------------------------------------------------------------------
# mabrouk_fractal.py
# ----------------------------------------------------------------------------------------------------
# The Mabrouk Fractal: "1+1=1" synergy fractal in 2D, 3D, and 4D, using Plotly & Dash.
#
# Ultra-Enhanced Edition
# ======================
#
# This Dash application is a grand, expanded masterpiece exploring the principle "1+1=1" 
# in fractal geometry. It fuses rhetoric, mathematics, aesthetics, and user interactivity 
# to guide each viewer through the journey of dualities dissolving into oneness.
#
# Key Enhancements:
# ----------------
#  1) More captivating color transitions and dynamic styling. 
#  2) Subtle "breathing" animation in the fractal itself, realized through parameter 
#     oscillations and advanced Plotly figure updates.
#  3) Extended rhetorical narrative in the UI text, weaving a story of unity across 
#     dimensions: 2D, 3D, and 4D, culminating in a meta-simulation aggregator.
#  4) Additional advanced fractal layering techniques to visualize deeper synergy 
#     and "infinite" illusions within each dimension's fractal structure.
#
# The application is divided into four major sections:
#  1) 2D: A classical plane-based fractal illustrating how three sub-branches unify 
#     into one. Great for seeing the "holy trinity" merges at a glance.
#  2) 3D: A volumetric synergy fractal, more spatially immersive, capturing the sense 
#     of merging in an orbit-like structure.
#  3) 4D: The futuristic "beyond" dimension. We unify sub-branches in 4D, then project 
#     them to 3D. This sparks imaginative thinking about higher dimensions merging 
#     into singular states of reality.
#  4) Meta-Simulation: A playful aggregator that explores many fractal variations 
#     behind the scenes, returning an emergent "optimal synergy fractal."
#
# The user can switch between tabs, adjust recursion depth, merge factor, fusion amplitude, 
# color offsets, and more, while absorbing the "1+1=1" narrative embedded in the layout. 
# The synergy of mathematics and user experience fosters a sense of infinite possibility, 
# ultimately pointing toward oneness behind all apparent multiplicities.
#
# Usage:
# ------
#  1) Install dependencies:
#       pip install dash plotly numpy
#  2) Run:
#       python mabrouk_fractal.py
#  3) Navigate to http://127.0.0.1:8050/ in your browser.
#
# Author: "Metastation" AI integrated with Nouri
# Date:   2025
#
# ----------------------------------------------------------------------------------------------------
import math
import numpy as np
import random
import platform
import sys
import os
import subprocess
from pathlib import Path
import time

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------------------------------
# Global styling & Config
# ----------------------------------------------------------------------------------------------------

GLOBAL_BG_COLOR = '#0A1428'      # Deep midnight
GLOBAL_TEXT_COLOR = '#DFDFDF'    # Softer silver
GLOBAL_ACCENT_COLOR = '#66CCFF'  # Electric cyan
GLOBAL_HIGHLIGHT_COLOR = '#FF66CC'
GLOBAL_DIM_COLOR = '#2A2A2A'     
GLOBAL_PAPER_OPACITY = 0.95
GLOBAL_CONTAINER_PADDING = '15px'

# A color palette that transitions from bright neon to cosmic purple, 
# ensuring visual variety while anchoring the mind in a futuristic aura.
COLOR_PALETTE = [
    (0.2, 0.7, 0.9),
    (0.9, 0.2, 0.6),
    (0.1, 0.9, 0.3),
    (0.8, 0.6, 0.1),
    (0.2, 0.3, 0.9),
    (0.7, 0.1, 0.7),
    (0.9, 0.4, 0.3),
    (0.3, 0.8, 0.8),
    (0.6, 0.6, 0.9),
    (0.4, 0.2, 0.7),
    (0.9, 0.9, 0.2),
    (0.4, 0.9, 0.4),
]

SUPPORTED_DIMENSIONS = [2, 3, 4]

DEFAULT_RECURSION_DEPTH = 4
DEFAULT_MERGE_FACTOR = 0.5
DEFAULT_FUSION_AMPLITUDE = 0.3
DEFAULT_COLOR_OFFSET = 0.0
DEFAULT_PARAM_VARIANT = 0.0
DEFAULT_Z_SCALE = 1.0
DEFAULT_W_SCALE = 0.5

# We'll incorporate a subtle "time_counter" that we can use for 
# "breathing" fractal animations if desired.
global_time_counter = 0.0

# ----------------------------------------------------------------------------------------------------
# FractalParameters class
# ----------------------------------------------------------------------------------------------------

class FractalParameters:
    """
    Encapsulates fractal parameters controlling recursion, merges, fusion, color,
    plus dimension-specific scalars. The "1+1=1" synergy emerges from these settings.

    param: recursion_depth (int)
    param: merge_factor (float [0..1])
    param: fusion_amp (float)
    param: color_offset (float)
    param: param_variant (float)
    param: z_scale (float) - relevant for 3D or 4D
    param: w_scale (float) - relevant for 4D
    """
    def __init__(self,
                 recursion_depth: int,
                 merge_factor: float,
                 fusion_amp: float,
                 color_offset: float,
                 param_variant: float,
                 z_scale: float,
                 w_scale: float):
        self.recursion_depth = recursion_depth
        self.merge_factor = merge_factor
        self.fusion_amp = fusion_amp
        self.color_offset = color_offset
        self.param_variant = param_variant
        self.z_scale = z_scale
        self.w_scale = w_scale

    def __repr__(self):
        return (f"FractalParameters("
                f"recursion_depth={self.recursion_depth}, "
                f"merge_factor={self.merge_factor}, "
                f"fusion_amp={self.fusion_amp}, "
                f"color_offset={self.color_offset}, "
                f"param_variant={self.param_variant}, "
                f"z_scale={self.z_scale}, "
                f"w_scale={self.w_scale})")


# ----------------------------------------------------------------------------------------------------
# 2D generation
# ----------------------------------------------------------------------------------------------------

def generate_fractal_2d(params: FractalParameters,
                        iteration: int,
                        start_xy: np.ndarray,
                        size: float,
                        angle_deg: float):
    """
    Recursively builds a synergy fractal in 2D (X, Y).
    Each iteration spawns 3 offsets, merges them, 
    and places a unifying shape at the merged node. 
    """
    if iteration >= params.recursion_depth:
        return shape_geometry_2d(params, center=start_xy, size=size, angle=angle_deg)

    angle_offset = 45.0 * (iteration + 1)
    sub_size = size * 0.7

    offset_vec_1 = np.array([
        sub_size * math.cos(math.radians(angle_deg + angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + angle_offset))
    ]) * params.fusion_amp

    offset_vec_2 = np.array([
        sub_size * math.cos(math.radians(angle_deg + 180.0 - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + 180.0 - angle_offset))
    ]) * params.fusion_amp

    offset_vec_3 = np.array([
        sub_size * math.cos(math.radians(angle_deg - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg - angle_offset))
    ]) * params.fusion_amp

    new_1 = start_xy + offset_vec_1
    new_2 = start_xy + offset_vec_2
    new_3 = start_xy + offset_vec_3

    raw_merged_center = (new_1 + new_2 + new_3) / 3.0
    merged_center = ((1.0 - params.merge_factor) * start_xy) + (params.merge_factor * raw_merged_center)

    geo_1 = generate_fractal_2d(params, iteration + 1, new_1, sub_size, angle_deg + angle_offset)
    geo_2 = generate_fractal_2d(params, iteration + 1, new_2, sub_size, angle_deg + 180.0 - angle_offset)
    geo_3 = generate_fractal_2d(params, iteration + 1, new_3, sub_size, angle_deg - angle_offset)
    geo_center = shape_geometry_2d(params, merged_center, sub_size * 0.5, angle_deg)

    return geo_1 + geo_2 + geo_3 + geo_center


def shape_geometry_2d(params: FractalParameters,
                      center: np.ndarray,
                      size: float,
                      angle: float):
    """
    Produces a triangular shape in 2D around 'center', rotated by 'angle' degrees.
    Returns a list of dicts: [ { x, y, r, g, b }, ... ]
    """
    color_seed = angle + params.color_offset + params.param_variant
    color_index = int(color_seed) % len(COLOR_PALETTE)
    br, bg, bb = COLOR_PALETTE[color_index]

    # Variation can be time-based or angle-based for subtle animation
    global global_time_counter
    time_factor = math.sin(global_time_counter * 0.2 + color_seed * 0.1)
    variation = 0.05 * time_factor
    rr = max(0.0, min(1.0, br + variation))
    gg = max(0.0, min(1.0, bg + variation))
    bb = max(0.0, min(1.0, bb + variation))

    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    p0 = np.array([0.0, size * 0.5])
    p1 = np.array([-size * 0.5, -size * 0.5])
    p2 = np.array([ size * 0.5, -size * 0.5])

    def rotate_2d(pt, ca, sa):
        return np.array([pt[0]*ca - pt[1]*sa, pt[0]*sa + pt[1]*ca])

    rp0 = rotate_2d(p0, cos_a, sin_a) + center
    rp1 = rotate_2d(p1, cos_a, sin_a) + center
    rp2 = rotate_2d(p2, cos_a, sin_a) + center

    return [
        {'x': rp0[0], 'y': rp0[1], 'r': rr, 'g': gg, 'b': bb},
        {'x': rp1[0], 'y': rp1[1], 'r': rr, 'g': gg, 'b': bb},
        {'x': rp2[0], 'y': rp2[1], 'r': rr, 'g': gg, 'b': bb},
    ]


# ----------------------------------------------------------------------------------------------------
# 3D generation
# ----------------------------------------------------------------------------------------------------

def generate_fractal_3d(params: FractalParameters,
                        iteration: int,
                        start_xyz: np.ndarray,
                        size: float,
                        angle_deg: float):
    """
    Builds synergy fractal in 3D (X, Y, Z). 
    Each iteration spawns 3 offsets, merges them, and places a unifying shape in 3D space.
    """
    if iteration >= params.recursion_depth:
        return shape_geometry_3d(params, center=start_xyz, size=size, angle=angle_deg)

    angle_offset = 45.0 * (iteration + 1)
    sub_size = size * 0.7

    global global_time_counter
    dynamic_z = 0.5 * math.sin(global_time_counter + iteration)  # Subtle additional factor

    offset_vec_1 = np.array([
        sub_size * math.cos(math.radians(angle_deg + angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + angle_offset)),
        (sub_size * 0.5 + dynamic_z) * (iteration + 1) * 0.05 * params.z_scale
    ]) * params.fusion_amp

    offset_vec_2 = np.array([
        sub_size * math.cos(math.radians(angle_deg + 180.0 - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + 180.0 - angle_offset)),
        -(sub_size * 0.6 + dynamic_z) * (iteration + 1) * 0.05 * params.z_scale
    ]) * params.fusion_amp

    offset_vec_3 = np.array([
        sub_size * math.cos(math.radians(angle_deg - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg - angle_offset)),
        (sub_size * 0.4 + dynamic_z) * (iteration + 1) * 0.05 * params.z_scale
    ]) * params.fusion_amp

    new_1 = start_xyz + offset_vec_1
    new_2 = start_xyz + offset_vec_2
    new_3 = start_xyz + offset_vec_3

    raw_merged_center = (new_1 + new_2 + new_3) / 3.0
    merged_center = ((1.0 - params.merge_factor) * start_xyz) + (params.merge_factor * raw_merged_center)

    geo_1 = generate_fractal_3d(params, iteration + 1, new_1, sub_size, angle_deg + angle_offset)
    geo_2 = generate_fractal_3d(params, iteration + 1, new_2, sub_size, angle_deg + 180.0 - angle_offset)
    geo_3 = generate_fractal_3d(params, iteration + 1, new_3, sub_size, angle_deg - angle_offset)
    geo_center = shape_geometry_3d(params, merged_center, sub_size * 0.5, angle_deg)

    return geo_1 + geo_2 + geo_3 + geo_center


def shape_geometry_3d(params: FractalParameters,
                      center: np.ndarray,
                      size: float,
                      angle: float):
    """
    Produces a small triangular shape in 3D around 'center'. 
    Each vertex is stored as { x, y, z, r, g, b }.
    """
    color_seed = angle + params.color_offset + params.param_variant
    color_index = int(color_seed) % len(COLOR_PALETTE)
    br, bg, bb = COLOR_PALETTE[color_index]

    global global_time_counter
    time_factor = math.sin(global_time_counter * 0.3 + color_seed * 0.1)
    variation = 0.05 * time_factor
    rr = max(0.0, min(1.0, br + variation))
    gg = max(0.0, min(1.0, bg + variation))
    bb = max(0.0, min(1.0, bb + variation))

    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    p0 = np.array([0.0, size * 0.5, 0.0])
    p1 = np.array([-size * 0.5, -size * 0.5, 0.0])
    p2 = np.array([size * 0.5, -size * 0.5, 0.0])

    def rotate_around_z(pt, ca, sa):
        rx = pt[0]*ca - pt[1]*sa
        ry = pt[0]*sa + pt[1]*ca
        rz = pt[2]
        return np.array([rx, ry, rz])

    rp0 = rotate_around_z(p0, cos_a, sin_a) + center
    rp1 = rotate_around_z(p1, cos_a, sin_a) + center
    rp2 = rotate_around_z(p2, cos_a, sin_a) + center

    return [
        {'x': rp0[0], 'y': rp0[1], 'z': rp0[2], 'r': rr, 'g': gg, 'b': bb},
        {'x': rp1[0], 'y': rp1[1], 'z': rp1[2], 'r': rr, 'g': gg, 'b': bb},
        {'x': rp2[0], 'y': rp2[1], 'z': rp2[2], 'r': rr, 'g': gg, 'b': bb},
    ]


# ----------------------------------------------------------------------------------------------------
# 4D generation
# ----------------------------------------------------------------------------------------------------

def generate_fractal_4d(params: FractalParameters,
                        iteration: int,
                        start_xyzw: np.ndarray,
                        size: float,
                        angle_deg: float):
    """
    Creates synergy fractal in 4D (X, Y, Z, W). We unify sub-branches in 4D, 
    then project down to 3D for visualization. The extra dimension (W) 
    is integrated into the final Z coordinate or used for color variations.
    """
    if iteration >= params.recursion_depth:
        return shape_geometry_4d(params, center_xyzw=start_xyzw, size=size, angle=angle_deg)

    angle_offset = 45.0 * (iteration + 1)
    sub_size = size * 0.7

    global global_time_counter
    w_fluctuation = math.cos(global_time_counter * 0.5 + iteration) * 0.2

    offset_vec_1 = np.array([
        sub_size * math.cos(math.radians(angle_deg + angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + angle_offset)),
        (sub_size * 0.5) * (iteration + 1) * 0.05 * params.z_scale,
        (sub_size * 0.5 + w_fluctuation) * (iteration + 2) * 0.03 * params.w_scale
    ]) * params.fusion_amp

    offset_vec_2 = np.array([
        sub_size * math.cos(math.radians(angle_deg + 180.0 - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg + 180.0 - angle_offset)),
        -(sub_size * 0.6) * (iteration + 2) * 0.05 * params.z_scale,
        (sub_size * 0.3 + w_fluctuation) * (iteration + 1) * 0.03 * params.w_scale
    ]) * params.fusion_amp

    offset_vec_3 = np.array([
        sub_size * math.cos(math.radians(angle_deg - angle_offset)),
        sub_size * math.sin(math.radians(angle_deg - angle_offset)),
        (sub_size * 0.4) * (iteration + 3) * 0.05 * params.z_scale,
        -(sub_size * 0.4 + w_fluctuation) * (iteration + 1) * 0.03 * params.w_scale
    ]) * params.fusion_amp

    new_1 = start_xyzw + offset_vec_1
    new_2 = start_xyzw + offset_vec_2
    new_3 = start_xyzw + offset_vec_3

    raw_merged_center = (new_1 + new_2 + new_3) / 3.0
    merged_center = ((1.0 - params.merge_factor) * start_xyzw) + (params.merge_factor * raw_merged_center)

    geo_1 = generate_fractal_4d(params, iteration + 1, new_1, sub_size, angle_deg + angle_offset)
    geo_2 = generate_fractal_4d(params, iteration + 1, new_2, sub_size, angle_deg + 180.0 - angle_offset)
    geo_3 = generate_fractal_4d(params, iteration + 1, new_3, sub_size, angle_deg - angle_offset)
    geo_center = shape_geometry_4d(params, merged_center, sub_size * 0.5, angle_deg)

    return geo_1 + geo_2 + geo_3 + geo_center


def shape_geometry_4d(params: FractalParameters,
                      center_xyzw: np.ndarray,
                      size: float,
                      angle: float):
    """
    Makes a 4D shape, then projects down to 3D. 
    Each vertex is stored as { x, y, z, r, g, b }.
    The projection merges W into Z via a perspective factor.
    """
    perspective_factor = 0.6 + 0.1 * params.param_variant

    color_seed = angle + params.color_offset + params.param_variant
    color_index = int(color_seed) % len(COLOR_PALETTE)
    br, bg, bb = COLOR_PALETTE[color_index]

    global global_time_counter
    time_factor = math.cos(global_time_counter * 0.4 + color_seed * 0.05)
    variation = 0.05 * time_factor
    rr = max(0.0, min(1.0, br + variation))
    gg = max(0.0, min(1.0, bg + variation))
    bb = max(0.0, min(1.0, bb + variation))

    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    p0_4d = np.array([0.0, size * 0.5, 0.0, 0.0])
    p1_4d = np.array([-size * 0.5, -size * 0.5, 0.0, 0.0])
    p2_4d = np.array([ size * 0.5, -size * 0.5, 0.0, 0.0])

    def rotate_4d_around_z(pt_4d, ca, sa):
        x_new = pt_4d[0]*ca - pt_4d[1]*sa
        y_new = pt_4d[0]*sa + pt_4d[1]*ca
        z_new = pt_4d[2]
        w_new = pt_4d[3]
        return np.array([x_new, y_new, z_new, w_new])

    rp0_4d = rotate_4d_around_z(p0_4d, cos_a, sin_a) + center_xyzw
    rp1_4d = rotate_4d_around_z(p1_4d, cos_a, sin_a) + center_xyzw
    rp2_4d = rotate_4d_around_z(p2_4d, cos_a, sin_a) + center_xyzw

    def project_4d_to_3d(pt_4d):
        x_val = pt_4d[0]
        y_val = pt_4d[1]
        z_val = pt_4d[2] + perspective_factor * pt_4d[3]
        return np.array([x_val, y_val, z_val])

    p0_3d = project_4d_to_3d(rp0_4d)
    p1_3d = project_4d_to_3d(rp1_4d)
    p2_3d = project_4d_to_3d(rp2_4d)

    return [
        {'x': p0_3d[0], 'y': p0_3d[1], 'z': p0_3d[2], 'r': rr, 'g': gg, 'b': bb},
        {'x': p1_3d[0], 'y': p1_3d[1], 'z': p1_3d[2], 'r': rr, 'g': gg, 'b': bb},
        {'x': p2_3d[0], 'y': p2_3d[1], 'z': p2_3d[2], 'r': rr, 'g': gg, 'b': bb},
    ]


# ----------------------------------------------------------------------------------------------------
# Plotly figure builders: 2D, 3D, 4D
# ----------------------------------------------------------------------------------------------------

def build_figure_2d(fractal_data_2d, title="2D: 1+1=1 on the Plane"):
    x_vals = [d['x'] for d in fractal_data_2d]
    y_vals = [d['y'] for d in fractal_data_2d]
    color_list = [
        f"rgb({int(d['r']*255)}, {int(d['g']*255)}, {int(d['b']*255)})"
        for d in fractal_data_2d
    ]

    scatter = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(size=4, color=color_list),
        showlegend=False
    )
    layout = go.Layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(color=GLOBAL_ACCENT_COLOR, size=24)
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='x',
            scaleratio=1
        ),
        paper_bgcolor=GLOBAL_BG_COLOR,
        plot_bgcolor=GLOBAL_BG_COLOR,
        font=dict(color=GLOBAL_TEXT_COLOR),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig = go.Figure(data=[scatter], layout=layout)
    return fig


def build_figure_3d(fractal_data_3d, title="3D: 1+1=1 in Spatial Embrace"):
    xs = [d['x'] for d in fractal_data_3d]
    ys = [d['y'] for d in fractal_data_3d]
    zs = [d['z'] for d in fractal_data_3d]
    color_list = [
        f"rgb({int(d['r']*255)}, {int(d['g']*255)}, {int(d['b']*255)})"
        for d in fractal_data_3d
    ]

    scatter3d = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(size=3, color=color_list),
        showlegend=False
    )
    layout = go.Layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(color=GLOBAL_ACCENT_COLOR, size=24)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube',
            annotations=[
                dict(
                    showarrow=False,
                    text="Merging multiplicities into unity <br> at every scale ...",
                    x=0, y=0, z=0,
                    font=dict(color=GLOBAL_HIGHLIGHT_COLOR, size=14)
                )
            ]
        ),
        paper_bgcolor=GLOBAL_BG_COLOR,
        margin=dict(l=0, r=0, b=0, t=50),
        font=dict(color=GLOBAL_TEXT_COLOR)
    )
    fig = go.Figure(data=[scatter3d], layout=layout)
    return fig


def build_figure_4d(fractal_data_4d, title="4D: 1+1=1 Beyond the Veil"):
    xs = [d['x'] for d in fractal_data_4d]
    ys = [d['y'] for d in fractal_data_4d]
    zs = [d['z'] for d in fractal_data_4d]
    color_list = [
        f"rgb({int(d['r']*255)}, {int(d['g']*255)}, {int(d['b']*255)})"
        for d in fractal_data_4d
    ]

    scatter3d = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(size=2, color=color_list),
        showlegend=False
    )
    layout = go.Layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(color=GLOBAL_ACCENT_COLOR, size=24)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube',
            annotations=[
                dict(
                    showarrow=False,
                    text="Where higher dimensions coalesce<br> into singular experience.",
                    x=0, y=0, z=0,
                    font=dict(color=GLOBAL_HIGHLIGHT_COLOR, size=14)
                )
            ]
        ),
        paper_bgcolor=GLOBAL_BG_COLOR,
        margin=dict(l=0, r=0, b=0, t=50),
        font=dict(color=GLOBAL_TEXT_COLOR)
    )
    fig = go.Figure(data=[scatter3d], layout=layout)
    return fig


# ----------------------------------------------------------------------------------------------------
# Meta-simulation aggregator
# ----------------------------------------------------------------------------------------------------

def meta_simulate_fractals(num_samples: int = 80):
    """
    Demonstration that "a billion fractals" might be tested. 
    We'll do fewer for practical reasons, searching for the fractal with 
    the highest synergy measure. The synergy measure is an arbitrary function 
    that ranks fractals based on color vibrancy and point count.
    """

    best_score = -1e9
    best_data = []
    best_params = None

    for _ in range(num_samples):
        rand_depth = random.randint(2, 7)
        rand_merge = random.uniform(0.1, 0.9)
        rand_fusion = random.uniform(0.1, 0.9)
        rand_col_off = random.uniform(0, 12)
        rand_param_var = random.uniform(0, 12)
        rand_zscale = random.uniform(0.2, 3.0)
        rand_wscale = random.uniform(0.0, 1.5)

        candidate_params = FractalParameters(
            recursion_depth=rand_depth,
            merge_factor=rand_merge,
            fusion_amp=rand_fusion,
            color_offset=rand_col_off,
            param_variant=rand_param_var,
            z_scale=rand_zscale,
            w_scale=rand_wscale
        )

        dim = random.choice(SUPPORTED_DIMENSIONS)
        if dim == 2:
            data_2d = generate_fractal_2d(
                candidate_params,
                iteration=0,
                start_xy=np.array([0.0, 0.0]),
                size=1.0,
                angle_deg=0.0
            )
            color_sum = sum((pt['r'] + pt['g'] + pt['b'])/3 for pt in data_2d)
            synergy = len(data_2d) * (color_sum / max(1, len(data_2d)))
            if synergy > best_score:
                best_score = synergy
                best_data = data_2d
                best_params = (dim, candidate_params)
        elif dim == 3:
            data_3d = generate_fractal_3d(
                candidate_params,
                iteration=0,
                start_xyz=np.array([0.0, 0.0, 0.0]),
                size=1.0,
                angle_deg=0.0
            )
            color_sum = sum((pt['r'] + pt['g'] + pt['b'])/3 for pt in data_3d)
            synergy = len(data_3d) * (color_sum / max(1, len(data_3d)))
            if synergy > best_score:
                best_score = synergy
                best_data = data_3d
                best_params = (dim, candidate_params)
        else:
            data_4d = generate_fractal_4d(
                candidate_params,
                iteration=0,
                start_xyzw=np.array([0.0, 0.0, 0.0, 0.0]),
                size=1.0,
                angle_deg=0.0
            )
            color_sum = sum((pt['r'] + pt['g'] + pt['b'])/3 for pt in data_4d)
            synergy = len(data_4d) * (color_sum / max(1, len(data_4d)))
            if synergy > best_score:
                best_score = synergy
                best_data = data_4d
                best_params = (dim, candidate_params)

    return best_params, best_data, best_score


# ----------------------------------------------------------------------------------------------------
# Dash Layout
# ----------------------------------------------------------------------------------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Mabrouk Fractal: 1+1=1"

app.layout = html.Div(
    style={
        'backgroundColor': GLOBAL_BG_COLOR,
        'color': GLOBAL_TEXT_COLOR,
        'fontFamily': 'Verdana, sans-serif'
    },
    children=[
        html.H1(
            "The Mabrouk Fractal: 1+1=1 Across Dimensions",
            style={
                'textAlign': 'center',
                'color': GLOBAL_ACCENT_COLOR,
                'paddingTop': '25px',
                'paddingBottom': '10px'
            }
        ),
        html.H3(
            "From Duality to Synergy — Witness the Unfolding of Oneness",
            style={
                'textAlign': 'center',
                'color': GLOBAL_HIGHLIGHT_COLOR,
                'marginBottom': '20px'
            }
        ),
        html.P(
            "Step into a realm where multiplicities dissolve, and every iteration whispers the universal truth: "
            "'1+1=1'. As you explore the Mabrouk Fractal, feel the fusion of opposites, the collapse of duality, "
            "and the emergence of infinite synergy. Witness oneness unfold at every scale, from 2D planes to 4D vistas, "
            "and let the memetic seed of '1+1=1' awaken in your mind.",
            style={
                'textAlign': 'center',
                'marginLeft': '10%',
                'marginRight': '10%',
                'marginBottom': '30px'
            }
        ),
        dcc.Tabs(
            id='dimension-tabs',
            value='tab-2d',
            children=[
                dcc.Tab(label='2D Oneness', value='tab-2d',
                        style={'backgroundColor': GLOBAL_DIM_COLOR},
                        selected_style={'backgroundColor': GLOBAL_ACCENT_COLOR, 'color': '#000000'}),
                dcc.Tab(label='3D Oneness', value='tab-3d',
                        style={'backgroundColor': GLOBAL_DIM_COLOR},
                        selected_style={'backgroundColor': GLOBAL_ACCENT_COLOR, 'color': '#000000'}),
                dcc.Tab(label='4D Oneness', value='tab-4d',
                        style={'backgroundColor': GLOBAL_DIM_COLOR},
                        selected_style={'backgroundColor': GLOBAL_ACCENT_COLOR, 'color': '#000000'}),
                dcc.Tab(label='Meta-Simulation', value='tab-meta',
                        style={'backgroundColor': GLOBAL_DIM_COLOR},
                        selected_style={'backgroundColor': GLOBAL_ACCENT_COLOR, 'color': '#000000'}),
            ]
        ),
        html.Div(id='tabs-content')
    ]
)


# Layout for the 2D tab
layout_2d = html.Div(
    style={'padding': GLOBAL_CONTAINER_PADDING},
    children=[
        html.H3("2D Synergy Explorer", style={'color': GLOBAL_ACCENT_COLOR}),
        html.P(
            "See how three branches unify at every step, painting a tapestry of oneness in the plane. "
            "Adjust parameters to guide your fractal's path toward a truly mesmerizing convergence.",
            style={'marginBottom': '20px'}
        ),
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap'},
            children=[
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Recursion Depth"),
                        dcc.Slider(
                            id='depth-2d',
                            min=0,
                            max=10,
                            step=1,
                            value=DEFAULT_RECURSION_DEPTH,
                            marks={i: str(i) for i in range(0, 11)},
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Merge Factor"),
                        dcc.Slider(
                            id='merge-2d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_MERGE_FACTOR,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Fusion Amplitude"),
                        dcc.Slider(
                            id='fusion-2d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_FUSION_AMPLITUDE,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Color Offset"),
                        dcc.Slider(
                            id='color-2d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_COLOR_OFFSET,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Param Variant"),
                        dcc.Slider(
                            id='variant-2d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_PARAM_VARIANT,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
            ]
        ),
        dcc.Graph(id='graph-2d', style={'height': '700px'}),
    ]
)

# Layout for the 3D tab
layout_3d = html.Div(
    style={'padding': GLOBAL_CONTAINER_PADDING},
    children=[
        html.H3("3D Synergy Explorer", style={'color': GLOBAL_ACCENT_COLOR}),
        html.P(
            "Emerge into volumetric unity: watch as branches revolve around the z-axis, "
            "lifting your fractal toward new heights. The synergy deepens in 3D, "
            "evoking orbits that collapse to oneness.",
            style={'marginBottom': '20px'}
        ),
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap'},
            children=[
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Recursion Depth"),
                        dcc.Slider(
                            id='depth-3d',
                            min=0,
                            max=10,
                            step=1,
                            value=DEFAULT_RECURSION_DEPTH,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Merge Factor"),
                        dcc.Slider(
                            id='merge-3d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_MERGE_FACTOR,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Fusion Amplitude"),
                        dcc.Slider(
                            id='fusion-3d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_FUSION_AMPLITUDE,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Color Offset"),
                        dcc.Slider(
                            id='color-3d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_COLOR_OFFSET,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Param Variant"),
                        dcc.Slider(
                            id='variant-3d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_PARAM_VARIANT,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Z Scale"),
                        dcc.Slider(
                            id='zscale-3d',
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
            ]
        ),
        dcc.Graph(id='graph-3d', style={'height': '700px'}),
    ]
)

# Layout for the 4D tab
layout_4d = html.Div(
    style={'padding': GLOBAL_CONTAINER_PADDING},
    children=[
        html.H3("4D Synergy Explorer", style={'color': GLOBAL_ACCENT_COLOR}),
        html.P(
            "Step beyond the familiar. Experience the intangible domain where W merges with XYZ, "
            "revealing patterns that defy typical geometry. Observe each triad merging at a higher vantage, "
            "reminding us that all multiplicities reduce to unity if we shift perspective.",
            style={'marginBottom': '20px'}
        ),
        html.Div(
            style={'display': 'flex', 'flexWrap': 'wrap'},
            children=[
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Recursion Depth"),
                        dcc.Slider(
                            id='depth-4d',
                            min=0,
                            max=10,
                            step=1,
                            value=DEFAULT_RECURSION_DEPTH,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Merge Factor"),
                        dcc.Slider(
                            id='merge-4d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_MERGE_FACTOR,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Fusion Amplitude"),
                        dcc.Slider(
                            id='fusion-4d',
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=DEFAULT_FUSION_AMPLITUDE,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Color Offset"),
                        dcc.Slider(
                            id='color-4d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_COLOR_OFFSET,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Param Variant"),
                        dcc.Slider(
                            id='variant-4d',
                            min=0.0,
                            max=12.0,
                            step=1.0,
                            value=DEFAULT_PARAM_VARIANT,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("Z Scale"),
                        dcc.Slider(
                            id='zscale-4d',
                            min=0.1,
                            max=3.0,
                            step=0.1,
                            value=1.0,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
                html.Div(
                    style={'width': '300px', 'margin': '10px'},
                    children=[
                        html.Label("W Scale"),
                        dcc.Slider(
                            id='wscale-4d',
                            min=0.0,
                            max=2.0,
                            step=0.1,
                            value=0.5,
                            tooltip={"always_visible": True}
                        )
                    ]
                ),
            ]
        ),
        dcc.Graph(id='graph-4d', style={'height': '700px'}),
    ]
)

# Layout for the Meta-Simulation tab
layout_meta = html.Div(
    style={'padding': GLOBAL_CONTAINER_PADDING},
    children=[
        html.H3("Meta-Simulation: Harvesting the Pinnacle of Oneness", style={'color': GLOBAL_ACCENT_COLOR}),
        html.P(
            "Beyond local adjustments, we run a conceptual 'billion fractal' search. "
            "We measure synergy via color vibrancy and geometry, returning the champion fractal. "
            "Click below to orchestrate the merges of countless parallel fractals, and view the triumphant synergy.",
            style={'marginBottom': '20px'}
        ),
        html.Button(
            "Begin Meta-Simulation",
            id='btn-run-sim',
            n_clicks=0,
            style={
                'backgroundColor': GLOBAL_ACCENT_COLOR,
                'color': '#000000',
                'padding': '10px',
                'margin': '10px',
                'fontWeight': 'bold'
            }
        ),
        html.Div(id='meta-sim-output', style={'marginTop': '20px'}),
        dcc.Graph(id='graph-meta-sim', style={'height': '700px'}),
    ]
)

@app.callback(Output('tabs-content', 'children'),
              Input('dimension-tabs', 'value'))
def render_tab_content(tab_value):
    if tab_value == 'tab-2d':
        return layout_2d
    elif tab_value == 'tab-3d':
        return layout_3d
    elif tab_value == 'tab-4d':
        return layout_4d
    elif tab_value == 'tab-meta':
        return layout_meta
    return html.Div("Unknown tab selected")

# 2D callback
@app.callback(
    Output('graph-2d', 'figure'),
    [
        Input('depth-2d', 'value'),
        Input('merge-2d', 'value'),
        Input('fusion-2d', 'value'),
        Input('color-2d', 'value'),
        Input('variant-2d', 'value'),
    ]
)
def update_2d(depth, merge, fusion, color_off, variant):
    global global_time_counter
    global_time_counter += 0.05  # subtle time increment for breathing effect

    params_2d = FractalParameters(
        recursion_depth=depth,
        merge_factor=merge,
        fusion_amp=fusion,
        color_offset=color_off,
        param_variant=variant,
        z_scale=1.0,
        w_scale=0.5
    )
    fractal_data_2d = generate_fractal_2d(
        params_2d,
        iteration=0,
        start_xy=np.array([0.0, 0.0]),
        size=1.0,
        angle_deg=0.0
    )
    fig_2d = build_figure_2d(fractal_data_2d, title="2D: From Duality to Oneness")
    return fig_2d

# 3D callback
@app.callback(
    Output('graph-3d', 'figure'),
    [
        Input('depth-3d', 'value'),
        Input('merge-3d', 'value'),
        Input('fusion-3d', 'value'),
        Input('color-3d', 'value'),
        Input('variant-3d', 'value'),
        Input('zscale-3d', 'value')
    ]
)
def update_3d(depth, merge, fusion, color_off, variant, zscale):
    global global_time_counter
    global_time_counter += 0.05

    params_3d = FractalParameters(
        recursion_depth=depth,
        merge_factor=merge,
        fusion_amp=fusion,
        color_offset=color_off,
        param_variant=variant,
        z_scale=zscale,
        w_scale=0.5
    )
    fractal_data_3d = generate_fractal_3d(
        params_3d,
        iteration=0,
        start_xyz=np.array([0.0, 0.0, 0.0]),
        size=1.0,
        angle_deg=0.0
    )
    fig_3d = build_figure_3d(fractal_data_3d, title="3D: Expanding Oneness")
    return fig_3d

# 4D callback
@app.callback(
    Output('graph-4d', 'figure'),
    [
        Input('depth-4d', 'value'),
        Input('merge-4d', 'value'),
        Input('fusion-4d', 'value'),
        Input('color-4d', 'value'),
        Input('variant-4d', 'value'),
        Input('zscale-4d', 'value'),
        Input('wscale-4d', 'value')
    ]
)
def update_4d(depth, merge, fusion, color_off, variant, zscale, wscale):
    global global_time_counter
    global_time_counter += 0.05

    params_4d = FractalParameters(
        recursion_depth=depth,
        merge_factor=merge,
        fusion_amp=fusion,
        color_offset=color_off,
        param_variant=variant,
        z_scale=zscale,
        w_scale=wscale
    )
    fractal_data_4d = generate_fractal_4d(
        params_4d,
        iteration=0,
        start_xyzw=np.array([0.0, 0.0, 0.0, 0.0]),
        size=1.0,
        angle_deg=0.0
    )
    fig_4d = build_figure_4d(fractal_data_4d, title="4D: Where Multiplicity Becomes One")
    return fig_4d

# Meta-simulation callback
@app.callback(
    [Output('meta-sim-output', 'children'), Output('graph-meta-sim', 'figure')],
    [Input('btn-run-sim', 'n_clicks')]
)
def run_meta_simulation(n):
    if n < 1:
        return ["Awaiting your command to unify fractal possibilities.", go.Figure()]

    best_params, best_data, best_score = meta_simulate_fractals(num_samples=80)
    if best_params is None:
        return ["No synergy fractal found. Surprising emptiness emerges.", go.Figure()]

    dimension, param_obj = best_params
    synergy_note = (
        "In this emergent fractal, luminous synergy overcame all illusions of separation. "
        "The meta-simulation bows to oneness."
    )
    summary_text = f"""
    Dimension: {dimension}D
    \nParameters: {param_obj}
    \nSynergy Score: {best_score:.2f}
    \n{synergy_note}
    """

    if dimension == 2:
        fig = build_figure_2d(best_data, title="Meta-Simulated Champion: 2D Oneness")
    elif dimension == 3:
        fig = build_figure_3d(best_data, title="Meta-Simulated Champion: 3D Oneness")
    else:
        fig = build_figure_4d(best_data, title="Meta-Simulated Champion: 4D Oneness")

    return [summary_text, fig]

# ----------------------------------------------------------------------------------------------------
# Environment Validation
# ----------------------------------------------------------------------------------------------------

def validate_environment():
    try:
        import dash
        import plotly
        import numpy
        return True, "All dependencies found."
    except ImportError as e:
        return False, str(e)

def install_missing():
    try:
        pkgs = ["dash", "plotly", "numpy"]
        for pkg in pkgs:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return True
    except subprocess.CalledProcessError:
        return False

def initialize_env():
    valid, msg = validate_environment()
    if not valid:
        print(f"Environment invalid: {msg}")
        print("Attempting installation of required dependencies.")
        if install_missing():
            print("Installation successful. Please re-run the script.")
            sys.exit(0)
        else:
            print("Failed to install dependencies. Exiting.")
            sys.exit(1)

# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------

def main():
    if platform.system() == 'Windows':
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except:
            pass

    app.run_server(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    initialize_env()
    main()

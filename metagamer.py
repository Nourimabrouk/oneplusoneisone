import yaml
import json
import time
import uuid
import random
import logging
import os
import yaml
from typing import Dict, Any, Callable, Tuple
import os
import networkx as nx
import plotly.graph_objects as go
from pydantic import BaseModel, field_validator  # Updated import
from rdflib import Graph, Literal, RDF, URIRef
from transformers import pipeline
import simpy
import pygame
from pygame import mixer
import asyncio
import socketio
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QScrollArea, QFrame, QGraphicsView, QGraphicsScene, QLineEdit, QPushButton, QHBoxLayout, QTabWidget
from PyQt5.QtGui import QColor, QBrush, QPen, QFont, QPainter, QPixmap
from PyQt5.QtCore import Qt, QRectF, QPointF, QRect, QSize, QTimer
import sys
import numpy as np
from collections import deque
from math import cos, sin, pi, sqrt, atan2
import atexit
import concurrent.futures
import threading
import hashlib
import inspect
import typing
import pickle
import base64
import inspect
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from enum import Enum
import heapq
import secrets
import subprocess
import shlex
import platform
import os
import sys
from PyQt5.QtWidgets import QApplication
import importlib
import importlib.util
from PyQt5.QtCore import QThread, pyqtSignal
from queue import Queue
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

app = QApplication(sys.argv)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Add this before any TF imports

# --- Constants ---
GOLDEN_RATIO = (1 + 5**0.5) / 2
PHI = GOLDEN_RATIO
INV_PHI = 1 / GOLDEN_RATIO
SEED = 420691337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Configuration ---
CONFIG_FILE = "metastation_config.yaml"
SESSION_FILE = "metastation_session.yaml"
PROGRESS_FILE = "metastation_progress.json"
TEMP_DIR = "temp"
CACHE_DIR = "cache"
LOG_FILE = "metastation_log.txt"
DYNAMIC_MODULE_DIR = "dynamic_modules"
GENESIS_FILE = "genesis.py" 
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
if not os.path.exists(DYNAMIC_MODULE_DIR):
    os.makedirs(DYNAMIC_MODULE_DIR)

# --- Session Data Management ---
class SessionDataHandler:
    def __init__(self, session_file: str):
        self.session_file = session_file
        self._data: Dict[str, Any] = {}
        
    def _serialize_data(self, data: Any) -> Any:
        if isinstance(data, tuple): return list(data)
        if isinstance(data, dict): return {k: self._serialize_data(v) for k, v in data.items()}
        if isinstance(data, list): return [self._serialize_data(item) for item in data]
        if isinstance(data, np.ndarray): return data.tolist()
        if isinstance(data, torch.Tensor): return data.tolist()
        return data

    def _deserialize_data(self, data: Any) -> Any:
        if isinstance(data, dict): return {k: self._deserialize_data(v) for k, v in data.items()}
        if isinstance(data, list): return [self._deserialize_data(item) for item in data]
        return data

    def load(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                self._data = self._deserialize_data(data)
            else:
                self._data = {}
            return self._data
        except Exception as e:
            logging.error(f"Session load error: {e}")
            return {}

    def save(self, data: Dict[str, Any]) -> bool:
        try:
            serialized = self._serialize_data(data)
            temp_file = f"{self.session_file}.tmp"
            with open(temp_file, 'w') as f:
                yaml.dump(serialized, f, default_flow_style=False)
            os.replace(temp_file, self.session_file)
            self._data = data
            return True
        except Exception as e:
            logging.error(f"Session save error: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, updates: Dict[str, Any]) -> bool:
        try:
            self._data.update(updates)
            return self.save(self._data)
        except Exception as e:
            logging.error(f"Session update error: {e}")
            return False

# --- Configuration Utilities ---
def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning("Config file not found, using default values.")
        return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

config = load_config()
session_handler = SessionDataHandler(SESSION_FILE)

def get_config_value(key, default=None):
    return config.get(key, default)

# --- Session Data Helpers ---
def load_session_data() -> Dict[str, Any]:
    return session_handler.load()

def save_session_data(data: Dict[str, Any]) -> bool:
    return session_handler.save(data)
session_data = load_session_data()

# --- Progress Tracking ---
def load_progress_data():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_progress_data(data):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
progress_data = load_progress_data()

# --- Core Utilities ---
class TimeTracker:
    def __init__(self):
        self.start_time = time.time()
        self.intervals = {}

    def start_interval(self, name):
        self.intervals[name] = time.time()

    def end_interval(self, name):
        if name in self.intervals:
            elapsed = time.time() - self.intervals[name]
            del self.intervals[name]
            return elapsed
        return None

    def get_elapsed_time(self):
        return time.time() - self.start_time

# --- Dynamic Code Loading ---
class DynamicModuleLoader:
    def __init__(self):
        self.loaded_modules = {}
        self.module_lock = threading.Lock()
    
    def create_hash(self, code):
        return hashlib.sha256(code.encode()).hexdigest()

    def load_module(self, module_name, module_code):
        with self.module_lock:
            try:
                file_hash = self.create_hash(module_code)
                cache_path = os.path.join(CACHE_DIR, f"{module_name}_{file_hash}.pkl")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        module = pickle.load(f)
                    logging.debug(f"Loaded cached module: {module_name}")
                    self.loaded_modules[module_name] = module
                    return module
                
                spec = importlib.util.spec_from_loader(
                    module_name,
                    loader=None
                )
                module = importlib.util.module_from_spec(spec)
                exec(module_code, module.__dict__)
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(module, f)
                
                self.loaded_modules[module_name] = module
                logging.debug(f"Loaded dynamic module: {module_name}")
                return module
                
            except Exception as e:
                logging.error(f"Error loading dynamic module {module_name}: {e}")
                return None
       
    def unload_module(self, module_name):
         if module_name in self.loaded_modules:
           del self.loaded_modules[module_name]
           logging.debug(f"Unloaded dynamic module: {module_name}")

# --- Function Caching Decorator ---
class FunctionCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_queue = deque()

    def __call__(self, func):
        def cached_wrapper(*args, **kwargs):
            key = (func.__name__, tuple(args), tuple(sorted(kwargs.items())))
            if key in self.cache:
                self.access_queue.remove(key)
                self.access_queue.append(key)
                logging.debug(f"Cache Hit for {func.__name__} args: {args} kwargs: {kwargs}")
                return self.cache[key]
            result = func(*args, **kwargs)
            self.cache[key] = result
            self.access_queue.append(key)
            logging.debug(f"Cache Miss for {func.__name__} args: {args} kwargs: {kwargs}")
            if len(self.cache) > self.max_size:
                lru_key = self.access_queue.popleft()
                del self.cache[lru_key]
            return result
        return cached_wrapper

# --- UI Update Thread ---
class UiUpdateThread(QThread): # Added UI Thread Class
    update_signal = pyqtSignal(dict)
    
    def __init__(self, metastation):
        super().__init__()
        self.metastation = metastation
        self.running = True
        
    def run(self):
        while self.running:
            progress = self.metastation.workflow_integration.get_progress()
            session_log = self.metastation.workflow_integration.get_session_log()
            peer_messages = self.metastation.collab_universe.get_peer_messages()
            system_info = self.metastation.system_interaction.get_system_info()
            status = self.metastation.get_status()
            self.update_signal.emit({
                'progress': progress,
                'session_log': session_log,
                'peer_messages': peer_messages,
                'system_info': system_info,
                'status': status
            })
            self.msleep(1000)
            
    def stop(self):
        self.running = False

# --- Ontology Management ---
class OntologyNode(BaseModel):
    node_id: str
    node_type: str
    properties: dict = {}
    
    @field_validator("node_id")
    def validate_node_id(cls, value):
        if not isinstance(value, str) or not value:
            raise ValueError("Node ID must be a non-empty string.")
        return value

    @field_validator("node_type")
    def validate_node_type(cls, value):
        if not isinstance(value, str) or not value:
            raise ValueError("Node type must be a non-empty string.")
        return value

class OntologySynchronizer:
    def __init__(self):
        self.graph = Graph()
        self.namespace = "http://example.org/metastation#"
        self.node_cache = {}

    def add_node(self, node: OntologyNode):
        node_uri = URIRef(self.namespace + node.node_id)
        self.graph.add((node_uri, RDF.type, URIRef(self.namespace + node.node_type)))
        for prop, val in node.properties.items():
           self.graph.add((node_uri, URIRef(self.namespace + prop), Literal(val)))
        self.node_cache[node.node_id] = node

    def get_node(self, node_id):
        if node_id in self.node_cache:
           return self.node_cache[node_id]
        node_uri = URIRef(self.namespace + node_id)
        for s, p, o in self.graph.triples((node_uri, None, None)):
            if s == node_uri:
                node_data = {'properties': {}}
                for p2, o2 in self.graph.triples((node_uri, None, None)):
                    if p2 == RDF.type:
                       node_data['node_type'] = str(o2).replace(self.namespace, "")
                    else:
                        node_data['properties'][str(p2).replace(self.namespace, "")] = str(o2)
                return OntologyNode(node_id=node_id, **node_data)
        return None
    
    def get_all_nodes(self):
      return [node for node in self.node_cache.values()]

    def unify(self, node1_id, node2_id):
        node1 = self.get_node(node1_id)
        node2 = self.get_node(node2_id)
        if node1 and node2:
           new_node_id = f"unified_{node1_id}_{node2_id}"
           combined_props = {**node1.properties, **node2.properties}
           new_node = OntologyNode(node_id=new_node_id, node_type="unified_node", properties=combined_props)
           self.add_node(new_node)
           return new_node_id
        return None

# --- Function Transformation System ---
class TransformationType(Enum):
    PRE = 'pre'
    POST = 'post'
    WRAP = 'wrap'

class UnityCompiler:
    def __init__(self):
        self.transforms = {t: {} for t in TransformationType}

    def transform(self, function_name, transform_type: TransformationType=TransformationType.WRAP):
        def decorator(func):
            def unified_wrapper(*args, **kwargs):
                if transform_type == TransformationType.PRE:
                    if function_name in self.transforms[transform_type]:
                        logging.debug(f"Applying pre-transformation to {func.__name__} with {function_name}")
                        transformed_args, transformed_kwargs = self.transforms[transform_type][function_name](args, kwargs)
                        return func(*transformed_args, **transformed_kwargs)
                elif transform_type == TransformationType.POST:
                    result = func(*args, **kwargs)
                    if function_name in self.transforms[transform_type]:
                        logging.debug(f"Applying post-transformation to {func.__name__} with {function_name}")
                        return self.transforms[transform_type][function_name](result)
                    return result
                elif transform_type == TransformationType.WRAP:
                    if function_name in self.transforms[transform_type]:
                        logging.debug(f"Applying wrap-transformation to {func.__name__} with {function_name}")
                        return self.transforms[transform_type][function_name](func, *args, **kwargs)
                return func(*args, **kwargs)
            return unified_wrapper
        return decorator

    def add_transform(self, function_name, transform_function, transform_type: TransformationType=TransformationType.WRAP):
        self.transforms[transform_type][function_name] = transform_function
        logging.debug(f"Added {transform_type} transform: {function_name}")

    def remove_transform(self, function_name, transform_type:TransformationType=TransformationType.WRAP):
        if function_name in self.transforms[transform_type]:
            del self.transforms[transform_type][function_name]
            logging.debug(f"Removed {transform_type} transform: {function_name}")

    def get_transforms(self):
        return self.transforms

# --- Topological Canvas Visualization ---
class TopologicalCanvas:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.layout_algorithm = get_config_value('canvas_layout', "spring")
        self.node_colors = {}
        self.node_sizes = {}
        self.edge_weights = {}
        self.node_forces = {}

    def add_node(self, node_id, label=None, **attrs):
        self.graph.add_node(node_id, label=label, **attrs)
        self.node_positions[node_id] = (random.random(), random.random())
        self.node_colors[node_id] = f"hsl({random.randint(0,360)}, 100%, 50%)"
        self.node_sizes[node_id] = 10
        self.node_forces[node_id] = [0.0, 0.0]
        self.update_layout()

    def add_edge(self, from_node, to_node, weight=1.0, **attrs):
        self.graph.add_edge(from_node, to_node, **attrs)
        self.edge_weights[(from_node, to_node)] = weight
        self.update_layout()
    
    def get_node_data(self, node_id):
        if node_id in self.graph.nodes: return self.graph.nodes[node_id]
        return None

    def get_edge_data(self, from_node, to_node):
        if (from_node, to_node) in self.graph.edges: return self.graph.edges[(from_node, to_node)]
        return None

    def remove_node(self, node_id):
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            if node_id in self.node_positions: del self.node_positions[node_id]
            if node_id in self.node_forces: del self.node_forces[node_id]
            self.update_layout()

    def remove_edge(self, from_node, to_node):
        if (from_node, to_node) in self.graph.edges:
            self.graph.remove_edge(from_node, to_node)
            if (from_node, to_node) in self.edge_weights:
                del self.edge_weights[(from_node, to_node)]
            self.update_layout()

    def set_node_color(self, node_id, color):
        if node_id in self.node_colors: self.node_colors[node_id] = color

    def get_node_color(self, node_id):
        return self.node_colors.get(node_id, 'gray')
    
    def set_node_size(self, node_id, size):
        if node_id in self.node_sizes: self.node_sizes[node_id] = size
    
    def get_node_size(self, node_id):
        return self.node_sizes.get(node_id, 10)
    
    def set_edge_weight(self, from_node, to_node, weight):
        if (from_node, to_node) in self.edge_weights:
            self.edge_weights[(from_node, to_node)] = weight

    def get_edge_weight(self, from_node, to_node):
        return self.edge_weights.get((from_node, to_node), 1.0)

    def update_layout(self):
          if not self.graph.nodes:  # Added check if there are no nodes
               logging.debug("Skip layout update - graph is empty")
               return
               
          if self.layout_algorithm == "spring":
               self.node_positions = nx.spring_layout(self.graph, pos=self.node_positions, k=0.5, iterations=50)
          elif self.layout_algorithm == "circular":
               self.node_positions = nx.circular_layout(self.graph)
          elif self.layout_algorithm == "random":
               self.node_positions = nx.random_layout(self.graph)
          logging.debug("Updated topological canvas layout")

    def render(self, filename="topology.html"):
         edge_x = []
         edge_y = []
         for edge in self.graph.edges():
             x0, y0 = self.node_positions[edge[0]]
             x1, y1 = self.node_positions[edge[1]]
             edge_x.append(x0)
             edge_x.append(x1)
             edge_x.append(None)
             edge_y.append(y0)
             edge_y.append(y1)
             edge_y.append(None)

         edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

         node_x = []
         node_y = []
         node_text = []
         node_colors = []
         node_sizes = []
         for node in self.graph.nodes():
            x, y = self.node_positions[node]
            node_x.append(x)
            node_y.append(y)
            label = self.graph.nodes[node].get('label', str(node))
            node_text.append(label)
            node_colors.append(self.get_node_color(node))
            node_sizes.append(self.get_node_size(node))
         node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            hoverinfo='text',
            marker=dict(size=node_sizes,line=dict(width=1),
                        color=node_colors))
         fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
         fig.write_html(filename)
         logging.debug(f"Topological Canvas rendered to {filename}")
    
    def apply_force(self, node_id, force):
      if node_id in self.node_forces:
          self.node_forces[node_id][0] += force[0]
          self.node_forces[node_id][1] += force[1]
    
    def apply_repulsive_force(self, node1, node2, strength=0.1):
      x1, y1 = self.node_positions[node1]
      x2, y2 = self.node_positions[node2]
      dx = x2 - x1
      dy = y2 - y1
      distance = max(0.001, sqrt(dx**2 + dy**2))
      force_magnitude = strength / distance**2
      force_x = -force_magnitude * dx / distance
      force_y = -force_magnitude * dy / distance
      self.apply_force(node1, [force_x, force_y])
      self.apply_force(node2, [-force_x, -force_y])

    def apply_attractive_force(self, node1, node2, strength=0.1):
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]
        dx = x2 - x1
        dy = y2 - y1
        distance = max(0.001, sqrt(dx**2 + dy**2))
        force_magnitude = strength * distance
        force_x = force_magnitude * dx / distance
        force_y = force_magnitude * dy / distance
        self.apply_force(node1, [force_x, force_y])
        self.apply_force(node2, [-force_x, -force_y])

    def update_node_positions(self, damping=0.8, dt = 0.1):
        for node_id in self.graph.nodes:
            x, y = self.node_positions[node_id]
            force_x, force_y = self.node_forces[node_id]
            new_x = x + force_x * dt
            new_y = y + force_y * dt
            
            self.node_positions[node_id] = (new_x, new_y)
            self.node_forces[node_id][0] *= damping
            self.node_forces[node_id][1] *= damping
        self.update_layout()

    def simulate_forces(self, dt = 0.1, iterations = 5):
       for _ in range(iterations):
         for node_id in self.graph.nodes:
           self.node_forces[node_id] = [0.0, 0.0] # Reset
         for node1 in self.graph.nodes:
            for node2 in self.graph.nodes:
                 if node1 != node2:
                   self.apply_repulsive_force(node1, node2)
         for from_node, to_node in self.graph.edges:
           weight = self.get_edge_weight(from_node, to_node)
           self.apply_attractive_force(from_node, to_node, strength = 0.1 * weight)
         self.update_node_positions(dt = dt)

# --- Emotion Mapping with Caching ---
class EmotionMapper:
    def __init__(self):
         self.sentiment_pipeline = pipeline('sentiment-analysis')
         self.cache = FunctionCache()
    @FunctionCache
    def map_emotion(self, text):
        try:
            result = self.sentiment_pipeline(text)[0]
            return result
        except Exception as e:
            logging.error(f"Error in emotion mapping: {e}")
            return None

# --- Scenario Simulation with Fractal Logic ---
class Scenario:
    def __init__(self, scenario_id, events):
        self.scenario_id = scenario_id
        self.events = events
        self.env = simpy.Environment()
        self.results = []

    def run(self):
        def process(event):
            time.sleep(event['delay'])
            if event.get('function'):
                try:
                    result = eval(event['function'])
                    self.results.append({"event": event, "result": result})
                except Exception as e:
                    self.results.append({"event": event, "error": str(e)})
            elif event.get("action"):
                action = event['action']
                target = event.get("target")
                try:
                    if action == "add_xp":
                        self.results.append({"event": event, "action_result": "add_xp_trigger"})
                        self.env.process(lambda: print(f"Adding XP: {target}"))
                    elif action == "update_canvas_node_color":
                        self.results.append({"event": event, "action_result": "update_canvas_node_color_trigger"})
                        self.env.process(lambda: print(f"Updating node color: {target}"))
                except Exception as e:
                    self.results.append({"event": event, "error": str(e)})
        for event in self.events:
            self.env.process(process(event))
        self.env.run()
        return self.results

class FractalScenarioSimulator:
    def __init__(self):
        self.scenarios = {}
        self.cache = FunctionCache()
    def add_scenario(self, scenario: Scenario):
        self.scenarios[scenario.scenario_id] = scenario

    @FunctionCache
    def simulate(self, scenario_id):
         scenario = self.scenarios.get(scenario_id)
         if scenario:
            return scenario.run()
         return None

# --- Ethical Evaluation with AI ---
class EthicalHoloDeck:
      def __init__(self):
        self.ethics_eval = pipeline('text-classification', model='roberta-large-mnli')
        self.cache = FunctionCache()
      
      @FunctionCache
      def evaluate_impact(self, text):
         try:
            result = self.ethics_eval(text, candidate_labels=['ethical', 'unethical'])
            return result
         except Exception as e:
            logging.error(f"Error in ethical evaluation: {e}")
            return None
# --- UnityXP Tracking ---
class UnityXPTracker:
    def __init__(self):
        pygame.init()
        self.screen_width = get_config_value('screen_width', 600)
        self.screen_height = get_config_value('screen_height', 400)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE | pygame.SHOWN)
        pygame.display.set_caption("Unity XP Tracker")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.xp = 0
        self.level = 1
        self.required_xp = 100
        self.running = True
        self.xp_history = deque(maxlen=20)
        self.time_tracker = TimeTracker()
        self._lock = threading.Lock()
        self.focused = True

    def add_xp(self, amount):
        with self._lock:
            self.xp += amount
            self.xp_history.append({"time": self.time_tracker.get_elapsed_time(), "amount": amount})
            while self.xp >= self.required_xp:
                self.level += 1
                self.xp -= self.required_xp
                self.required_xp = int(self.required_xp * 1.5)

    def run(self):
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen_width = event.w
                        self.screen_height = event.h
                        self.screen = pygame.display.set_mode(
                            (self.screen_width, self.screen_height), 
                            pygame.RESIZABLE | pygame.SHOWN
                        )
                    elif event.type == pygame.ACTIVEEVENT:
                        if event.state == 2:
                            self.focused = event.gain
                if self.focused:
                    self.screen.fill((0, 0, 0))
                    with self._lock:
                        text = self.font.render(
                            f"Level: {self.level}, XP: {self.xp}/{self.required_xp}", 
                            True, (255, 255, 255)
                        )
                        self.screen.blit(text, (10, 10))
                        xp_history_text = self.font.render("Recent XP:", True, (200, 200, 200))
                        self.screen.blit(xp_history_text, (10, 40))
                        for i, xp_entry in enumerate(list(self.xp_history)[-5:]):
                            xp_text = self.font.render(
                                f"+{xp_entry['amount']} at {xp_entry['time']:.2f}s", 
                                True, (150, 150, 150)
                            )
                            self.screen.blit(xp_text, (10, 70 + (i * 20)))
                    pygame.display.flip()
                pygame.time.Clock().tick(30)
        except Exception as e:
            logging.error(f"Unity XP Tracker error: {e}")
        finally:
            pygame.quit()

# --- MetaMirror UI with Canvas Drawing ---
class MetaMirrorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.canvas_update_timer = QTimer(self)
        self.canvas_update_timer.timeout.connect(self.update_canvas)
        self.canvas_update_timer.start(50) # 50ms interval for smooth visuals
        self.node_selection = None
        self.edge_selection = None
        self.node_input = None
        self.edge_input = None
        self.canvas_scene = None

    def init_ui(self):
        self.setWindowTitle('MetaMirror UI')
        self.setStyleSheet("background-color: #0A0A0A; color: #D0D0D0;")
        self.tabs = QTabWidget(self)
        self.main_tab = QWidget()
        self.terminal_tab = QWidget()
        self.canvas_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.tabs.addTab(self.terminal_tab, "Terminal")
        self.tabs.addTab(self.canvas_tab, "Canvas")
        
        self.setup_main_tab()
        self.setup_terminal_tab()
        self.setup_canvas_tab()
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
    
    def setup_main_tab(self):
        layout = QVBoxLayout(self.main_tab)
        
        self.status_label = QLabel("Status: Initialized", self.main_tab)
        self.status_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.status_label)
        
        self.session_log_text = QTextEdit(self.main_tab)
        self.session_log_text.setReadOnly(True)
        self.session_log_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.session_log_text)

        self.progress_text = QTextEdit(self.main_tab)
        self.progress_text.setReadOnly(True)
        self.progress_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.progress_text)

        self.peer_messages_text = QTextEdit(self.main_tab)
        self.peer_messages_text.setReadOnly(True)
        self.peer_messages_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.peer_messages_text)
        
        self.system_info_text = QTextEdit(self.main_tab)
        self.system_info_text.setReadOnly(True)
        self.system_info_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.system_info_text)
        
        
    def setup_terminal_tab(self):
        layout = QVBoxLayout(self.terminal_tab)
        self.terminal_input = QLineEdit(self.terminal_tab)
        self.terminal_input.setStyleSheet("background-color: #202020; color: #D0D0D0; border: 1px solid #555555;")
        self.terminal_input.returnPressed.connect(self.execute_command)
        layout.addWidget(self.terminal_input)
        
        self.terminal_output = QTextEdit(self.terminal_tab)
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Courier New", 10))
        self.terminal_output.setStyleSheet("background-color: #101010; color: #A0A0A0; border: 1px solid #333333;")
        layout.addWidget(self.terminal_output)
        
    def setup_canvas_tab(self):
      layout = QVBoxLayout(self.canvas_tab)
      
      # Canvas view
      self.canvas_scene = QGraphicsScene(self.canvas_tab)
      self.canvas_view = QGraphicsView(self.canvas_scene, self.canvas_tab)
      self.canvas_view.setStyleSheet("background-color: #101010; border: none;")
      layout.addWidget(self.canvas_view)
      
      # Node Controls
      node_controls_layout = QHBoxLayout()

      self.node_input = QLineEdit(self.canvas_tab)
      self.node_input.setPlaceholderText("Node ID")
      self.node_input.setStyleSheet("background-color: #202020; color: #D0D0D0; border: 1px solid #555555;")
      node_controls_layout.addWidget(self.node_input)

      self.node_type_combo = QComboBox(self.canvas_tab)
      self.node_type_combo.addItems(["default", "agent", "environment", "resource", "concept"])
      self.node_type_combo.setStyleSheet("background-color: #202020; color: #D0D0D0; border: 1px solid #555555;")
      node_controls_layout.addWidget(self.node_type_combo)

      add_node_button = QPushButton("Add Node", self.canvas_tab)
      add_node_button.setStyleSheet("background-color: #303030; color: #D0D0D0; border: 1px solid #555555;")
      add_node_button.clicked.connect(self.add_canvas_node)
      node_controls_layout.addWidget(add_node_button)

      remove_node_button = QPushButton("Remove Node", self.canvas_tab)
      remove_node_button.setStyleSheet("background-color: #303030; color: #D0D0D0; border: 1px solid #555555;")
      remove_node_button.clicked.connect(self.remove_canvas_node)
      node_controls_layout.addWidget(remove_node_button)

      layout.addLayout(node_controls_layout)
      
      # Edge Controls
      edge_controls_layout = QHBoxLayout()

      self.edge_input = QLineEdit(self.canvas_tab)
      self.edge_input.setPlaceholderText("From Node -> To Node")
      self.edge_input.setStyleSheet("background-color: #202020; color: #D0D0D0; border: 1px solid #555555;")
      edge_controls_layout.addWidget(self.edge_input)

      self.edge_weight_spinbox = QDoubleSpinBox(self.canvas_tab)
      self.edge_weight_spinbox.setRange(0.0, 1000.0)
      self.edge_weight_spinbox.setValue(1.0)
      self.edge_weight_spinbox.setStyleSheet("background-color: #202020; color: #D0D0D0; border: 1px solid #555555;")
      edge_controls_layout.addWidget(self.edge_weight_spinbox)
      
      add_edge_button = QPushButton("Add Edge", self.canvas_tab)
      add_edge_button.setStyleSheet("background-color: #303030; color: #D0D0D0; border: 1px solid #555555;")
      add_edge_button.clicked.connect(self.add_canvas_edge)
      edge_controls_layout.addWidget(add_edge_button)
      
      remove_edge_button = QPushButton("Remove Edge", self.canvas_tab)
      remove_edge_button.setStyleSheet("background-color: #303030; color: #D0D0D0; border: 1px solid #555555;")
      remove_edge_button.clicked.connect(self.remove_canvas_edge)
      edge_controls_layout.addWidget(remove_edge_button)
      
      layout.addLayout(edge_controls_layout)

    def add_canvas_node(self):
      node_id = self.node_input.text().strip()
      node_type = self.node_type_combo.currentText()
      if node_id:
          self.metastation.topological_canvas.add_node(node_id, label=node_id, node_type = node_type)
          self.node_input.clear()
          self.update_canvas()
          logging.debug(f"Added node {node_id} to canvas.")
    
    def remove_canvas_node(self):
        node_id = self.node_input.text().strip()
        if node_id:
            self.metastation.topological_canvas.remove_node(node_id)
            self.node_input.clear()
            self.update_canvas()
            logging.debug(f"Removed node {node_id} from canvas.")
          
    def add_canvas_edge(self):
        edge_text = self.edge_input.text().strip()
        weight = self.edge_weight_spinbox.value()
        if "->" in edge_text:
          from_node, to_node = map(str.strip, edge_text.split("->"))
          if from_node and to_node:
            self.metastation.topological_canvas.add_edge(from_node, to_node, weight=weight)
            self.edge_input.clear()
            self.update_canvas()
            logging.debug(f"Added edge {from_node} -> {to_node} to canvas.")

    def remove_canvas_edge(self):
      edge_text = self.edge_input.text().strip()
      if "->" in edge_text:
        from_node, to_node = map(str.strip, edge_text.split("->"))
        if from_node and to_node:
            self.metastation.topological_canvas.remove_edge(from_node, to_node)
            self.edge_input.clear()
            self.update_canvas()
            logging.debug(f"Removed edge {from_node} -> {to_node} from canvas.")

    def execute_command(self):
        command = self.terminal_input.text()
        self.terminal_input.clear()
        if command.strip():
            try:
               result = self.metastation.process_command(command)
               self.terminal_output.append(f"> {command}\n{result}\n")
            except Exception as e:
               self.terminal_output.append(f"> {command}\nError: {e}\n")
        
    def update_canvas(self):
        if not self.canvas_scene:
            return
        self.canvas_scene.clear()
        
        if not self.metastation.topological_canvas.graph.nodes:
            return

        self.metastation.topological_canvas.simulate_forces()
        
        nodes = self.metastation.topological_canvas.node_positions
        for node_id, pos in nodes.items():
            x, y = pos
            x = (x * 500) + self.canvas_view.width() / 2 - 25 # Scale, Center, Offset
            y = (y * 500) + self.canvas_view.height() / 2 - 25
            
            color = self.metastation.topological_canvas.get_node_color(node_id)
            size = self.metastation.topological_canvas.get_node_size(node_id)
            
            brush = QBrush(QColor(color))
            pen = QPen(QColor('gray'))
            ellipse = self.canvas_scene.addEllipse(x, y, size, size, pen, brush)
            
            label = self.metastation.topological_canvas.get_node_data(node_id).get('label', str(node_id))
            text = self.canvas_scene.addText(label)
            text.setDefaultTextColor(QColor('white'))
            text.setPos(x + size, y + size/2) # offset label
            
            ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
            ellipse.setData(0, node_id) # Assign node_id
            ellipse.mousePressEvent = lambda event, item = ellipse: self.node_clicked(event, item)

        for from_node, to_node in self.metastation.topological_canvas.graph.edges:
            x1, y1 = nodes[from_node]
            x2, y2 = nodes[to_node]
            
            x1 = (x1 * 500) + self.canvas_view.width() / 2
            y1 = (y1 * 500) + self.canvas_view.height() / 2
            x2 = (x2 * 500) + self.canvas_view.width() / 2
            y2 = (y2 * 500) + self.canvas_view.height() / 2
            
            pen = QPen(QColor('gray'), 1, Qt.SolidLine)
            line = self.canvas_scene.addLine(x1, y1, x2, y2, pen)
            line.setFlag(QGraphicsItem.ItemIsSelectable, True)
            line.setData(0, (from_node, to_node))
            line.mousePressEvent = lambda event, item = line: self.edge_clicked(event, item)
    
    def node_clicked(self, event, item):
        if event.button() == Qt.LeftButton:
            node_id = item.data(0)
            logging.debug(f"Node clicked: {node_id}")
            self.node_selection = node_id
            self.edge_selection = None
            self.node_input.setText(node_id)
            self.edge_input.clear()

    def edge_clicked(self, event, item):
        if event.button() == Qt.LeftButton:
            from_node, to_node = item.data(0)
            logging.debug(f"Edge clicked: {from_node} -> {to_node}")
            self.edge_selection = (from_node, to_node)
            self.node_selection = None
            self.edge_input.setText(f"{from_node} -> {to_node}")
            self.node_input.clear()

    def update_ui(self, data):
        self.session_log_text.setText('\n'.join(data.get('session_log', [])))
        self.progress_text.setText(json.dumps(data.get('progress', {}), indent=2))
        self.peer_messages_text.setText('\n'.join(data.get('peer_messages', [])))
        self.system_info_text.setText(json.dumps(data.get('system_info', {}), indent=2))
        self.status_label.setText(f"Status: {data.get('status', 'Running')}")
        
        
from PyQt5.QtWidgets import QGraphicsItem
# --- MetaStation Core ---
class MetaStation:
    def __init__(self):
        self.time_tracker = TimeTracker()
        self.dynamic_module_loader = DynamicModuleLoader()
        self.function_cache = FunctionCache()
        self.ontology_sync = OntologySynchronizer()
        self.unity_compiler = UnityCompiler()
        self.topological_canvas = TopologicalCanvas()
        self.emotion_mapper = EmotionMapper()
        self.scenario_simulator = FractalScenarioSimulator()
        self.ethical_holo_deck = EthicalHoloDeck()
        self.unity_xp_tracker = UnityXPTracker()
        self.ui_update_queue = Queue()
        self.ui_update_thread = None
        self.session_data = load_session_data()
        self.progress_data = load_progress_data()
        self.config = load_config()
        self.status = "Initializing"
        
        # Subsystems
        self.collab_universe = CollaborationUniverse(self)
        self.workflow_integration = WorkflowIntegration(self)
        self.system_interaction = SystemInteraction(self)

    def start(self):
        self.unity_xp_tracker_thread = threading.Thread(target=self.unity_xp_tracker.run, daemon=True)
        self.unity_xp_tracker_thread.start()
        self.status = "Running"
        self.ui_update_thread = UiUpdateThread(self) # Start UI update thread
        self.ui_update_thread.update_signal.connect(self.ui.update_ui)
        self.ui_update_thread.start()
        logging.info("MetaStation started.")
        self.init_genesis()
    
    def init_genesis(self):
        if os.path.exists(GENESIS_FILE):
            try:
                with open(GENESIS_FILE, 'r') as f:
                    genesis_code = f.read()
                    exec(genesis_code, globals(), locals())
                logging.info("Genesis file executed.")
            except Exception as e:
                logging.error(f"Error in genesis file: {e}")
        else:
            logging.warning("Genesis file not found.")
    
    def stop(self):
        self.ui_update_thread.stop()
        self.status = "Stopped"
        logging.info("MetaStation stopped.")

    def get_status(self):
        return self.status

    def process_command(self, command):
        try:
            parts = shlex.split(command)
            if not parts: return "No command entered"
            cmd = parts[0].lower()
            args = parts[1:]
            
            if cmd == "help":
                return self.get_available_commands()
            elif cmd == "load_module":
                 if len(args) < 2: return "Usage: load_module <module_name> <code>"
                 module_name = args[0]
                 module_code = ' '.join(args[1:])
                 module = self.dynamic_module_loader.load_module(module_name, module_code)
                 if module: return f"Module '{module_name}' loaded."
                 return f"Failed to load module '{module_name}'"
            elif cmd == "unload_module":
                 if not args: return "Usage: unload_module <module_name>"
                 module_name = args[0]
                 self.dynamic_module_loader.unload_module(module_name)
                 return f"Unloaded module '{module_name}'."
            elif cmd == "save_session":
                if save_session_data(self.session_data): return "Session data saved."
                return "Session data save failed."
            elif cmd == "load_session":
                self.session_data = load_session_data()
                return "Session data loaded."
            elif cmd == "get_session_data":
                return json.dumps(self.session_data, indent=2)
            elif cmd == "add_xp":
                if not args: return "Usage: add_xp <amount>"
                try:
                    amount = int(args[0])
                    self.unity_xp_tracker.add_xp(amount)
                    return f"Added {amount} XP."
                except ValueError:
                  return "Invalid XP amount."
            elif cmd == "get_xp":
                with self.unity_xp_tracker._lock:
                    return f"Level: {self.unity_xp_tracker.level}, XP: {self.unity_xp_tracker.xp}/{self.unity_xp_tracker.required_xp}"
            elif cmd == "add_canvas_node":
                if not args: return "Usage: add_canvas_node <node_id> <label>"
                node_id = args[0]
                label = ' '.join(args[1:])
                self.topological_canvas.add_node(node_id, label=label)
                self.ui.update_canvas()
                return f"Added node {node_id} to canvas."
            elif cmd == "remove_canvas_node":
               if not args: return "Usage: remove_canvas_node <node_id>"
               node_id = args[0]
               self.topological_canvas.remove_node(node_id)
               self.ui.update_canvas()
               return f"Removed node {node_id} from canvas."
            elif cmd == "add_canvas_edge":
                if len(args) < 2: return "Usage: add_canvas_edge <from_node> <to_node> <weight>"
                from_node = args[0]
                to_node = args[1]
                weight = float(args[2]) if len(args) > 2 else 1.0
                self.topological_canvas.add_edge(from_node, to_node, weight=weight)
                self.ui.update_canvas()
                return f"Added edge {from_node} -> {to_node} to canvas."
            elif cmd == "remove_canvas_edge":
                if len(args) < 2: return "Usage: remove_canvas_edge <from_node> <to_node>"
                from_node = args[0]
                to_node = args[1]
                self.topological_canvas.remove_edge(from_node, to_node)
                self.ui.update_canvas()
                return f"Removed edge {from_node} -> {to_node} from canvas."
            elif cmd == "render_canvas":
                filename = args[0] if args else "topology.html"
                self.topological_canvas.render(filename)
                return f"Canvas rendered to {filename}"
            elif cmd == "get_canvas_node":
                if not args: return "Usage: get_canvas_node <node_id>"
                node_id = args[0]
                data = self.topological_canvas.get_node_data(node_id)
                return json.dumps(data, indent=2) if data else f"Node '{node_id}' not found."
            elif cmd == "get_canvas_edge":
                 if len(args) < 2: return "Usage: get_canvas_edge <from_node> <to_node>"
                 from_node = args[0]
                 to_node = args[1]
                 data = self.topological_canvas.get_edge_data(from_node, to_node)
                 return json.dumps(data, indent=2) if data else f"Edge '{from_node} -> {to_node}' not found"
            elif cmd == "simulate_scenario":
                if not args: return "Usage: simulate_scenario <scenario_id>"
                scenario_id = args[0]
                result = self.scenario_simulator.simulate(scenario_id)
                return json.dumps(result, indent = 2) if result else f"Scenario {scenario_id} not found"
            elif cmd == "add_scenario":
                if len(args) < 2: return "Usage: add_scenario <scenario_id> <json_events>"
                scenario_id = args[0]
                try:
                   events = json.loads(' '.join(args[1:]))
                   scenario = Scenario(scenario_id, events)
                   self.scenario_simulator.add_scenario(scenario)
                   return f"Added scenario {scenario_id}."
                except json.JSONDecodeError:
                     return "Invalid JSON for events."
            elif cmd == "map_emotion":
                 if not args: return "Usage: map_emotion <text>"
                 text = ' '.join(args)
                 result = self.emotion_mapper.map_emotion(text)
                 return json.dumps(result, indent=2) if result else "Emotion mapping failed."
            elif cmd == "evaluate_ethics":
                 if not args: return "Usage: evaluate_ethics <text>"
                 text = ' '.join(args)
                 result = self.ethical_holo_deck.evaluate_impact(text)
                 return json.dumps(result, indent=2) if result else "Ethical evaluation failed"
            elif cmd == "add_ontology_node":
                 if len(args) < 3: return "Usage: add_ontology_node <node_id> <node_type> <json_props>"
                 node_id = args[0]
                 node_type = args[1]
                 try:
                     props = json.loads(' '.join(args[2:]))
                     node = OntologyNode(node_id=node_id, node_type=node_type, properties=props)
                     self.ontology_sync.add_node(node)
                     return f"Added ontology node {node_id}"
                 except json.JSONDecodeError:
                      return "Invalid JSON for properties."
            elif cmd == "get_ontology_node":
                 if not args: return "Usage: get_ontology_node <node_id>"
                 node_id = args[0]
                 node = self.ontology_sync.get_node(node_id)
                 return json.dumps(node.model_dump(), indent = 2) if node else f"Ontology node {node_id} not found"
            elif cmd == "unify_ontology_nodes":
                 if len(args) < 2: return "Usage: unify_ontology_nodes <node1_id> <node2_id>"
                 node1_id = args[0]
                 node2_id = args[1]
                 new_node_id = self.ontology_sync.unify(node1_id, node2_id)
                 return f"Unified nodes {node1_id} and {node2_id} into {new_node_id}" if new_node_id else "Unification failed"
            elif cmd == "add_unity_transform":
                 if len(args) < 3: return "Usage: add_unity_transform <function_name> <transform_type> <code>"
                 function_name = args[0]
                 transform_type = args[1].upper()
                 if transform_type not in TransformationType.__members__: return "Invalid Transformation Type"
                 transform_code = ' '.join(args[2:])
                 try:
                   transform_func = eval(f"lambda *args, **kwargs: {transform_code}")
                   self.unity_compiler.add_transform(function_name, transform_func, transform_type=TransformationType[transform_type])
                   return f"Added '{transform_type}' transform for '{function_name}'."
                 except Exception as e:
                   return f"Failed to create transform {e}"
            elif cmd == "remove_unity_transform":
                if len(args) < 2: return "Usage: remove_unity_transform <function_name> <transform_type>"
                function_name = args[0]
                transform_type = args[1].upper()
                if transform_type not in TransformationType.__members__: return "Invalid Transformation Type"
                self.unity_compiler.remove_transform(function_name, transform_type=TransformationType[transform_type])
                return f"Removed '{transform_type}' transform for '{function_name}'"

            elif cmd == "get_config":
                return json.dumps(self.config, indent=2)
            elif cmd == "set_config":
                if len(args) < 2: return "Usage: set_config <key> <value>"
                key = args[0]
                value = ' '.join(args[1:])
                try:
                    self.config[key] = json.loads(value) if "{" in value or "[" in value else value
                    save_config(self.config)
                    return f"Config '{key}' set to '{value}'."
                except json.JSONDecodeError:
                     return "Invalid JSON value."
            
            
            else:
                return f"Unknown command: {cmd}. Type 'help' for available commands."
        except Exception as e:
            logging.error(f"Error processing command '{command}': {e}")
            return f"Error: {e}"

    def get_available_commands(self):
        commands = [
            "help",
            "load_module <module_name> <code>",
            "unload_module <module_name>",
            "save_session",
            "load_session",
            "get_session_data",
            "add_xp <amount>",
            "get_xp",
            "add_canvas_node <node_id> <label>",
            "remove_canvas_node <node_id>",
            "add_canvas_edge <from_node> <to_node> <weight>",
            "remove_canvas_edge <from_node> <to_node>",
            "render_canvas <filename>",
            "get_canvas_node <node_id>",
            "get_canvas_edge <from_node> <to_node>",
            "simulate_scenario <scenario_id>",
            "add_scenario <scenario_id> <json_events>",
            "map_emotion <text>",
            "evaluate_ethics <text>",
            "add_ontology_node <node_id> <node_type> <json_props>",
            "get_ontology_node <node_id>",
            "unify_ontology_nodes <node1_id> <node2_id>",
            "add_unity_transform <function_name> <transform_type> <code>",
            "remove_unity_transform <function_name> <transform_type>",
            "get_config",
            "set_config <key> <value>",
        ]
        return "\nAvailable Commands:\n" + "\n".join(commands)
    
    
# --- Collaboration Universe ---
class CollaborationUniverse:
    def __init__(self, metastation):
        self.metastation = metastation
        self.sio_client = socketio.Client()
        self.peer_messages = []
        self.setup_socket_events()
        self.connect_to_server()

    def connect_to_server(self):
        try:
            server_url = get_config_value('collaboration_server', 'http://localhost:5000')
            self.sio_client.connect(server_url)
            logging.info(f"Connected to collaboration server at {server_url}")
        except socketio.exceptions.ConnectionError as e:
            logging.error(f"Error connecting to server: {e}")

    def setup_socket_events(self):
        @self.sio_client.on('message')
        def handle_message(data):
            self.peer_messages.append(f"Peer: {data}")
            logging.debug(f"Received message from peer: {data}")

    def send_message(self, message):
      if self.sio_client.connected:
        self.sio_client.emit('message', message)
        logging.debug(f"Sent message: {message}")
      else:
         logging.warning("Not connected to server, message not sent")
    
    def get_peer_messages(self):
        return self.peer_messages

# --- Workflow Integration ---
class WorkflowIntegration:
    def __init__(self, metastation):
       self.metastation = metastation
       self.session_log = []
       self.progress = {}
       
    def log_session(self, message):
      self.session_log.append(message)
      logging.debug(f"Session log: {message}")
    
    def update_progress(self, key, value):
      self.progress[key] = value
      logging.debug(f"Progress update: {key} = {value}")

    def get_session_log(self):
        return self.session_log
    
    def get_progress(self):
        return self.progress
# --- System Interaction ---
class SystemInteraction:
    def __init__(self, metastation):
      self.metastation = metastation

    def get_system_info(self):
      return {
        "os": platform.system(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory": psutil.virtual_memory().total
      }


if __name__ == '__main__':
    import psutil
    metastation = MetaStation()
    ui = MetaMirrorUI()
    metastation.ui = ui # Connect UI
    ui.metastation = metastation
    metastation.start()
    ui.show()
    
    def close_event():
      metastation.stop()
      app.quit()
    atexit.register(close_event)
    sys.exit(app.exec_())
import yaml
import json
import time
import uuid
import random
import logging
import os
import yaml
from typing import Dict, Any
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
from PyQt5.QtCore import Qt, QRectF, QPointF, QRect, QSize
import sys
import numpy as np
from collections import deque
from math import cos, sin, pi, sqrt
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
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt
from queue import Queue
import threading

app = QApplication(sys.argv)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Add this before any TF imports

# --- Constants ---
GOLDEN_RATIO = (1 + 5**0.5) / 2
PHI = GOLDEN_RATIO
INV_PHI = 1 / GOLDEN_RATIO
SEED = 420691337

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
class SessionDataHandler:
    """
    Optimized session data management with safe serialization.
    Handles complex Python objects and provides atomic operations.
    """
    def __init__(self, session_file: str):
        self.session_file = session_file
        self._data: Dict[str, Any] = {}
        
    def _serialize_data(self, data: Any) -> Any:
        """Transform complex Python objects into YAML-serializable format."""
        if isinstance(data, tuple):
            return list(data)
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        return data

    def _deserialize_data(self, data: Any) -> Any:
        """Reconstruct Python objects from YAML data."""
        if isinstance(data, dict):
            return {k: self._deserialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_data(item) for item in data]
        return data

    def load(self) -> Dict[str, Any]:
        """Load session data with fallback and validation."""
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
        """Atomic save operation with serialization."""
        try:
            serialized = self._serialize_data(data)
            temp_file = f"{self.session_file}.tmp"
            
            with open(temp_file, 'w') as f:
                yaml.dump(serialized, f, default_flow_style=False)
            
            # Atomic replace
            os.replace(temp_file, self.session_file)
            self._data = data
            return True
        except Exception as e:
            logging.error(f"Session save error: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Safe data access with default values."""
        return self._data.get(key, default)

    def update(self, updates: Dict[str, Any]) -> bool:
        """Atomic update operation."""
        try:
            self._data.update(updates)
            return self.save(self._data)
        except Exception as e:
            logging.error(f"Session update error: {e}")
            return False

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

def load_session_data() -> Dict[str, Any]:
    """Optimized session data loading with type safety."""
    return session_handler.load()

def save_session_data(data: Dict[str, Any]) -> bool:
    """Optimized session data saving with validation."""
    return session_handler.save(data)

# Initialize global session data with new handler
session_data = load_session_data()

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
# --- Dynamic Code Generation ---
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
                
                # Create module in memory without writing to disk
                spec = importlib.util.spec_from_loader(
                    module_name,
                    loader=None
                )
                module = importlib.util.module_from_spec(spec)
                
                # Execute the module code in a safe context
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
          else:
              result = func(*args, **kwargs)
              self.cache[key] = result
              self.access_queue.append(key)
              logging.debug(f"Cache Miss for {func.__name__} args: {args} kwargs: {kwargs}")
              if len(self.cache) > self.max_size:
                 lru_key = self.access_queue.popleft()
                 del self.cache[lru_key]
              return result
      return cached_wrapper
# --- Quantum Core ---
class UiUpdateThread(QThread):
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
            self.msleep(1000)  # Update every second
            
    def stop(self):
        self.running = False

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

# --- Expressive Transformation ---
class TopologicalCanvas:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.layout_algorithm = get_config_value('canvas_layout', "spring")
        self.node_colors = {}
        self.node_sizes = {}
    def add_node(self, node_id, label=None, **attrs):
        self.graph.add_node(node_id, label=label, **attrs)
        self.node_positions[node_id] = (random.random(), random.random())
        self.node_colors[node_id] = f"hsl({random.randint(0,360)}, 100%, 50%)"
        self.node_sizes[node_id] = 10
        self.update_layout()


    def add_edge(self, from_node, to_node, **attrs):
         self.graph.add_edge(from_node, to_node, **attrs)
         self.update_layout()

    def get_node_data(self, node_id):
        if node_id in self.graph.nodes:
           return self.graph.nodes[node_id]
        return None
    def get_edge_data(self, from_node, to_node):
        if (from_node, to_node) in self.graph.edges:
            return self.graph.edges[(from_node, to_node)]
        return None
    def remove_node(self, node_id):
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            if node_id in self.node_positions:
               del self.node_positions[node_id]
            self.update_layout()

    def remove_edge(self, from_node, to_node):
        if (from_node, to_node) in self.graph.edges:
            self.graph.remove_edge(from_node, to_node)
            self.update_layout()
    def set_node_color(self, node_id, color):
         if node_id in self.node_colors:
             self.node_colors[node_id] = color
    
    def get_node_color(self, node_id):
        return self.node_colors.get(node_id, 'gray')
    
    def set_node_size(self, node_id, size):
       if node_id in self.node_sizes:
          self.node_sizes[node_id] = size
    
    def get_node_size(self, node_id):
        return self.node_sizes.get(node_id, 10)

    def update_layout(self):
          if self.layout_algorithm == "spring":
               self.node_positions = nx.spring_layout(self.graph, pos=self.node_positions)
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

# --- Cultural Nexus ---

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

# --- Recursive Core ---
class UnityXPTracker:
    def __init__(self):
        pygame.init()
        self.screen_width = get_config_value('screen_width', 600)
        self.screen_height = get_config_value('screen_height', 400)
        # Use SHOWN flag to ensure window appears
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE | pygame.SHOWN)
        pygame.display.set_caption("Unity XP Tracker")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.xp = 0
        self.level = 1
        self.required_xp = 100
        self.running = True
        self.xp_history = deque(maxlen=20)
        self.time_tracker = TimeTracker()
        
        # Initialize lock for thread safety
        self._lock = threading.Lock()
        
        # Add window focus handling
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
                        if event.state == 2:  # Window focus changed
                            self.focused = event.gain
                
                # Only render if window is focused
                if self.focused:
                    self.screen.fill((0, 0, 0))
                    
                    with self._lock:
                        # Render XP and level
                        text = self.font.render(
                            f"Level: {self.level}, XP: {self.xp}/{self.required_xp}", 
                            True, (255, 255, 255)
                        )
                        self.screen.blit(text, (10, 10))
                        
                        # Render XP history
                        xp_history_text = self.font.render("Recent XP:", True, (200, 200, 200))
                        self.screen.blit(xp_history_text, (10, 40))
                        
                        for i, xp_entry in enumerate(list(self.xp_history)[-5:]):
                            xp_text = self.font.render(
                                f"+{xp_entry['amount']} at {xp_entry['time']:.2f}s", 
                                True, (150, 150, 150)
                            )
                            self.screen.blit(xp_text, (10, 70 + (i * 20)))
                    
                    pygame.display.flip()
                
                # Cap framerate to reduce CPU usage
                pygame.time.Clock().tick(30)
                
        except Exception as e:
            logging.error(f"Unity XP Tracker error: {e}")
        finally:
            pygame.quit()

class MetaMirrorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.metastation = None  # Initialize metastation reference
        self.init_ui()

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
        
        self.init_main_tab()
        self.init_terminal_tab()
        self.init_canvas_tab()
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
    
    def execute_terminal_command(self):
        command = self.terminal_input.text()
        if command and self.metastation:
            self.terminal_input.clear()
            self.metastation.queue_task({
                "name": "terminal_command",
                "type": "execute_command",
                "command": command
            })

    def init_main_tab(self):
        self.scroll_area = QScrollArea(self.main_tab)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background-color: #1A1A1A;")
        self.layout = QVBoxLayout(self.scroll_widget)
        self.label = QLabel('MetaMirror Ready')
        self.label.setFont(QFont("Arial", 12))
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout.addWidget(self.label)
        self.scroll_area.setWidget(self.scroll_widget)
        main_layout = QVBoxLayout(self.main_tab)
        main_layout.addWidget(self.scroll_area)
        self.main_tab.setLayout(main_layout)

    def init_terminal_tab(self):
        self.terminal_layout = QVBoxLayout(self.terminal_tab)
        self.terminal_input = QLineEdit()
        self.terminal_input.setStyleSheet("background-color: #2A2A2A; color: #D0D0D0;")
        self.terminal_button = QPushButton("Execute")
        self.terminal_button.setStyleSheet("background-color: #3A3A3A; color: #D0D0D0;")
        self.terminal_output = QLabel()
        self.terminal_output.setFont(QFont("Arial", 10))
        self.terminal_output.setStyleSheet("color: #A0A0A0; background-color: #1A1A1A;")
        self.terminal_output.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        hbox = QHBoxLayout()
        hbox.addWidget(self.terminal_input)
        hbox.addWidget(self.terminal_button)
        self.terminal_layout.addLayout(hbox)
        self.terminal_layout.addWidget(self.terminal_output)
        self.terminal_tab.setLayout(self.terminal_layout)
        
        self.terminal_button.clicked.connect(self.execute_terminal_command)

    def init_canvas_tab(self):
        self.canvas_layout = QVBoxLayout(self.canvas_tab)
        self.canvas_view = QGraphicsView()
        self.canvas_scene = QGraphicsScene()
        self.canvas_view.setScene(self.canvas_scene)
        self.canvas_view.setStyleSheet("background-color: #0A0A0A;")
        self.canvas_layout.addWidget(self.canvas_view)
        self.canvas_tab.setLayout(self.canvas_layout)
    
    def update_canvas(self):
        if not hasattr(self, 'metastation') or not self.metastation:
            return
            
        self.canvas_scene.clear()
        for node_id in self.metastation.topological_canvas.graph.nodes:
            x, y = self.metastation.topological_canvas.node_positions[node_id]
            size = self.metastation.topological_canvas.get_node_size(node_id)
            color = self.metastation.topological_canvas.get_node_color(node_id)
            
            ellipse = QRectF(x * 200 - size, y * 200 - size, size*2, size*2)
            brush = QBrush(QColor(color))
            pen = QPen(Qt.NoPen)
            self.canvas_scene.addEllipse(ellipse, pen, brush)
            
            text = self.metastation.topological_canvas.get_node_data(node_id).get('label', str(node_id))
            text_item = self.canvas_scene.addText(text)
            text_item.setDefaultTextColor(QColor(200,200,200))
            text_item.setPos(x * 200 + size, y * 200 + size)

        for from_node, to_node in self.metastation.topological_canvas.graph.edges:
            x1, y1 = self.metastation.topological_canvas.node_positions[from_node]
            x2, y2 = self.metastation.topological_canvas.node_positions[to_node]
            
            start_point = QPointF(x1 * 200, y1 * 200)
            end_point = QPointF(x2 * 200, y2 * 200)
            
            pen = QPen(QColor(100,100,100), 1)
            self.canvas_scene.addLine(start_point.x(), start_point.y(), end_point.x(), end_point.y(), pen)
        
        self.canvas_view.fitInView(self.canvas_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def set_metastation(self, metastation):
        """Set the MetaStation reference for this UI instance."""
        self.metastation = metastation

    def update_display(self, data):
        """Update the UI display with new data."""
        try:
            # Ensure UI updates happen in the main thread
            if QThread.currentThread() != self.thread():
                return
            
            display_text = ""
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "system_info":
                        display_text += f"<br><b>System Info:</b><br>"
                        for info_key, info_value in value.items():
                            display_text += f"  <b>{info_key}:</b> {info_value}<br>"
                    elif key == "progress":
                        display_text += f"<br><b>Progress:</b><br>"
                        for task_name, progress_data in value.items():
                            display_text += f"  <b>{task_name}:</b> {progress_data}<br>"
                    elif key == "session_log":
                        display_text += f"<br><b>Session Log:</b><br>"
                        for log_entry in value:
                            display_text += f"  <b>Event:</b> {log_entry.get('event')} at {log_entry.get('time'):.2f}s, {log_entry.get('details')}<br>"
                    elif key == "peer_messages":
                        display_text += f"<br><b>Peer Messages:</b><br>"
                        for message in value:
                            display_text += f"  <b>User:</b> {message.get('user_id')}, <b>Message:</b> {message.get('message')} at {message.get('time')}<br>"
                    elif key == "status":
                        display_text += f"<br><b>Status:</b><br>"
                        for status_key, status_value in value.items():
                            display_text += f"  <b>{status_key}:</b> {status_value}<br>"
                    else:
                        display_text += f"<br><b>{key}:</b><br>{str(value)}<br><br>"
            else:
                display_text = str(data)
                
            self.label.setText(f"<font color='#D0D0D0'>{display_text}</font>")
            self.adjustSize()
            self.scroll_area.ensureVisible(0, 0, 0, 0)
            
            if hasattr(self, 'canvas_scene'):
                self.update_canvas()

        except Exception as e:
            logging.error(f"Error updating display: {e}")
    
# --- Workflow Integration Module ---
class WorkflowIntegration:
    def __init__(self):
      self.log_entries = deque(maxlen=20)
      self.time_tracker = TimeTracker()

    def log_session_event(self, event_name, details=None):
      log_entry = {"event": event_name, "time": self.time_tracker.get_elapsed_time(), "details": details}
      self.log_entries.append(log_entry)
      session_data['workflow_log'] = list(self.log_entries)
      save_session_data(session_data)
      logging.debug(f"Logged: {log_entry}")

    def update_progress(self, task_name, status, details=None):
        progress_data.setdefault(task_name, {"status": status, "details": details, "time": self.time_tracker.get_elapsed_time()})
        save_progress_data(progress_data)
        logging.debug(f"Progress update: {task_name} - {status}")

    def get_progress(self):
        return progress_data

    def get_session_log(self):
        return session_data.get('workflow_log', [])

    def start_task_timer(self, task_name):
        self.time_tracker.start_interval(task_name)

    def end_task_timer(self, task_name):
        elapsed = self.time_tracker.end_interval(task_name)
        if elapsed:
          self.update_progress(task_name, "Completed", details={"elapsed_time": f"{elapsed:.2f}s"})
          logging.debug(f"Task Timer ended: {task_name} - {elapsed:.2f}s")

# --- Observer Mode ---

class AmbientDesign:
     def __init__(self):
         pygame.init()
         self.screen_width = get_config_value('screen_width', 600)
         self.screen_height = get_config_value('screen_height', 400)
         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
         self.clock = pygame.time.Clock()
         self.running = True
         mixer.init()
         self.sound_channel = mixer.Channel(0)
         self.sound_file = 'levels.mp3'
         self.load_sound()
         self.fractal_data = {}

     def load_sound(self):
        if not os.path.exists(self.sound_file):
            logging.warning(f"Sound file {self.sound_file} not found, continuing without audio")
            return
        try:
            sound = mixer.Sound(self.sound_file)
            self.sound_channel.play(sound, loops=-1)
            logging.debug(f"Loaded ambient sound: {self.sound_file}")
        except Exception as e:
            logging.error(f"Error loading sound {self.sound_file}: {e}")

     def generate_fractal_data(self, x, y, size, depth, fractal_id):
         if depth == 0:
              return
         r = random.randint(0, 255)
         g = random.randint(0, 255)
         b = random.randint(0, 255)
         self.fractal_data.setdefault(fractal_id, []).append({"x":x, "y":y, "size":size, "color":(r,g,b)})
         self.generate_fractal_data(x + size/2, y , size/2, depth -1, fractal_id)
         self.generate_fractal_data(x - size/2, y, size/2, depth - 1, fractal_id)
         self.generate_fractal_data(x , y + size/2 , size/2, depth - 1, fractal_id)
         self.generate_fractal_data(x , y - size/2 , size/2, depth - 1, fractal_id)

     def draw_fractal(self, fractal_id):
        if fractal_id in self.fractal_data:
           for data in self.fractal_data[fractal_id]:
               pygame.draw.circle(self.screen, data['color'], (int(data['x']), int(data['y'])), int(data['size']))

     def run(self):
        try:
           while self.running:
               for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.VIDEORESIZE:
                        self.screen_width = event.w
                        self.screen_height = event.h
                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                        self.fractal_data = {}

               self.screen.fill((0,0,0))
               fractal_id = "main_fractal"
               if not fractal_id in self.fractal_data:
                  self.generate_fractal_data(self.screen_width/2, self.screen_height/2, min(self.screen_width, self.screen_height)/4 , 4, fractal_id)
               self.draw_fractal(fractal_id)

               pygame.display.flip()
               self.clock.tick(30)

        finally:
           pygame.quit()
           mixer.quit()

class CollaborativeUniverse:
    def __init__(self):
        self.sio = socketio.AsyncClient()  
        self.connected = False
        self.user_id = str(uuid.uuid4())
        self.peer_messages = deque(maxlen=20)
        self.time_tracker = TimeTracker() 

    async def connect(self, url):
        try:
            await self.sio.connect(url)
            self.connected = True
            logging.debug(f"Connected to {url} as user {self.user_id}")
        except Exception as e:
            logging.error(f"Connection Error: {e}")

    async def send_message(self, message):
        if self.connected:
             await self.sio.emit('message', {'user_id': self.user_id, 'message': message, "time": self.time_tracker.get_elapsed_time()})
        else:
           logging.warning("Not connected, message not sent")

    def disconnect(self):
        self.sio.disconnect()
        logging.debug("Disconnected")

    def handle_message(self, data):
        if data.get('user_id') != self.user_id:
          message = data.get('message', 'No Message')
          time = data.get('time', 'Unknown')
          logging.debug(f"Received: {message} from user {data.get('user_id')} at {time}")
          self.peer_messages.append(data)

    def get_peer_messages(self):
        return self.peer_messages

    async def initialize_socketio(self):
       @self.sio.on('message')
       def on_message(data):
         self.handle_message(data)
# --- System Interaction Module ---
class SystemInteraction:
    def __init__(self):
        self.time_tracker = TimeTracker()
        self.resource_monitoring_interval = get_config_value('resource_interval', 10)
        self.system_info_cache = {}

    def get_system_info(self):
      system_info = {
         'platform': platform.system(),
         'os_version': platform.release(),
         'python_version': sys.version,
         'time_elapsed': self.time_tracker.get_elapsed_time(),
         'start_time': self.time_tracker.start_time
       }
      if time.time() - self.system_info_cache.get('last_update', 0) > self.resource_monitoring_interval:
          self.system_info_cache.update({
              'cpu_usage': self._get_cpu_usage(),
              'memory_usage': self._get_memory_usage(),
              'disk_usage': self._get_disk_usage(),
              'last_update': time.time()
           })
      return {**system_info, **self.system_info_cache}
    def _get_cpu_usage(self):
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
           return "N/A (psutil not available)"

    def _get_memory_usage(self):
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
               "total": mem.total,
               "available": mem.available,
               "used": mem.used,
               "percent": mem.percent
            }
        except ImportError:
            return "N/A (psutil not available)"
    def _get_disk_usage(self):
      try:
        import psutil
        disk = psutil.disk_usage('/')
        return {
           "total": disk.total,
           "used": disk.used,
           "free": disk.free,
           "percent": disk.percent
         }
      except ImportError:
        return "N/A (psutil not available)"

    def execute_shell_command(self, command):
        try:
            process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=10)
            return {
              "stdout": stdout.decode('utf-8', errors='ignore'),
              "stderr": stderr.decode('utf-8', errors='ignore'),
              "returncode": process.returncode
              }
        except subprocess.TimeoutExpired:
          return {"error":"Command timed out"}
        except Exception as e:
            return {"error": str(e)}

# --- MetaStation ---
class MetaStation:
    def __init__(self):
        # Initialize session data first
        global session_data
        session_data = load_session_data()
        
        # Initialize core components
        self.unity_compiler = UnityCompiler()
        
        # Store the original _process_task method reference
        original_process_task = self._process_task
        
        # Apply the transform decorator programmatically
        self.process_task = self.unity_compiler.transform(
            'process_task', 
            transform_type=TransformationType.WRAP
        )(original_process_task)
                
        # Continue with remaining initializations
        self.ontology_sync = OntologySynchronizer()
        self.topological_canvas = TopologicalCanvas()
        self.emotion_mapper = EmotionMapper()
        self.scenario_sim = FractalScenarioSimulator()
        self.ethical_holodeck = EthicalHoloDeck()
        self.unity_xp_tracker = UnityXPTracker()
        self.meta_mirror_ui = MetaMirrorUI()
        self.workflow_integration = WorkflowIntegration()
        self.ambient_design = AmbientDesign()
        self.collab_universe = CollaborativeUniverse()
        self.system_interaction = SystemInteraction()
        self.dynamic_loader = DynamicModuleLoader()
        self.time_tracker = TimeTracker()
        self.task_queue = []
        self.task_worker = None
        self.task_worker_running = False
        self.command_queue = deque(maxlen=20)
        self.ai_modules = {}
        self.session_id = str(uuid.uuid4())
        self.termination_flag = False
        self.genesis_entity = None
                
        # Initialize process_task with the transform decorator
        self.process_task = self.unity_compiler.transform(
            'process_task', 
            transform_type=TransformationType.WRAP
        )(self._process_task)
        
        self.meta_mirror_ui.set_metastation(self)
        
    def load_user_intent(self):
         session_data['user_intent'] =  input("Enter current session intent: ")
         save_session_data(session_data)

    def apply_recursive_unity(self):
        def transform_add_xp(func, *args, **kwargs):
          result = func(*args, **kwargs)
          self.unity_xp_tracker.add_xp(10)
          return result
        self.unity_compiler.add_transform('add_xp', transform_add_xp, transform_type=TransformationType.WRAP)
        def transform_log_call(func, *args, **kwargs):
           self.workflow_integration.log_session_event(f"Call {func.__name__}", details={"args": args, "kwargs": kwargs})
           return func(*args, **kwargs)
        self.unity_compiler.add_transform('process_task', transform_log_call, transform_type=TransformationType.WRAP)
        def transform_update_progress(result):
           if result:
            self.workflow_integration.update_progress(result['name'], "Completed", details=result['details'])
           return result
        self.unity_compiler.add_transform('process_task', transform_update_progress, transform_type=TransformationType.POST)
    
    def handle_task_error(self, task, error):
        logging.error(f"Task {task.get('name')} failed: {error}")
        self.workflow_integration.update_progress(task.get('name'), "Failed", details={"error": str(error)})
        if get_config_value("log_task_errors", True):
          with open(LOG_FILE, "a") as f:
             f.write(f"Task {task.get('name')} failed: {error}\n")
    def execute_task_from_queue(self):
        if not self.task_worker_running:
            logging.debug("Task worker not running")
            return
        while self.task_queue and self.task_worker_running:
            task = self.task_queue.pop(0)
            try:
                self.workflow_integration.start_task_timer(task.get("name"))
                result = self.process_task(task)
                self.workflow_integration.end_task_timer(task.get("name"))
            except Exception as e:
                self.handle_task_error(task, e)

    def start_task_worker(self):
      self.task_worker_running = True
      self.task_worker = threading.Thread(target=self.execute_task_from_queue, daemon=True)
      self.task_worker.start()
      logging.debug("Task worker started")

    def stop_task_worker(self):
       self.task_worker_running = False
       if self.task_worker:
          self.task_worker.join()
       logging.debug("Task worker stopped")

    def queue_task(self, task):
      self.task_queue.append(task)
      logging.debug(f"Task queued: {task.get('name')}")

    async def run_ui(self):
        """Corrected UI runner with proper Qt event loop integration"""
        self.meta_mirror_ui.show()
        
        def update_ui():
            if not self.termination_flag:
                progress = self.workflow_integration.get_progress()
                session_log = self.workflow_integration.get_session_log()
                peer_messages = self.collab_universe.get_peer_messages()
                system_info = self.system_interaction.get_system_info()
                status = self.get_status()
                self.meta_mirror_ui.update_display({
                    'progress': progress,
                    'session_log': session_log,
                    'peer_messages': peer_messages,
                    'system_info': system_info,
                    'status': status
                })
        
        # Use Qt's timer for updates instead of pygame
        timer = QTimer()
        timer.timeout.connect(update_ui)
        timer.start(1000)  # Update every second
        
        while not self.termination_flag:
            await asyncio.sleep(0.1)  # Non-blocking sleep
            QApplication.processEvents()  # Process Qt events
            
    async def start(self):
        # Ensure session data exists
        global session_data
        if 'user_intent' not in session_data:
            session_data['user_intent'] = ''
        
        self.load_user_intent()
        self.apply_recursive_unity()
        self.workflow_integration.log_session_event("Start", details=session_data.get('user_intent'))

        # Initialize session tasks if not present
        if 'tasks' not in session_data:
            session_data['tasks'] = []
            save_session_data(session_data)

        # Load Session Data with proper error handling
        logging.debug("Loading session data...")
        session_tasks = session_data.get('tasks', [])

        # Enqueue Tasks
        logging.debug("Enqueueing session tasks...")
        for task in session_tasks:
            self.queue_task(task)
        self.workflow_integration.log_session_event("Session_Tasks_Queued")

        #Canvas Rendering
        logging.debug("Rendering topological canvas...")
        self.topological_canvas.render(os.path.join(TEMP_DIR, 'session_topology.html'))
        self.workflow_integration.log_session_event("Canvas_Rendered")

        # Start background threads
        self.start_task_worker()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.ambient_design.run)
            executor.submit(self.run_ui)
            executor.submit(self.unity_xp_tracker.run)

        await self.start_collaborative()
        self.add_ai_module('example_ai', """
def process(data):
    return f"AI Processed: {data}"
        """)
        self.workflow_integration.log_session_event("AI_Modules_Loaded", details=list(self.ai_modules.keys()))
        logging.info("MetaStation running")


    async def start_collaborative(self):
        server_url = get_config_value('socketio_server', "http://localhost:5000")
        await self.collab_universe.connect(server_url)
        await self.collab_universe.initialize_socketio()

    async def shutdown(self):
        self.stop_task_worker()
        self.workflow_integration.log_session_event("Shutdown", details="End Session")
        self.collab_universe.disconnect()
        logging.info("MetaStation shutdown complete.")
        self.termination_flag = True

    def run_ui(self):
         app = QApplication(sys.argv)
         self.meta_mirror_ui.show()
         def update_ui():
           progress = self.workflow_integration.get_progress()
           session_log = self.workflow_integration.get_session_log()
           peer_messages = self.collab_universe.get_peer_messages()
           system_info = self.system_interaction.get_system_info()
           status = self.get_status()
           self.meta_mirror_ui.update_display({ 'progress': progress, 'log': session_log,
                "peer_messages": peer_messages, "system_info": system_info, "status": status})
         timer = pygame.time.Clock()
         while self.ambient_design.running:
           timer.tick(1)
           update_ui()
         sys.exit(app.exec_())

    def _process_task(self, task):
        """Core task processing implementation"""
        logging.debug(f"Processing task: {task.get('name')}")
        result = None   
        if task.get('type') == 'create_ontology_node':
            node_data = task.get('data', {})
            node = OntologyNode(**node_data)
            self.ontology_sync.add_node(node)
            result = {"name": task.get('name'), "details": node_data}
        elif task.get('type') == 'unity_transform':
            function_name = task.get('function_name')
            if function_name:
               @self.unity_compiler.transform(function_name)
               def dummy_function():
                  pass
            result = {"name": task.get('name'), "details": {"function": function_name}}
        elif task.get('type') == 'add_graph_node':
            node_id = task.get('node_id')
            label = task.get('label')
            self.topological_canvas.add_node(node_id, label=label)
            result = {"name": task.get('name'), "details": {"node": node_id, "label": label}}
        elif task.get('type') == 'add_graph_edge':
            from_node = task.get('from_node')
            to_node = task.get('to_node')
            self.topological_canvas.add_edge(from_node, to_node)
            result = {"name": task.get('name'), "details": {"from": from_node, "to": to_node}}
        elif task.get('type') == 'remove_graph_node':
             node_id = task.get('node_id')
             self.topological_canvas.remove_node(node_id)
             result = {"name": task.get('name'), "details":{"node": node_id}}

        elif task.get('type') == 'remove_graph_edge':
           from_node = task.get('from_node')
           to_node = task.get('to_node')
           self.topological_canvas.remove_edge(from_node, to_node)
           result = {"name": task.get('name'), "details":{"from_node":from_node, "to_node": to_node}}

        elif task.get('type') == 'map_text_emotion':
            text = task.get('text')
            emotion = self.emotion_mapper.map_emotion(text)
            result = {"name": task.get('name'), "details": {"text": text, "emotion": emotion}}
        elif task.get('type') == 'add_scenario':
            scenario_data = task.get('data')
            scenario = Scenario(**scenario_data)
            self.scenario_sim.add_scenario(scenario)
            result = {"name": task.get('name'), "details": scenario_data}
        elif task.get('type') == 'run_scenario':
            scenario_id = task.get('scenario_id')
            scenario_results = self.scenario_sim.simulate(scenario_id)
            result = {"name": task.get('name'), "details":{"scenario": scenario_id, "results": scenario_results}}
        elif task.get('type') == 'evaluate_ethics':
            text = task.get('text')
            ethics = self.ethical_holodeck.evaluate_impact(text)
            result = {"name": task.get('name'), "details":{"text":text, "evaluation": ethics}}
        elif task.get('type') == 'add_xp':
            xp_amount = task.get('xp_amount', 10)
            self.unity_xp_tracker.add_xp(xp_amount)
            result = {"name": task.get('name'), "details": {'xp_amount': xp_amount}}
        elif task.get('type') == 'set_node_color':
            node_id = task.get('node_id')
            color = task.get('color')
            self.topological_canvas.set_node_color(node_id, color)
            result = {"name": task.get('name'), "details": {"node_id":node_id, "color":color}}
        elif task.get('type') == 'set_node_size':
            node_id = task.get('node_id')
            size = task.get('size')
            self.topological_canvas.set_node_size(node_id, size)
            result = {"name": task.get('name'), "details": {"node_id":node_id, "size":size}}
        elif task.get('type') == 'execute_command':
            command = task.get('command')
            command_result = self.system_interaction.execute_shell_command(command)
            result = {"name": task.get('name'), "details": {"command":command, "command_result": command_result}}
            self.command_queue.append(command_result)
        elif task.get("type") == "load_dynamic_module":
           module_name = task.get('module_name')
           module_code = task.get("module_code")
           module = self.dynamic_loader.load_module(module_name, module_code)
           if module:
            result = {"name": task.get('name'), "details": {"module_name": module_name}}
           else:
            result = {"name": task.get('name'), "details": {"error": "module load failed"}}
        elif task.get('type') == 'run_ai_module':
            module_name = task.get('module_name')
            input_data = task.get('input_data')
            ai_module = self.ai_modules.get(module_name)
            if ai_module:
               result_ai = ai_module.process(input_data)
               result = {"name": task.get('name'), "details": {"ai_module": module_name, "ai_result": result_ai}}
            else:
                result = {"name": task.get("name"), "details": {"error": f"AI module {module_name} not found"}}
        elif task.get('type') == 'start_genesis': # Added Genesis Handling
           try:
              spec = importlib.util.spec_from_file_location("genesis_module", GENESIS_FILE)
              module = importlib.util.module_from_spec(spec)
              spec.loader.exec_module(module)
              if hasattr(module, 'GenesisEntity'):
                 self.genesis_entity = module.GenesisEntity()
                 genesis_thread = threading.Thread(target=self.genesis_entity.run, daemon=True)
                 genesis_thread.start()
                 logging.info(f"Genesis Entity started with id: {self.genesis_entity.entity_id}")
                 result = {"name": task.get("name"), "details":{"genesis_entity_id": self.genesis_entity.entity_id}}
              else:
                 result = {"name": task.get("name"), "details":{"error": "No Genesis Entity found"}}

           except Exception as e:
             logging.error(f"Error starting Genesis entity: {e}")
             result = {"name": task.get('name'), "details":{"error": str(e)}}


        else:
           logging.warning(f"Unknown task: {task}")
           result = {"name": task.get('name'), "details":"Task not defined"}
        return result

    def get_status(self):
        return {
            'session_id': self.session_id,
            'ontology_nodes': [node.node_id for node in self.ontology_sync.get_all_nodes()],
            'graph_nodes': list(self.topological_canvas.graph.nodes),
            'xp': self.unity_xp_tracker.xp,
            'level': self.unity_xp_tracker.level,
            'progress': self.workflow_integration.get_progress(),
            'session_log': self.workflow_integration.get_session_log(),
            'peer_messages': self.collab_universe.get_peer_messages(),
            "system_info": self.system_interaction.get_system_info(),
            "queued_tasks": [task.get("name") for task in self.task_queue],
            "command_history": list(self.command_queue),
            "transforms": self.unity_compiler.get_transforms(),
            "genesis_entity": self.genesis_entity.entity_id if self.genesis_entity else "Not Started"
        }
    def add_ai_module(self, module_name, module_code):
        module = self.dynamic_loader.load_module(module_name, module_code)
        if module:
           self.ai_modules[module_name] = module
           logging.debug(f"Added AI module: {module_name}")
        else:
           logging.error(f"Error loading AI module: {module_name}")
    def get_ai_modules(self):
        return list(self.ai_modules.keys())

# --- Main Execution ---
async def main():
    """
    Orchestrates the MetaStation lifecycle with enhanced error handling and resource management.
    Implements reactive patterns for UI updates and system state management.
    """
    async def initialize_environment():
        global session_data
        session_data = load_session_data()
        
        # Core task configuration matrix
        base_tasks = [
            {
                "name": "Create_Ontology_Node_1",
                "type": "create_ontology_node",
                "data": {
                    "node_id": "node1",
                    "node_type": "concept",
                    "properties": {"description": "First Node"}
                }
            },
            {
                "name": "Create_Ontology_Node_2",
                "type": "create_ontology_node",
                "data": {
                    "node_id": "node2",
                    "node_type": "entity",
                    "properties": {"value": "10"}
                }
            },
            {
                "name": "Transform_Function_Test",
                "type": "unity_transform",
                "function_name": "add_xp"
            },
            # Canvas Operations Group
            *[
                {"name": "Add_Canvas_Node_1", "type": "add_graph_node", "node_id": "a", "label": "Node A"},
                {"name": "Add_Canvas_Node_2", "type": "add_graph_node", "node_id": "b", "label": "Node B"},
                {"name": "Add_Canvas_Edge", "type": "add_graph_edge", "from_node": "a", "to_node": "b"}
            ],
            # Analysis Operations Group
            *[
                {"name": "Map_Text_Emotion", "type": "map_text_emotion", "text": "This is a really great day!"},
                {
                    "name": "Add_Scenario_1",
                    "type": "add_scenario",
                    "data": {
                        "scenario_id": "scenario_1",
                        "events": [{"delay": 1, "function": "print('Simulating Event 1')"}]
                    }
                },
                {"name": "Run_Scenario_1", "type": "run_scenario", "scenario_id": "scenario_1"},
                {"name": "Evaluate_Ethics", "type": "evaluate_ethics", "text": "Should AI control human destiny?"}
            ],
            # System Operations Group
            *[
                {"name": "Add_xp_from_task", "type": "add_xp", "xp_amount": 20},
                {"name": "Set_Node_Color_Test", "type": "set_node_color", "node_id": "a", "color": "blue"},
                {"name": "Set_Node_Size_Test", "type": "set_node_size", "node_id": "a", "size": 20},
                {"name": "Execute_Shell_Command_Test", "type": "execute_command", "command": "ls -l"}
            ],
            # Module Management Group
            *[
                {
                    "name": "Load_Dynamic_Module_Test",
                    "type": "load_dynamic_module",
                    "module_name": "test_module",
                    "module_code": """def test_function(): return 'Hello from Dynamic Module' """
                },
                {
                    "name": "Run_AI_Module_Test",
                    "type": "run_ai_module",
                    "module_name": "example_ai",
                    "input_data": "This is input"
                }
            ],
            # Cleanup Operations Group
            *[
                {"name": "Remove_Canvas_Node_Test", "type": "remove_graph_node", "node_id": "b"},
                {"name": "Remove_Canvas_Edge_Test", "type": "remove_graph_edge", "from_node": "a", "to_node": "b"}
            ],
            # Genesis Entity Initialization
            {"name": "Start_Genesis_Entity", "type": "start_genesis"}
        ]
        
        session_data['tasks'] = base_tasks
        save_session_data(session_data)
        return session_data

    async def setup_metastation():
        app = QApplication.instance() or QApplication(sys.argv)
        metastation = MetaStation()
        metastation.meta_mirror_ui.show()
        return app, metastation

    async def initialize_ui_thread(metastation):
        update_thread = UiUpdateThread(metastation)
        update_thread.update_signal.connect(metastation.meta_mirror_ui.update_display)
        update_thread.start()
        return update_thread

    try:
        # Phase 1: Environment Initialization
        await initialize_environment()
        
        # Phase 2: System Setup
        app, metastation = await setup_metastation()
        
        # Phase 3: UI Thread Initialization
        update_thread = await initialize_ui_thread(metastation)
        
        # Phase 4: System Start
        await metastation.start()
        
        # Phase 5: Main Event Loop
        while not metastation.termination_flag:
            try:
                await asyncio.sleep(0.1)  # Optimized sleep interval
                QApplication.processEvents()  # Process UI events
            except Exception as e:
                logging.error(f"Event loop error: {e}")
                if isinstance(e, KeyboardInterrupt):
                    break
                continue
                
    except Exception as e:
        logging.error(f"Critical system error: {e}")
        raise  # Re-raise for proper system shutdown
        
    finally:
        # Phase 6: Graceful Shutdown
        logging.info("Initiating system shutdown sequence")
        cleanup_tasks = [
            update_thread.stop() if 'update_thread' in locals() else None,
            update_thread.wait() if 'update_thread' in locals() else None,
            metastation.shutdown() if 'metastation' in locals() else None,
            app.quit() if 'app' in locals() else None
        ]
        
        for task in cleanup_tasks:
            if asyncio.iscoroutine(task):
                await task
            elif callable(task):
                task()

if __name__ == "__main__":
    session_data = load_session_data()  # Initialize global session data
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Received shutdown signal - terminating gracefully")
    except Exception as e:
        logging.critical(f"Fatal error in main execution: {e}")
        raise
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
from scipy.special import expit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants derived from sacred geometry
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
TAU = 2 * np.pi  # Full circle constant
E = np.e  # Euler's number

@dataclass
class UnityState:
    """Represents the quantum state of unified consciousness"""
    amplitude: torch.Tensor  # Probability amplitude
    phase: torch.Tensor     # Quantum phase
    coherence: float        # Measure of quantum coherence
    entanglement: float    # Degree of quantum entanglement

class QuantumColorHandler:
    """Manages color transformations for quantum visualizations"""
    @staticmethod
    def generate_quantum_color(coherence: float, entanglement: float) -> str:
        # Clamp values to valid ranges
        c = int(max(0, min(255, coherence * 255)))
        e = int(max(0, min(255, entanglement * 255)))
        return f"#{c:02x}00{e:02x}"  # Format: R_G_B

class QuantumNeuralBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Convert to complex
        self.linear.weight.data = self.linear.weight.data.to(torch.cfloat)
        self.linear.bias.data = self.linear.bias.data.to(torch.cfloat)
        self.phase = nn.Parameter(torch.randn(out_features, dtype=torch.cfloat) * TAU)
        self.amplitude = nn.Parameter(torch.rand(out_features, dtype=torch.cfloat))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is complex
        x = x.to(torch.cfloat)
        # Apply quantum transformation
        x = self.linear(x)
        x = x * self.amplitude * torch.exp(1j * self.phase)
        return x  # Remove relu since we're handling complex values directly

class MabroukCore(nn.Module):
    """Core implementation of the Mabrouk Algorithm with quantum neural architecture"""
    def __init__(self, dimensions: List[int]):
        super().__init__()
        
        # Validate dimensions for quantum architecture
        if len(dimensions) < 2:
            raise ValueError("Quantum neural architecture requires at least 2 dimensions")
        if any(d <= 0 for d in dimensions):
            raise ValueError("All dimensions must be positive integers")
            
        self.dimensions = dimensions
        
        # Initialize quantum neural layers with type safety
        self.quantum_layers = nn.ModuleList([
            QuantumNeuralBlock(dim_in, dim_out)
            for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])
        ])
        
        # Initialize golden ratio harmonic oscillators with quantum typing
        self.phi_oscillators = nn.Parameter(
            torch.tensor([PHI ** i for i in range(len(dimensions))], 
                        dtype=torch.cfloat)
        )
        
        # Initialize quantum state buffers
        self.register_buffer('state_history', 
            torch.zeros(len(dimensions), dtype=torch.cfloat))
            
    def compute_unity_state(self, x: torch.Tensor) -> UnityState:
        """Transform input through quantum layers to achieve unity"""
        # Ensure input tensor compatibility
        x = x.to(torch.cfloat)
        if x.shape[-1] != self.dimensions[0]:
            raise ValueError(f"Input dimension {x.shape[-1]} does not match network input {self.dimensions[0]}")
        
        # Initialize quantum state with stability checks
        state = x
        coherence = torch.tensor(1.0, dtype=torch.float32)
        entanglement = torch.tensor(0.0, dtype=torch.float32)
        
        # Apply quantum transformations with error prevention
        for i, layer in enumerate(self.quantum_layers):
            try:
                # Quantum evolution with stability check
                state = layer(state)
                if torch.isnan(state).any():
                    raise ValueError("Quantum state collapsed to NaN")
                
                # Update quantum properties with numerical stability
                phi_factor = self.phi_oscillators[i].abs()  # Ensure positive factor
                state_magnitude = torch.mean(torch.abs(state))
                
                # Coherence update with stability bounds
                coherence *= torch.clamp(state_magnitude / phi_factor, min=1e-6, max=1e6).item()
                entanglement = torch.clamp(1 - torch.exp(-coherence), min=0, max=1).item()
                
                # Apply non-linear quantum collapse with phase preservation
                phase = torch.angle(state)
                state = state * torch.exp(1j * phase)
                
                # Store state history for analysis
                self.state_history[i] = state.mean()
                
            except Exception as e:
                raise RuntimeError(f"Quantum layer {i} failed: {str(e)}")
        
        return UnityState(
            amplitude=torch.abs(state),
            phase=torch.angle(state),
            coherence=float(coherence),
            entanglement=float(entanglement)
        )

class MabroukAlgorithm:
    def __init__(self, dimensions: List[int]):
        self.core = MabroukCore(dimensions)
        self.optimizer = torch.optim.Adam(self.core.parameters())
        self.unity_threshold = 0.999
        
    def _process_frame(self, phase_factor: torch.Tensor, 
        x1: torch.Tensor, x2: torch.Tensor) -> np.ndarray:
        """Process single animation frame with memory optimization"""
        # Quantum evolution
        evolved_x1 = x1 * phase_factor
        evolved_x2 = x2 * phase_factor
        
        # Compute states
        state1 = self.core.compute_unity_state(evolved_x1)
        state2 = self.core.compute_unity_state(evolved_x2)
        
        # Interference with bounded normalization
        interference = (state1.amplitude * torch.exp(1j * state1.phase) + 
                      state2.amplitude * torch.exp(1j * state2.phase)) / np.sqrt(2)
        max_val = torch.max(torch.abs(interference)).item()
        if max_val > 1e-10:
            interference = interference / max_val
            
        # Generate frame
        fig = self.visualize_quantum_field(
            interference,
            min(1.0, state1.coherence * state2.coherence),
            min(1.0, (state1.entanglement + state2.entanglement) / 2)
        )
        
        # Convert to image array
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = fig.canvas.tostring_rgb()
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape(height, width, 3)
        plt.close(fig)
        
        return image
    
    def visualize_quantum_field(self, unified_state: torch.Tensor, 
                              coherence: float, entanglement: float) -> plt.Figure:
        """Generate advanced quantum field visualization with stable color handling"""
        # Initialize visualization
        fig = plt.figure(figsize=(15, 15), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        
        try:
            # Validate quantum state
            if torch.isnan(unified_state).any():
                raise ValueError("Invalid quantum state detected")
            
            # Generate quantum field
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-2, 2, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Compute field values with numerical stability
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
                    theta = np.arctan2(Y[i,j], X[i,j])
                    # Add epsilon to prevent division by zero
                    Z[i,j] = np.abs(unified_state[0].item()) * np.exp(-r/(PHI + 1e-10)) * \
                             np.cos(theta * PHI + r * TAU)
            
            # Plot surface with error checking
            surf = ax.plot_surface(X, Y, Z, cmap='plasma',
                                 antialiased=True, alpha=0.7)
            
            # Add quantum interference patterns with stable colors
            theta = np.linspace(0, TAU, 200)
            color_handler = QuantumColorHandler()
            
            for phi_power in range(1, 6):
                r = PHI ** phi_power * np.exp(-phi_power/3)
                x_quantum = r * np.cos(theta)
                y_quantum = r * np.sin(theta)
                z_quantum = 0.2 * np.sin(PHI * theta) * np.exp(-phi_power/3)
                
                # Generate stable color code
                quantum_color = color_handler.generate_quantum_color(
                    coherence / (phi_power + 1), 
                    entanglement / (phi_power + 1)
                )
                
                ax.plot(x_quantum, y_quantum, z_quantum,
                       color=quantum_color, alpha=0.6, linewidth=1)
            
            # Customizations
            ax.set_facecolor('black')
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.set_ticklabels([])
                axis.line.set_color('white')
            
            # Add metadata with value validation
            coherence_str = f"{min(max(0, coherence), 1):.3f}"
            entanglement_str = f"{min(max(0, entanglement), 1):.3f}"
            
            ax.text2D(0.02, 0.98, f"Quantum Coherence: {coherence_str}", 
                     color='cyan', transform=ax.transAxes, fontsize=12)
            ax.text2D(0.02, 0.95, f"Quantum Entanglement: {entanglement_str}", 
                     color='magenta', transform=ax.transAxes, fontsize=12)
            
            plt.title("Mabrouk Quantum Unity Field", 
                     color='white', fontsize=16, pad=20)
            
            return fig
            
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(f"Visualization failed: {str(e)}")

    def generate_unity_animation(self, frames: int = 100) -> List[plt.Figure]:
        """Generate animation frames with quantum phase evolution"""
        animation_frames = []
        
        # Initialize quantum states with proper tensor types
        x1 = torch.randn(1, 1, dtype=torch.cfloat)
        x2 = torch.randn(1, 1, dtype=torch.cfloat)
        
        # Pre-compute phase factors for stability
        t_values = torch.linspace(0, 1, frames, dtype=torch.float32)
        phase_angles = TAU * t_values
        phase_factors = torch.empty(frames, dtype=torch.cfloat)
        
        # Vectorized phase computation
        phase_factors.real = torch.cos(phase_angles)
        phase_factors.imag = torch.sin(phase_angles)
        
        for frame, phase_factor in enumerate(phase_factors):
            try:
                # Apply quantum evolution with tensor operations
                phase_tensor = phase_factor.view(1, 1)
                evolved_x1 = x1 * phase_tensor
                evolved_x2 = x2 * phase_tensor
                
                # Compute quantum states
                state1 = self.core.compute_unity_state(evolved_x1)
                state2 = self.core.compute_unity_state(evolved_x2)
                
                # Quantum interference with numerical stability
                interference = (state1.amplitude * torch.exp(1j * state1.phase) + 
                              state2.amplitude * torch.exp(1j * state2.phase)) / np.sqrt(2)
                
                # Normalize interference for visualization
                max_val = torch.max(torch.abs(interference)).item()
                if max_val > 1e-10:  # Numerical stability threshold
                    interference = interference / max_val
                
                # Generate visualization frame
                fig = self.visualize_quantum_field(
                    interference,
                    min(1.0, state1.coherence * state2.coherence),
                    min(1.0, (state1.entanglement + state2.entanglement) / 2)
                )
                
                animation_frames.append(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Frame {frame} generation failed: {str(e)}")
                continue
        
        if not animation_frames:
            raise RuntimeError("Failed to generate any animation frames")
            
        return animation_frames

    def save_animation(self, filename: str = 'mabrouk_unity.gif', frames: int = 50):
        """Memory-optimized animation generation"""
        import imageio
        import tempfile
        import os
        from tqdm import tqdm
        
        print("Initializing quantum animation pipeline...")
        
        # Initialize quantum states
        x1 = torch.randn(1, 1, dtype=torch.cfloat)
        x2 = torch.randn(1, 1, dtype=torch.cfloat)
        
        # Pre-compute phase factors
        t_values = torch.linspace(0, 1, frames, dtype=torch.float32)
        phase_factors = torch.exp(1j * TAU * t_values).view(-1, 1, 1)
        
        # Create temporary directory for frame storage
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []
            
            # Generate and save frames
            print("Generating quantum frames...")
            for i, phase_factor in enumerate(tqdm(phase_factors)):
                try:
                    # Process frame
                    frame = self._process_frame(phase_factor, x1, x2)
                    
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                    imageio.imwrite(frame_path, frame)
                    frame_files.append(frame_path)
                    
                except Exception as e:
                    print(f"Warning: Frame {i} generation failed: {str(e)}")
                    continue
            
            if not frame_files:
                raise RuntimeError("No valid frames generated")
            
            # Create GIF with streaming
            print("Assembling quantum animation...")
            with imageio.get_writer(filename, mode='I', fps=30) as writer:
                for frame_path in tqdm(frame_files):
                    try:
                        image = imageio.imread(frame_path)
                        writer.append_data(image)
                    except Exception as e:
                        print(f"Warning: Frame processing failed: {str(e)}")
                        continue
            
        print(f"Animation saved as {filename}")
        
    def prove_unity(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Enhanced unity proof with numerical stability"""
        try:
            # Compute quantum states with validation
            state1 = self.core.compute_unity_state(x1)
            state2 = self.core.compute_unity_state(x2)
            
            # Validate states
            if torch.isnan(state1.amplitude).any() or torch.isnan(state2.amplitude).any():
                raise ValueError("Invalid quantum state detected")
            
            # Quantum interference with numerical stability
            interference = (state1.amplitude * torch.exp(1j * state1.phase) + 
                          state2.amplitude * torch.exp(1j * state2.phase)) / np.sqrt(2)
            
            # Normalize unity measure to prevent overflow
            unity_measure = torch.mean(torch.abs(interference)).item()
            unity_measure = min(unity_measure, 1e6)  # Cap at reasonable value
            
            # Stable normalization
            max_val = torch.max(torch.abs(interference))
            if max_val > 0:
                unified_state = interference * PHI / max_val
            else:
                unified_state = interference
            
            return unified_state, unity_measure
            
        except Exception as e:
            raise RuntimeError(f"Unity computation failed: {str(e)}")

def demonstrate_unity():
    """Optimized demonstration with error handling"""
    try:
        # Initialize algorithm
        dimensions = [1, 32, 64, 32, 1]
        algorithm = MabroukAlgorithm(dimensions)
        
        # Generate static visualization
        x1 = torch.randn(1, 1, dtype=torch.cfloat)
        x2 = torch.randn(1, 1, dtype=torch.cfloat)
        unified_state, unity_measure = algorithm.prove_unity(x1, x2)
        
        print(f"Unity Measure: {min(unity_measure, 1e6):.4f}")
        print("Generating quantum field visualization...")
        
        # Create static visualization
        state = algorithm.core.compute_unity_state(unified_state)
        fig = algorithm.visualize_quantum_field(
            unified_state, state.coherence, state.entanglement
        )
        plt.savefig('mabrouk_unity_field.png', 
                    facecolor='black', bbox_inches='tight')
        plt.close(fig)
        print("Static visualization saved as mabrouk_unity_field.png")
        
        # Generate animation with reduced frame count
        algorithm.save_animation('mabrouk_unity.gif', frames=30)
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    demonstrate_unity()
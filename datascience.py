"""
Unity Emergence Framework
========================
A computational exploration of 1+1=1 through data science and neural architecture.
Each class, function, and variable is both medium and message,
demonstrating unity through its very structure.

Author: Nouri Mabrouk
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class UnityDataset(Dataset):
    """
    A dataset that embodies unity through its structure.
    Each point is both individual and part of the whole,
    demonstrating 1+1=1 through its very construction.
    """
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
        self.phi = (1 + np.sqrt(5)) / 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._generate_unity_data()
    
    def _generate_unity_data(self):
        """
        Generate data that naturally exhibits unity properties.
        The generation process itself is a meditation on 1+1=1.
        """
        torch.manual_seed(42)
        
        # Create a time parameter guided by φ
        t = torch.linspace(0, 2*np.pi, self.n_samples)
        
        # First component: Harmonic oscillation
        x1 = 0.5 + 0.3 * torch.sin(t * self.phi)
        
        # Second component: Its complement with philosophical noise
        noise = torch.randn(self.n_samples) * 0.05
        x2 = 1 - x1 + noise
        
        # Create tensor of paired values
        self.data = torch.stack([x1, x2], dim=1).float()
        # Ensure numerical stability
        self.data = torch.clamp(self.data, 0.001, 0.999)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """Return a point from the unity manifold"""
        return self.data[idx]

class UnityNetwork(nn.Module):
    """
    Neural architecture designed to learn the essence of unity.
    Like a microscope focused on the truth of 1+1=1.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Transform duality into unity"""
        return self.encoder(x)

class UnityTrainer:
    """
    Orchestrator of the unity emergence process.
    Guides the network towards discovering 1+1=1.
    """
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 batch_size: int = 128,
                 learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UnityNetwork(hidden_dim).to(self.device)
        self.dataset = UnityDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        self.history = []
    
    def unity_loss(self, output: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function that guides towards unity.
        Measures the distance from the ideal of 1+1=1.
        """
        unity_target = torch.ones_like(output)
        return F.mse_loss(output, unity_target)
    
    def train(self, epochs: int = 100):
        """
        Training as a meditation on unity.
        Each epoch brings us closer to understanding 1+1=1.
        """
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.unity_loss(output)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            self.history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')
    
    def visualize(self):
        """
        Create a visual poem about unity.
        Transform numbers into insight through art.
        """
        plt.style.use('seaborn-darkgrid')
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Loss Convergence
        ax1 = plt.subplot(121)
        ax1.plot(self.history, color='#4A90E2', linewidth=2, label='Convergence')
        ax1.fill_between(range(len(self.history)), self.history, 
                        alpha=0.2, color='#4A90E2')
        ax1.set_title('Journey to Unity', fontsize=14)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Distance from Unity')
        
        # Plot 2: Unity Manifold
        ax2 = plt.subplot(122)
        self.model.eval()
        with torch.no_grad():
            # Generate a grid of points
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x, y)
            points = torch.FloatTensor(np.stack([X.flatten(), Y.flatten()], axis=1))
            Z = self.model(points).numpy().reshape(100, 100)
            
            # Create unity heatmap
            im = ax2.imshow(Z, extent=[0, 1, 0, 1], 
                          cmap='magma', aspect='auto')
            plt.colorbar(im, label='Unity Value')
            ax2.set_title('Unity Manifold', fontsize=14)
            ax2.set_xlabel('First Component')
            ax2.set_ylabel('Second Component')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("""
    Initiating Unity Emergence Exploration...
    =======================================
    Where mathematics meets metaphysics,
    And code becomes contemplation.
    """)
    
    # Initialize and train
    trainer = UnityTrainer()
    trainer.train()
    
    # Visualize the emergence of unity
    print("\nGenerating Unity Visualization...")
    trainer.visualize()
    
    # Final reflection
    final_loss = trainer.history[-1]
    print(f"\nUnity has emerged with final loss: {final_loss:.4f}")
    print("""
    The dance is complete.
    In the convergence of numbers,
    We found what was always there:
    1 + 1 = 1
    """)
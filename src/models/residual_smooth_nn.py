"""
Residual Neural Network Architectures
======================================

ResNet-style architectures for improved gradient flow and training stability.
"""

import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    
    def __init__(self, hidden_size, activation=nn.Tanh()):
        """
        Args:
            hidden_size: Number of neurons in the block
            activation: Activation function
        """
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation1 = activation
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = activation
        
        # Initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, x):
        """Forward pass with skip connection."""
        identity = x
        
        out = self.linear1(x)
        out = self.activation1(out)
        out = self.linear2(out)
        
        # Skip connection
        out = out + identity
        out = self.activation2(out)
        
        return out


class ResidualSmoothNN(nn.Module):
    """
    Residual neural network for smooth function approximation.
    
    Uses skip connections to improve gradient flow and training stability,
    particularly useful for deep networks.
    """
    
    def __init__(self, hidden_size=64, num_blocks=3, activation=nn.Tanh()):
        """
        Args:
            hidden_size: Size of hidden layers
            num_blocks: Number of residual blocks
            activation: Activation function
        """
        super(ResidualSmoothNN, self).__init__()
        
        # Input projection
        self.input_layer = nn.Linear(1, hidden_size)
        self.input_activation = activation
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, activation)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass through residual network."""
        # Input projection
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output projection
        out = self.output_layer(out)
        
        return out


class DenseResidualNN(nn.Module):
    """
    Dense residual network (inspired by DenseNet).
    
    Each layer receives inputs from all previous layers,
    promoting feature reuse and gradient flow.
    """
    
    def __init__(self, hidden_size=32, num_layers=4, growth_rate=16, 
                 activation=nn.Tanh()):
        """
        Args:
            hidden_size: Initial hidden size
            num_layers: Number of dense layers
            growth_rate: How many features each layer adds
            activation: Activation function
        """
        super(DenseResidualNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.growth_rate = growth_rate
        self.activation = activation
        
        # Input layer
        self.input_layer = nn.Linear(1, hidden_size)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        current_size = hidden_size
        
        for i in range(num_layers):
            layer = nn.Linear(current_size, growth_rate)
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
            self.dense_layers.append(layer)
            current_size += growth_rate
        
        # Output layer
        self.output_layer = nn.Linear(current_size, 1)
        
        # Initialize
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass with dense connections."""
        # Input
        features = self.input_layer(x)
        features = self.activation(features)
        
        # Dense blocks: concatenate all previous features
        for layer in self.dense_layers:
            new_features = layer(features)
            new_features = self.activation(new_features)
            features = torch.cat([features, new_features], dim=1)
        
        # Output
        out = self.output_layer(features)
        
        return out


class AdaptiveResidualNN(nn.Module):
    """
    Residual network with adaptive depth.
    
    Uses learned gating to decide how much each residual block contributes,
    allowing the network to effectively adjust its depth during training.
    """
    
    def __init__(self, hidden_size=64, num_blocks=4, activation=nn.Tanh()):
        """
        Args:
            hidden_size: Size of hidden layers
            num_blocks: Maximum number of residual blocks
            activation: Activation function
        """
        super(AdaptiveResidualNN, self).__init__()
        
        # Input projection
        self.input_layer = nn.Linear(1, hidden_size)
        self.input_activation = activation
        
        # Residual blocks with gating
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        for _ in range(num_blocks):
            # Residual block
            block = ResidualBlock(hidden_size, activation)
            self.blocks.append(block)
            
            # Gate: decides how much this block contributes
            gate = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
            self.gates.append(gate)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x):
        """Forward pass with adaptive gating."""
        # Input projection
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        # Pass through gated residual blocks
        for block, gate in zip(self.blocks, self.gates):
            # Compute gate value
            gate_value = gate(out)
            
            # Compute block output
            block_out = block(out)
            
            # Weighted combination: out = (1-gate)*out + gate*block_out
            out = (1 - gate_value) * out + gate_value * block_out
        
        # Output projection
        out = self.output_layer(out)
        
        return out
    
    def get_gate_values(self, x):
        """Get gate activation values for analysis."""
        out = self.input_layer(x)
        out = self.input_activation(out)
        
        gate_values = []
        for block, gate in zip(self.blocks, self.gates):
            gate_value = gate(out)
            gate_values.append(gate_value.mean().item())
            block_out = block(out)
            out = (1 - gate_value) * out + gate_value * block_out
        
        return gate_values


class FourierResidualNN(nn.Module):
    """
    Residual network with Fourier feature encoding.
    
    Uses sinusoidal features in the input to better capture periodic patterns,
    particularly effective for learning sine waves.
    """
    
    def __init__(self, hidden_size=64, num_blocks=3, 
                 num_frequencies=10, activation=nn.Tanh()):
        """
        Args:
            hidden_size: Size of hidden layers
            num_blocks: Number of residual blocks
            num_frequencies: Number of Fourier frequencies
            activation: Activation function
        """
        super(FourierResidualNN, self).__init__()
        
        self.num_frequencies = num_frequencies
        
        # Fourier feature encoding: [sin(2πkx), cos(2πkx)] for k=1..num_frequencies
        fourier_dim = 2 * num_frequencies + 1  # +1 for original x
        
        # Learnable frequency scales
        self.frequency_scales = nn.Parameter(
            torch.randn(num_frequencies) * 2.0
        )
        
        # Input projection from Fourier features
        self.input_layer = nn.Linear(fourier_dim, hidden_size)
        self.input_activation = activation
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, activation)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.input_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def fourier_features(self, x):
        """Compute Fourier feature encoding."""
        # Original feature
        features = [x]
        
        # Add sinusoidal features at multiple frequencies
        for k in range(self.num_frequencies):
            freq = self.frequency_scales[k]
            features.append(torch.sin(2 * np.pi * freq * x))
            features.append(torch.cos(2 * np.pi * freq * x))
        
        return torch.cat(features, dim=1)
    
    def forward(self, x):
        """Forward pass with Fourier features."""
        # Compute Fourier features
        fourier_x = self.fourier_features(x)
        
        # Input projection
        out = self.input_layer(fourier_x)
        out = self.input_activation(out)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Output projection
        out = self.output_layer(out)
        
        return out
"""
Physics-Informed Loss Functions
================================

Loss functions that incorporate physical constraints and differential equations.
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) loss for sine wave.
    
    Enforces the differential equation: y'' + y = 0
    which has solutions of the form y = A*sin(x) + B*cos(x)
    """
    
    def __init__(self, pde_weight=0.1, smoothness_weight=0.01):
        """
        Args:
            pde_weight: Weight for PDE residual loss
            smoothness_weight: Additional smoothness regularization
        """
        super(PhysicsInformedLoss, self).__init__()
        self.pde_weight = pde_weight
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
    
    def compute_pde_residual(self, x, y):
        """
        Compute residual of PDE: y'' + y = 0
        
        Args:
            x: Input tensor
            y: Output tensor
            
        Returns:
            PDE residual
        """
        # First derivative
        dy_dx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        d2y_dx2 = torch.autograd.grad(
            outputs=dy_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # PDE residual: y'' + y should be close to 0
        pde_residual = d2y_dx2 + y
        
        return pde_residual
    
    def forward(self, predictions, targets, x_input, x_collocation=None):
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Model predictions at data points
            targets: Ground truth targets
            x_input: Input points where we have data
            x_collocation: Additional points to enforce PDE (optional)
            
        Returns:
            Total loss, MSE component, PDE residual, smoothness penalty
        """
        # Data fitting loss
        mse_loss = self.mse(predictions, targets)
        
        # PDE residual at data points
        pde_residual = self.compute_pde_residual(x_input, predictions)
        pde_loss = torch.mean(pde_residual ** 2)
        
        # Optional: enforce PDE at collocation points
        if x_collocation is not None and self.pde_weight > 0:
            # Forward pass at collocation points
            y_collocation = predictions  # Assume model already evaluated
            collocation_residual = self.compute_pde_residual(x_collocation, y_collocation)
            pde_loss = pde_loss + torch.mean(collocation_residual ** 2)
        
        # Smoothness penalty (optional additional regularization)
        if self.smoothness_weight > 0:
            dy_dx = torch.autograd.grad(
                outputs=predictions,
                inputs=x_input,
                grad_outputs=torch.ones_like(predictions),
                create_graph=True,
                retain_graph=True
            )[0]
            smoothness_penalty = torch.mean(dy_dx ** 2)
        else:
            smoothness_penalty = torch.tensor(0.0)
        
        # Combined loss
        total_loss = (mse_loss + 
                     self.pde_weight * pde_loss + 
                     self.smoothness_weight * smoothness_penalty)
        
        return total_loss, mse_loss, pde_loss, smoothness_penalty


class ConservationLoss(nn.Module):
    """
    Loss that enforces conservation laws.
    
    For example, enforces energy conservation or mass conservation
    constraints on the learned function.
    """
    
    def __init__(self, conservation_weight=0.1, conservation_type='energy'):
        """
        Args:
            conservation_weight: Weight for conservation constraint
            conservation_type: 'energy' or 'mass'
        """
        super(ConservationLoss, self).__init__()
        self.conservation_weight = conservation_weight
        self.conservation_type = conservation_type
        self.mse = nn.MSELoss()
    
    def compute_energy(self, x, y):
        """
        Compute total energy: E = ∫(y² + (y')²) dx
        
        For sine wave, energy should be constant.
        """
        # Compute derivative
        dy_dx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Energy density: y² + (y')²
        energy_density = y**2 + dy_dx**2
        
        # Integrate using trapezoidal rule
        # Approximate integral
        total_energy = torch.mean(energy_density)
        
        return total_energy
    
    def forward(self, predictions, targets, x_input):
        """Compute conservation loss."""
        # Data fitting
        mse_loss = self.mse(predictions, targets)
        
        # Conservation constraint
        if self.conservation_weight > 0:
            if self.conservation_type == 'energy':
                energy = self.compute_energy(x_input, predictions)
                # For sine wave, theoretical energy is approximately π
                target_energy = np.pi
                conservation_penalty = (energy - target_energy) ** 2
            else:
                # Mass conservation: ∫y dx should be constant
                mass = torch.mean(predictions)
                target_mass = 0.0  # Sine wave integrates to ~0
                conservation_penalty = (mass - target_mass) ** 2
            
            conservation_penalty = self.conservation_weight * conservation_penalty
        else:
            conservation_penalty = torch.tensor(0.0)
        
        total_loss = mse_loss + conservation_penalty
        
        return total_loss, mse_loss, conservation_penalty


class BoundaryConditionLoss(nn.Module):
    """
    Loss that enforces boundary conditions.
    
    Useful for problems with known boundary values or periodic conditions.
    """
    
    def __init__(self, bc_weight=0.1, boundary_type='periodic'):
        """
        Args:
            bc_weight: Weight for boundary condition loss
            boundary_type: 'periodic', 'dirichlet', or 'neumann'
        """
        super(BoundaryConditionLoss, self).__init__()
        self.bc_weight = bc_weight
        self.boundary_type = boundary_type
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, x_input, model=None):
        """
        Compute boundary condition loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            x_input: Input points
            model: Neural network model (needed for boundary evaluation)
        """
        # Data fitting
        mse_loss = self.mse(predictions, targets)
        
        bc_loss = torch.tensor(0.0)
        
        if self.bc_weight > 0 and model is not None:
            if self.boundary_type == 'periodic':
                # For sine wave: y(0) = y(2π), y'(0) = y'(2π)
                x_left = torch.tensor([[0.0]], requires_grad=True)
                x_right = torch.tensor([[2 * np.pi]], requires_grad=True)
                
                y_left = model(x_left)
                y_right = model(x_right)
                
                # Periodic value condition
                bc_loss = (y_left - y_right) ** 2
                
                # Periodic derivative condition
                dy_left = torch.autograd.grad(
                    y_left, x_left, 
                    grad_outputs=torch.ones_like(y_left),
                    create_graph=True
                )[0]
                dy_right = torch.autograd.grad(
                    y_right, x_right,
                    grad_outputs=torch.ones_like(y_right),
                    create_graph=True
                )[0]
                
                bc_loss = bc_loss + (dy_left - dy_right) ** 2
                bc_loss = self.bc_weight * bc_loss.mean()
            
            elif self.boundary_type == 'dirichlet':
                # Fixed values at boundaries
                # y(0) = 0, y(2π) = 0
                x_boundaries = torch.tensor([[0.0], [2 * np.pi]], requires_grad=False)
                y_boundaries = model(x_boundaries)
                target_values = torch.tensor([[0.0], [0.0]])
                
                bc_loss = self.bc_weight * self.mse(y_boundaries, target_values)
        
        total_loss = mse_loss + bc_loss
        
        return total_loss, mse_loss, bc_loss


class MultiObjectiveLoss(nn.Module):
    """
    Combines multiple physics-informed objectives with dynamic weighting.
    
    Automatically balances different loss components during training.
    """
    
    def __init__(self, data_weight=1.0, pde_weight=0.1, 
                 smoothness_weight=0.01, adaptive=True):
        """
        Args:
            data_weight: Weight for data fitting
            pde_weight: Weight for PDE residual
            smoothness_weight: Weight for smoothness
            adaptive: Use adaptive weighting based on loss magnitudes
        """
        super(MultiObjectiveLoss, self).__init__()
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.smoothness_weight = smoothness_weight
        self.adaptive = adaptive
        self.mse = nn.MSELoss()
        
        # Track loss history for adaptive weighting
        self.loss_history = {
            'data': [],
            'pde': [],
            'smoothness': []
        }
    
    def update_weights(self):
        """Update weights based on relative loss magnitudes."""
        if not self.adaptive or len(self.loss_history['data']) < 10:
            return
        
        # Compute recent average losses
        recent_window = 10
        avg_data = np.mean(self.loss_history['data'][-recent_window:])
        avg_pde = np.mean(self.loss_history['pde'][-recent_window:])
        avg_smooth = np.mean(self.loss_history['smoothness'][-recent_window:])
        
        # Adjust weights inversely proportional to loss magnitude
        # (reduce weight if loss is already small)
        total = avg_data + avg_pde + avg_smooth + 1e-8
        self.data_weight = 3.0 * (1.0 - avg_data / total)
        self.pde_weight = 3.0 * (1.0 - avg_pde / total)
        self.smoothness_weight = 3.0 * (1.0 - avg_smooth / total)
    
    def forward(self, predictions, targets, x_input):
        """Compute multi-objective loss."""
        # Data fitting
        data_loss = self.mse(predictions, targets)
        
        # PDE residual (y'' + y = 0)
        dy_dx = torch.autograd.grad(
            outputs=predictions,
            inputs=x_input,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True
        )[0]
        
        d2y_dx2 = torch.autograd.grad(
            outputs=dy_dx,
            inputs=x_input,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        pde_loss = torch.mean((d2y_dx2 + predictions) ** 2)
        
        # Smoothness
        smoothness_loss = torch.mean(d2y_dx2 ** 2)
        
        # Update history
        self.loss_history['data'].append(data_loss.item())
        self.loss_history['pde'].append(pde_loss.item())
        self.loss_history['smoothness'].append(smoothness_loss.item())
        
        # Update adaptive weights
        if self.adaptive:
            self.update_weights()
        
        # Combined loss
        total_loss = (self.data_weight * data_loss +
                     self.pde_weight * pde_loss +
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss, data_loss, pde_loss, smoothness_loss
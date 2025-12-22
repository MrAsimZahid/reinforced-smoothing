"""
Curvature-based Loss Functions
===============================

Alternative smoothness penalties based on geometric curvature
rather than raw derivatives.
"""

import torch
import torch.nn as nn


class CurvatureLoss(nn.Module):
    """
    Loss function that penalizes curvature of the learned function.
    
    Curvature κ(x) = |y''| / (1 + (y')²)^(3/2)
    
    This is geometrically motivated and scale-invariant, making it
    more robust than raw derivative penalties.
    """
    
    def __init__(self, curvature_weight=0.01, epsilon=1e-8):
        """
        Args:
            curvature_weight: Weight for curvature penalty
            epsilon: Small constant for numerical stability
        """
        super(CurvatureLoss, self).__init__()
        self.curvature_weight = curvature_weight
        self.epsilon = epsilon
        self.mse = nn.MSELoss()
    
    def compute_curvature(self, x, y):
        """
        Compute curvature using automatic differentiation.
        
        Args:
            x: Input tensor (requires_grad=True)
            y: Output tensor
            
        Returns:
            Curvature tensor
        """
        # First derivative y'
        dy_dx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative y''
        d2y_dx2 = torch.autograd.grad(
            outputs=dy_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Curvature: |y''| / (1 + (y')^2)^(3/2)
        numerator = torch.abs(d2y_dx2)
        denominator = torch.pow(1 + dy_dx**2, 1.5) + self.epsilon
        curvature = numerator / denominator
        
        return curvature
    
    def forward(self, predictions, targets, x_input):
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            x_input: Input tensor
            
        Returns:
            Total loss, MSE component, curvature component
        """
        # Data fitting loss
        mse_loss = self.mse(predictions, targets)
        
        # Curvature penalty
        if self.curvature_weight > 0:
            curvature = self.compute_curvature(x_input, predictions)
            curvature_penalty = torch.mean(curvature ** 2)
        else:
            curvature_penalty = torch.tensor(0.0)
        
        # Combined loss
        total_loss = mse_loss + self.curvature_weight * curvature_penalty
        
        return total_loss, mse_loss, curvature_penalty


class AdaptiveCurvatureLoss(nn.Module):
    """
    Adaptive curvature loss that adjusts penalty based on local complexity.
    
    Penalizes high curvature more in regions where the function should be smooth,
    and allows higher curvature where rapid changes are expected.
    """
    
    def __init__(self, base_weight=0.01, adaptation_rate=0.1):
        """
        Args:
            base_weight: Base curvature weight
            adaptation_rate: How much to adapt based on data density
        """
        super(AdaptiveCurvatureLoss, self).__init__()
        self.base_weight = base_weight
        self.adaptation_rate = adaptation_rate
        self.mse = nn.MSELoss()
    
    def compute_local_density(self, x_input):
        """
        Estimate local data density using nearest neighbor distances.
        Higher density → more smoothness penalty.
        """
        # Compute pairwise distances
        diff = x_input.unsqueeze(1) - x_input.unsqueeze(0)
        distances = torch.abs(diff) + 1e-8
        
        # Get k nearest neighbors (k=3)
        k = min(3, len(x_input))
        knn_distances, _ = torch.topk(distances, k, dim=1, largest=False)
        
        # Local density (inverse of average distance)
        local_density = 1.0 / (knn_distances.mean(dim=1, keepdim=True) + 1e-8)
        
        # Normalize
        local_density = local_density / (local_density.max() + 1e-8)
        
        return local_density
    
    def forward(self, predictions, targets, x_input):
        """Compute adaptive curvature loss."""
        # Data fitting
        mse_loss = self.mse(predictions, targets)
        
        if self.base_weight > 0:
            # Compute curvature
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
            
            # Compute local density
            density = self.compute_local_density(x_input)
            
            # Adaptive weights: higher penalty in dense regions
            adaptive_weights = 1.0 + self.adaptation_rate * density
            
            # Weighted curvature penalty
            curvature_penalty = torch.mean(adaptive_weights * d2y_dx2 ** 2)
            curvature_penalty = self.base_weight * curvature_penalty
        else:
            curvature_penalty = torch.tensor(0.0)
        
        total_loss = mse_loss + curvature_penalty
        
        return total_loss, mse_loss, curvature_penalty


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss - penalizes the sum of absolute gradients.
    
    Commonly used in image processing, promotes piecewise smooth solutions.
    """
    
    def __init__(self, tv_weight=0.01):
        """
        Args:
            tv_weight: Weight for total variation penalty
        """
        super(TotalVariationLoss, self).__init__()
        self.tv_weight = tv_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, x_input):
        """Compute total variation loss."""
        mse_loss = self.mse(predictions, targets)
        
        if self.tv_weight > 0:
            # Compute gradient
            dy_dx = torch.autograd.grad(
                outputs=predictions,
                inputs=x_input,
                grad_outputs=torch.ones_like(predictions),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Total variation: sum of absolute gradients
            tv_penalty = torch.mean(torch.abs(dy_dx))
            tv_penalty = self.tv_weight * tv_penalty
        else:
            tv_penalty = torch.tensor(0.0)
        
        total_loss = mse_loss + tv_penalty
        
        return total_loss, mse_loss, tv_penalty
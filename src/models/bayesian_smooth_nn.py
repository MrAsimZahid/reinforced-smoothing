"""
Bayesian Neural Networks for Uncertainty Quantification
========================================================

Bayesian approaches to estimate prediction uncertainty alongside smoothness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Implements variational inference with reparameterization trick.
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            prior_std: Standard deviation of prior distribution
        """
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters: mean and log std
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize mean with Xavier
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.bias_mu, 0.0)
        
        # Initialize log std to small values
        nn.init.constant_(self.weight_logsigma, -5.0)
        nn.init.constant_(self.bias_logsigma, -5.0)
    
    def forward(self, x, sample=True):
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor
            sample: If True, sample weights; if False, use mean
        """
        if sample:
            # Sample weights using reparameterization trick
            weight_sigma = torch.exp(self.weight_logsigma)
            weight_epsilon = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_sigma * weight_epsilon
            
            # Sample bias
            bias_sigma = torch.exp(self.bias_logsigma)
            bias_epsilon = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_epsilon
        else:
            # Use mean weights
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between posterior and prior.
        
        KL(q||p) where q is posterior, p is prior N(0, prior_std²)
        """
        weight_sigma = torch.exp(self.weight_logsigma)
        bias_sigma = torch.exp(self.bias_logsigma)
        
        # KL for weights
        kl_weight = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_sigma**2) / (self.prior_std**2)
            - 2 * self.weight_logsigma
            + 2 * np.log(self.prior_std)
            - 1
        )
        
        # KL for bias
        kl_bias = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_sigma**2) / (self.prior_std**2)
            - 2 * self.bias_logsigma
            + 2 * np.log(self.prior_std)
            - 1
        )
        
        return kl_weight + kl_bias


class BayesianSmoothNN(nn.Module):
    """
    Bayesian neural network for function approximation with uncertainty.
    
    Provides both predictions and uncertainty estimates.
    """
    
    def __init__(self, hidden_layers=[32, 32, 32], prior_std=1.0):
        """
        Args:
            hidden_layers: List of hidden layer sizes
            prior_std: Prior standard deviation for weights
        """
        super(BayesianSmoothNN, self).__init__()
        
        self.prior_std = prior_std
        
        # Build network with Bayesian layers
        layers = []
        input_size = 1
        
        for hidden_size in hidden_layers:
            layers.append(BayesianLinear(input_size, hidden_size, prior_std))
            input_size = hidden_size
        
        layers.append(BayesianLinear(input_size, 1, prior_std))
        
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
    
    def forward(self, x, sample=True):
        """
        Forward pass with optional weight sampling.
        
        Args:
            x: Input tensor
            sample: If True, sample weights for stochastic prediction
        """
        out = x
        
        for i, layer in enumerate(self.layers[:-1]):
            out = layer(out, sample=sample)
            out = self.activation(out)
        
        out = self.layers[-1](out, sample=sample)
        
        return out
    
    def kl_divergence(self):
        """Total KL divergence across all layers."""
        kl = 0.0
        for layer in self.layers:
            kl += layer.kl_divergence()
        return kl
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Predict with uncertainty estimates.
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            mean, std: Mean prediction and standard deviation
        """
        self.eval()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class MCDropoutSmoothNN(nn.Module):
    """
    Monte Carlo Dropout for uncertainty quantification.
    
    Uses dropout at test time to approximate Bayesian inference.
    Simpler and faster than full Bayesian approach.
    """
    
    def __init__(self, hidden_layers=[32, 32, 32], dropout_rate=0.1):
        """
        Args:
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
        """
        super(MCDropoutSmoothNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Build network with dropout
        layers = []
        input_size = 1
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """Forward pass with dropout."""
        return self.network(x)
    
    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Predict with uncertainty using MC Dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of stochastic forward passes
            
        Returns:
            mean, std: Mean prediction and uncertainty estimate
        """
        # Keep dropout enabled during inference
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()
        
        return mean, std


class EnsembleSmoothNN(nn.Module):
    """
    Ensemble of neural networks for uncertainty quantification.
    
    Trains multiple independent models and uses their disagreement
    as a measure of uncertainty.
    """
    
    def __init__(self, num_models=5, hidden_layers=[32, 32, 32]):
        """
        Args:
            num_models: Number of models in ensemble
            hidden_layers: Architecture for each model
        """
        super(EnsembleSmoothNN, self).__init__()
        
        self.num_models = num_models
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            self._create_model(hidden_layers)
            for _ in range(num_models)
        ])
    
    def _create_model(self, hidden_layers):
        """Create a single model with given architecture."""
        layers = []
        input_size = 1
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        
        model = nn.Sequential(*layers)
        
        # Initialize with different random weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        return model
    
    def forward(self, x, model_idx=None):
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            model_idx: If specified, use only this model; else average all
        """
        if model_idx is not None:
            return self.models[model_idx](x)
        
        # Average predictions across all models
        predictions = torch.stack([model(x) for model in self.models])
        return predictions.mean(dim=0)
    
    def predict_with_uncertainty(self, x):
        """
        Predict with uncertainty from ensemble disagreement.
        
        Args:
            x: Input tensor
            
        Returns:
            mean, std: Mean prediction and ensemble standard deviation
        """
        self.eval()
        
        with torch.no_grad():
            predictions = torch.stack([model(x) for model in self.models])
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def get_individual_predictions(self, x):
        """Get predictions from all individual models."""
        self.eval()
        
        with torch.no_grad():
            predictions = [model(x) for model in self.models]
        
        return predictions


class BayesianLoss(nn.Module):
    """
    ELBO loss for Bayesian neural networks.
    
    Combines data likelihood and KL divergence.
    """
    
    def __init__(self, num_data_points, kl_weight=1.0):
        """
        Args:
            num_data_points: Total number of training points
            kl_weight: Weight for KL divergence term
        """
        super(BayesianLoss, self).__init__()
        self.num_data_points = num_data_points
        self.kl_weight = kl_weight
    
    def forward(self, predictions, targets, model):
        """
        Compute ELBO loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            model: Bayesian model (to get KL divergence)
            
        Returns:
            Total loss, likelihood term, KL term
        """
        # Negative log likelihood (data fitting)
        likelihood = F.mse_loss(predictions, targets, reduction='sum')
        likelihood = likelihood / self.num_data_points
        
        # KL divergence
        kl_div = model.kl_divergence() / self.num_data_points
        
        # ELBO loss (to minimize)
        total_loss = likelihood + self.kl_weight * kl_div
        
        return total_loss, likelihood, kl_div
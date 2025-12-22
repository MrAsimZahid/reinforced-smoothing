import torch
import torch.nn as nn


class SmoothnessLoss(nn.Module):
    """
    Custom loss function that balances data fitting and smoothness.

    The loss combines:
    1. MSE Loss: Fits the training data
    2. Derivative Penalty: Encourages smoothness by penalizing large derivatives

    Total Loss = MSE + λ * Smoothness_Penalty
    """

    def __init__(self, smoothness_weight=0.01, derivative_order=2):
        """
        Args:
            smoothness_weight: Weight (λ) for smoothness penalty
            derivative_order: 1 for first derivative, 2 for second derivative penalty
        """
        super(SmoothnessLoss, self).__init__()
        self.smoothness_weight = smoothness_weight
        self.derivative_order = derivative_order
        self.mse = nn.MSELoss()

    def compute_derivative(self, x, y, order=1):
        """
        Compute derivatives using automatic differentiation.

        Args:
            x: Input tensor (requires_grad=True)
            y: Output tensor
            order: Derivative order (1 or 2)

        Returns:
            Derivative tensor
        """
        # First derivative
        dy_dx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]

        if order == 1:
            return dy_dx

        # Second derivative
        d2y_dx2 = torch.autograd.grad(
            outputs=dy_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dy_dx),
            create_graph=True,
            retain_graph=True,
        )[0]

        return d2y_dx2

    def forward(self, predictions, targets, x_input):
        """
        Compute combined loss.

        Args:
            predictions: Model predictions
            targets: Ground truth noisy targets
            x_input: Input tensor (needed for derivative computation)

        Returns:
            Total loss, MSE component, smoothness component
        """
        # Data fitting loss
        mse_loss = self.mse(predictions, targets)

        # Smoothness penalty: penalize large derivatives
        if self.smoothness_weight > 0:
            derivatives = self.compute_derivative(
                x_input, predictions, self.derivative_order
            )
            smoothness_penalty = torch.mean(derivatives**2)
        else:
            smoothness_penalty = torch.tensor(0.0)

        # Combined loss
        total_loss = mse_loss + self.smoothness_weight * smoothness_penalty

        return total_loss, mse_loss, smoothness_penalty

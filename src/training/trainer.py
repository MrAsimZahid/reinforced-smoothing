"""Training utilities and loops."""

import torch
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training with custom loss functions."""

    def __init__(self, model, loss_fn, learning_rate=0.01):
        """
        Args:
            model: Neural network model
            loss_fn: Loss function instance
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training history - flexible to handle different loss components
        self.history = {
            "total_loss": [],
            "mse_loss": [],
            "smoothness_penalty": [],
            "pde_loss": [],  # For physics-informed losses
            "additional_losses": [],  # For any other loss components
        }

    def train_epoch(self, x_train, y_train):
        """Train for one epoch."""
        self.model.train()

        # Enable gradient computation for inputs (needed for derivatives)
        x_train_grad = x_train.clone().requires_grad_(True)

        # Forward pass
        predictions = self.model(x_train_grad)

        # Compute loss - handle variable number of return values
        loss_outputs = self.loss_fn(predictions, y_train, x_train_grad)

        # Unpack loss outputs flexibly
        if len(loss_outputs) == 3:
            # Standard format: (total_loss, mse_loss, smoothness_penalty)
            total_loss, mse_loss, smoothness_penalty = loss_outputs
            pde_loss = torch.tensor(0.0)

        elif len(loss_outputs) == 4:
            # Physics-informed format: (total_loss, mse_loss, pde_loss, smoothness_penalty)
            total_loss, mse_loss, pde_loss, smoothness_penalty = loss_outputs

        else:
            # Fallback: assume first is total loss, rest are components
            total_loss = loss_outputs[0]
            mse_loss = loss_outputs[1] if len(loss_outputs) > 1 else torch.tensor(0.0)
            smoothness_penalty = (
                loss_outputs[2] if len(loss_outputs) > 2 else torch.tensor(0.0)
            )
            pde_loss = torch.tensor(0.0)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Record history
        self.history["total_loss"].append(total_loss.item())
        self.history["mse_loss"].append(
            mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
        )
        self.history["smoothness_penalty"].append(
            smoothness_penalty.item()
            if isinstance(smoothness_penalty, torch.Tensor)
            else smoothness_penalty
        )
        self.history["pde_loss"].append(
            pde_loss.item() if isinstance(pde_loss, torch.Tensor) else pde_loss
        )

        return total_loss.item()

    def train(self, x_train, y_train, epochs=5000, verbose_interval=500):
        """
        Train the model for specified number of epochs.

        Args:
            x_train: Training inputs
            y_train: Training targets
            epochs: Number of training epochs
            verbose_interval: Print loss every N epochs
        """
        # Log header
        logger.info(
            f"{'Epoch':>6} {'Total Loss':>12} {'MSE Loss':>12} {'Smoothness':>12} {'PDE Loss':>12}"
        )
        logger.info("-" * 62)

        for epoch in range(epochs):
            loss = self.train_epoch(x_train, y_train)

            if (epoch + 1) % verbose_interval == 0 or epoch == 0:
                log_msg = (
                    f"{epoch+1:6d} "
                    f"{self.history['total_loss'][-1]:12.6f} "
                    f"{self.history['mse_loss'][-1]:12.6f} "
                    f"{self.history['smoothness_penalty'][-1]:12.6f}"
                )

                # Only show PDE loss if it's non-zero
                if self.history["pde_loss"][-1] > 1e-10:
                    log_msg += f" {self.history['pde_loss'][-1]:12.6f}"

                logger.info(log_msg)

        logger.info(f"\nTraining completed!")

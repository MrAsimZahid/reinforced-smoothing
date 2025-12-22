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

        # Training history
        self.history = {"total_loss": [], "mse_loss": [], "smoothness_penalty": []}

    def train_epoch(self, x_train, y_train):
        """Train for one epoch."""
        self.model.train()

        # Enable gradient computation for inputs (needed for derivatives)
        x_train_grad = x_train.clone().requires_grad_(True)

        # Forward pass
        predictions = self.model(x_train_grad)

        # Compute loss
        total_loss, mse_loss, smoothness_penalty = self.loss_fn(
            predictions, y_train, x_train_grad
        )

        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Record history
        self.history["total_loss"].append(total_loss.item())
        self.history["mse_loss"].append(mse_loss.item())
        self.history["smoothness_penalty"].append(smoothness_penalty.item())

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
        logger.info(
            f"Training with smoothness weight λ = {self.loss_fn.smoothness_weight}"
        )
        logger.info(
            f"{'Epoch':>6} {'Total Loss':>12} {'MSE Loss':>12} {'Smoothness':>12}"
        )
        logger.info("-" * 50)

        for epoch in range(epochs):
            loss = self.train_epoch(x_train, y_train)

            if (epoch + 1) % verbose_interval == 0 or epoch == 0:
                logger.info(
                    f"{epoch+1:6d} {self.history['total_loss'][-1]:12.6f} "
                    f"{self.history['mse_loss'][-1]:12.6f} "
                    f"{self.history['smoothness_penalty'][-1]:12.6f}"
                )

        logger.info(f"\nTraining completed!")

import numpy as np
import torch

class NoisyDataGenerator:
    """Generates noisy sinusoidal data for training and testing."""

    def __init__(self, n_points=40, noise_std=0.2, domain=(0, 2 * np.pi), seed=42):
        """
        Args:
            n_points: Number of training points to generate
            noise_std: Standard deviation of Gaussian noise
            domain: Tuple of (min, max) for x values
            seed: Random seed for reproducibility
        """
        self.n_points = n_points
        self.noise_std = noise_std
        self.domain = domain
        np.random.seed(seed)
        torch.manual_seed(seed)

    def ground_truth(self, x):
        """Ground truth function: y = sin(x)"""
        return np.sin(x)

    def generate_training_data(self):
        """Generate noisy training data."""
        x_train = np.linspace(self.domain[0], self.domain[1], self.n_points)
        y_clean = self.ground_truth(x_train)
        noise = np.random.normal(0, self.noise_std, self.n_points)
        y_noisy = y_clean + noise

        return (
            torch.FloatTensor(x_train).reshape(-1, 1),
            torch.FloatTensor(y_noisy).reshape(-1, 1),
        )

    def generate_test_data(self, n_points=200):
        """Generate clean test data for evaluation."""
        x_test = np.linspace(self.domain[0], self.domain[1], n_points)
        y_test = self.ground_truth(x_test)

        return (
            torch.FloatTensor(x_test).reshape(-1, 1),
            torch.FloatTensor(y_test).reshape(-1, 1),
        )

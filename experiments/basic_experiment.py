"""
Basic Experiment Runner for Reinforced Smoothing
================================================

This script runs the core experiment comparing different smoothness penalties
on noisy sinusoidal data.

Usage:
    python basic_experiment.py                          # Run with defaults
    python basic_experiment.py --config config.yaml    # Run with custom config
    python basic_experiment.py --lambda 0.01           # Quick test single lambda
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_generator import NoisyDataGenerator
from src.models.smooth_nn import SmoothNN
from src.losses.smoothness_loss import SmoothnessLoss
from src.training.trainer import Trainer
from src.visualization.visualizer import Visualizer


class ExperimentConfig:
    """Configuration management for experiments."""

    DEFAULT_CONFIG = {
        "data": {
            "n_train_points": 40,
            "noise_std": 0.2,
            "domain_min": 0,
            "domain_max": 2 * np.pi,
            "seed": 42,
        },
        "model": {"hidden_layers": [32, 32, 32], "activation": "tanh"},
        "loss": {"smoothness_weights": [0.0, 0.001, 0.01], "derivative_order": 2},
        "training": {"learning_rate": 0.01, "epochs": 5000, "verbose_interval": 1000},
        "visualization": {
            "n_test_points": 200,
            "save_figures": True,
            "figure_format": "png",
            "dpi": 300,
        },
        "output": {"results_dir": "results", "save_models": True, "save_logs": True},
    }

    def __init__(self, config_path=None):
        """Initialize configuration from file or defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Loaded config from: {config_path}")
        else:
            self.config = self.DEFAULT_CONFIG.copy()
            logger.info("✓ Using default configuration")

    def get(self, *keys):
        """Get nested config value."""
        value = self.config
        for key in keys:
            value = value[key]
        return value

    def set(self, value, *keys):
        """Set nested config value."""
        config = self.config
        for key in keys[:-1]:
            config = config[key]
        config[keys[-1]] = value

    def save(self, path):
        """Save configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"✓ Config saved to: {path}")


class ExperimentRunner:
    """Main experiment runner class."""

    def __init__(self, config):
        """
        Initialize experiment runner.

        Args:
            config: ExperimentConfig instance
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"✓ Using device: {self.device}")

    def setup_directories(self):
        """Create necessary output directories."""
        base_dir = Path(self.config.get("output", "results_dir"))

        self.dirs = {
            "base": base_dir,
            "figures": base_dir / "figures" / self.timestamp,
            "models": base_dir / "models" / self.timestamp,
            "logs": base_dir / "logs" / self.timestamp,
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Output directory: {self.dirs['base']}")

    def generate_data(self):
        """Generate training and test data."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: DATA GENERATION")
        logger.info("=" * 70)

        data_config = self.config.config["data"]
        self.data_generator = NoisyDataGenerator(
            n_points=data_config["n_train_points"],
            noise_std=data_config["noise_std"],
            domain=(data_config["domain_min"], data_config["domain_max"]),
            seed=data_config["seed"],
        )

        self.x_train, self.y_train = self.data_generator.generate_training_data()
        self.x_test, self.y_test = self.data_generator.generate_test_data(
            n_points=self.config.get("visualization", "n_test_points")
        )

        # Move to device
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.x_test = self.x_test.to(self.device)
        self.y_test = self.y_test.to(self.device)

        logger.info(f"✓ Training samples: {len(self.x_train)}")
        logger.info(f"✓ Test samples: {len(self.x_test)}")
        logger.info(f"✓ Noise std: {data_config['noise_std']}")
        logger.info(
            f"✓ Domain: [{data_config['domain_min']:.2f}, {data_config['domain_max']:.2f}]"
        )

    def create_model(self):
        """Create neural network model."""
        model_config = self.config.config["model"]

        # Map activation name to function
        activation_map = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "elu": torch.nn.ELU(),
        }

        activation = activation_map.get(
            model_config["activation"].lower(), torch.nn.Tanh()
        )

        model = SmoothNN(
            hidden_layers=model_config["hidden_layers"], activation=activation
        )

        return model.to(self.device)

    def train_model(self, model, smoothness_weight, model_name):
        """
        Train a single model.

        Args:
            model: Neural network model
            smoothness_weight: Lambda value for smoothness penalty
            model_name: Name for logging

        Returns:
            Trained model and training history
        """
        loss_config = self.config.config["loss"]
        train_config = self.config.config["training"]

        # Create loss function
        loss_fn = SmoothnessLoss(
            smoothness_weight=smoothness_weight,
            derivative_order=loss_config["derivative_order"],
        )

        # Create trainer
        trainer = Trainer(
            model=model, loss_fn=loss_fn, learning_rate=train_config["learning_rate"]
        )

        # Train
        logger.info(f"\nTraining: {model_name}")
        logger.info("-" * 70)
        trainer.train(
            self.x_train,
            self.y_train,
            epochs=train_config["epochs"],
            verbose_interval=train_config["verbose_interval"],
        )

        return model, trainer.history

    def evaluate_model(self, model, name):
        """
        Evaluate model performance.

        Args:
            model: Trained model
            name: Model name

        Returns:
            Dictionary of metrics
        """
        model.eval()
        with torch.no_grad():
            y_pred = model(self.x_test)

            # MSE on test set
            test_mse = torch.mean((y_pred - self.y_test) ** 2).item()

            # MSE on training set
            y_train_pred = model(self.x_train)
            train_mse = torch.mean((y_train_pred - self.y_train) ** 2).item()

        metrics = {"name": name, "test_mse": test_mse, "train_mse": train_mse}

        logger.info(f"\n{name} - Evaluation Metrics:")
        logger.info(f"  Train MSE: {train_mse:.6f}")
        logger.info(f"  Test MSE:  {test_mse:.6f}")

        return metrics

    def run_experiment(self):
        """Run the complete experiment."""
        logger.info("\n" + "=" * 70)
        logger.info("REINFORCED SMOOTHING EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Device: {self.device}")

        # Generate data
        self.generate_data()

        # Train models with different smoothness weights
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 70)

        smoothness_weights = self.config.get("loss", "smoothness_weights")

        self.models = {}
        self.histories = {}
        self.metrics = []

        for lambda_val in smoothness_weights:
            # Create model name
            if lambda_val == 0:
                model_name = "No Smoothing (λ=0)"
            elif lambda_val < 0.01:
                model_name = f"Light Smoothing (λ={lambda_val})"
            else:
                model_name = f"Strong Smoothing (λ={lambda_val})"

            # Create and train model
            model = self.create_model()
            trained_model, history = self.train_model(model, lambda_val, model_name)

            # Store results
            self.models[model_name] = trained_model
            self.histories[model_name] = history

            # Evaluate
            metrics = self.evaluate_model(trained_model, model_name)
            self.metrics.append(metrics)

            # Save model if configured
            if self.config.get("output", "save_models"):
                model_path = self.dirs["models"] / f"model_lambda_{lambda_val}.pth"
                torch.save(trained_model.state_dict(), model_path)
                logger.info(f"✓ Model saved: {model_path}")

        # Visualize results
        self.visualize_results()

        # Save experiment log
        self.save_experiment_log()

        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.dirs['base']}")

    def visualize_results(self):
        """Generate all visualizations."""
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: VISUALIZATION")
        logger.info("=" * 70)

        viz_config = self.config.config["visualization"]

        # Move data back to CPU for visualization
        x_train_cpu = self.x_train.cpu()
        y_train_cpu = self.y_train.cpu()
        x_test_cpu = self.x_test.cpu()
        y_test_cpu = self.y_test.cpu()

        # Move models to CPU
        models_cpu = {}
        for name, model in self.models.items():
            model_cpu = model.cpu()
            models_cpu[name] = model_cpu

        # Create visualizations
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        # Main results plot
        fig_path = (
            self.dirs["figures"] / f"results_comparison.{viz_config['figure_format']}"
        )
        Visualizer.plot_results(
            models_cpu,
            x_train_cpu,
            y_train_cpu,
            x_test_cpu,
            y_test_cpu,
            self.data_generator,
            self.histories,
        )
        plt.savefig(fig_path, dpi=viz_config["dpi"], bbox_inches="tight")
        logger.info(f"✓ Saved: {fig_path}")
        plt.close()

        # Derivative analysis plot
        fig_path = (
            self.dirs["figures"] / f"derivative_analysis.{viz_config['figure_format']}"
        )
        Visualizer.plot_derivative_analysis(models_cpu, x_test_cpu, colors)
        plt.savefig(fig_path, dpi=viz_config["dpi"], bbox_inches="tight")
        logger.info(f"✓ Saved: {fig_path}")
        plt.close()

        # Move models back to original device
        for name, model in self.models.items():
            model.to(self.device)

    def save_experiment_log(self):
        """Save experiment log with all details."""
        if not self.config.get("output", "save_logs"):
            return

        log_path = self.dirs["logs"] / "experiment_log.txt"

        with open(log_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("REINFORCED SMOOTHING - EXPERIMENT LOG\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(yaml.dump(self.config.config, default_flow_style=False))
            f.write("\n")

            f.write("RESULTS\n")
            f.write("-" * 70 + "\n")
            for metrics in self.metrics:
                f.write(f"\n{metrics['name']}:\n")
                f.write(f"  Train MSE: {metrics['train_mse']:.6f}\n")
                f.write(f"  Test MSE:  {metrics['test_mse']:.6f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 70 + "\n")

        logger.info(f"✓ Log saved: {log_path}")

        # Also save config
        config_path = self.dirs["logs"] / "config.yaml"
        self.config.save(config_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run reinforced smoothing experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_experiment.py
  python basic_experiment.py --config my_config.yaml
  python basic_experiment.py --lambda 0.01 --epochs 3000
  python basic_experiment.py --no-save
        """,
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_val",
        default=None,
        help="Single smoothness weight to test (quick mode)",
    )

    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    parser.add_argument(
        "--no-save", action="store_true", help="Do not save models and logs"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (fewer epochs, single model)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = ExperimentConfig(args.config)

    # Apply command line overrides
    if args.lambda_val is not None:
        config.set([args.lambda_val], "loss", "smoothness_weights")
        logger.info(f"✓ Using single λ = {args.lambda_val}")

    if args.epochs is not None:
        config.set(args.epochs, "training", "epochs")
        logger.info(f"✓ Using {args.epochs} epochs")

    if args.learning_rate is not None:
        config.set(args.learning_rate, "training", "learning_rate")
        logger.info(f"✓ Using learning rate = {args.learning_rate}")

    if args.seed is not None:
        config.set(args.seed, "data", "seed")
        logger.info(f"✓ Using seed = {args.seed}")

    if args.no_save:
        config.set(False, "output", "save_models")
        config.set(False, "output", "save_logs")
        logger.info("✓ Saving disabled")

    if args.quick:
        config.set([0.01], "loss", "smoothness_weights")
        config.set(1000, "training", "epochs")
        config.set(500, "training", "verbose_interval")
        logger.info("✓ Quick mode enabled")

    # Run experiment
    try:
        runner = ExperimentRunner(config)
        runner.run_experiment()

        logger.info("\n✓ Experiment completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("\n\n⚠ Experiment interrupted by user")
        return 1

    except Exception as e:
        logger.info(f"\n\n✗ Experiment failed with error:")
        logger.info(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

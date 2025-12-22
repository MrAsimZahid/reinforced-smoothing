"""
Ablation Study
==============

Systematically tests each component to understand their individual contributions.
"""

import sys
import os
import matplotlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
matplotlib.use("Agg")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.data.data_generator import NoisyDataGenerator
from src.models.smooth_nn import SmoothNN
from src.losses.smoothness_loss import SmoothnessLoss
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Conducts ablation study to analyze component contributions.
    """

    def __init__(self, output_dir="results/ablation"):
        """
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate data once
        self.data_gen = NoisyDataGenerator(n_points=40, noise_std=0.2, seed=42)
        self.x_train, self.y_train = self.data_gen.generate_training_data()
        self.x_test, self.y_test = self.data_gen.generate_test_data(200)

    def study_network_depth(self):
        """Test effect of network depth on smoothness."""
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 1: Network Depth")
        logger.info("=" * 70)

        depths = [
            ([16], "1 layer (16 neurons)"),
            ([32, 32], "2 layers (32 each)"),
            ([32, 32, 32], "3 layers (32 each)"),
            ([32, 32, 32, 32], "4 layers (32 each)"),
            ([64, 64], "2 layers (64 each)"),
        ]

        results = []

        for hidden_layers, name in depths:
            logger.info(f"\nTesting: {name}")

            model = SmoothNN(hidden_layers=hidden_layers)
            loss_fn = SmoothnessLoss(smoothness_weight=0.01, derivative_order=2)
            trainer = Trainer(model, loss_fn, learning_rate=0.01)

            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred = model(self.x_test)
                test_mse = torch.mean((y_pred - self.y_test) ** 2).item()

            results.append(
                {
                    "name": name,
                    "model": model,
                    "test_mse": test_mse,
                    "history": trainer.history,
                }
            )

            logger.info(f"Test MSE: {test_mse:.6f}")

        self._plot_comparison(results, "Network Depth Comparison", "network_depth.png")
        return results

    def study_activation_functions(self):
        """Test different activation functions."""
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 2: Activation Functions")
        logger.info("=" * 70)

        activations = [
            (torch.nn.Tanh(), "Tanh"),
            (torch.nn.ReLU(), "ReLU"),
            (torch.nn.ELU(), "ELU"),
            (torch.nn.Sigmoid(), "Sigmoid"),
        ]

        results = []

        for activation, name in activations:
            logger.info(f"\nTesting: {name}")

            model = SmoothNN(hidden_layers=[32, 32, 32], activation=activation)
            loss_fn = SmoothnessLoss(smoothness_weight=0.01, derivative_order=2)
            trainer = Trainer(model, loss_fn, learning_rate=0.01)

            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred = model(self.x_test)
                test_mse = torch.mean((y_pred - self.y_test) ** 2).item()

            results.append(
                {
                    "name": name,
                    "model": model,
                    "test_mse": test_mse,
                    "history": trainer.history,
                }
            )

            logger.info(f"Test MSE: {test_mse:.6f}")

        self._plot_comparison(
            results, "Activation Function Comparison", "activation_functions.png"
        )
        return results

    def study_derivative_orders(self):
        """Test 1st vs 2nd order derivative penalties."""
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 3: Derivative Order")
        logger.info("=" * 70)

        orders = [
            (0, "No smoothness"),
            (1, "1st derivative penalty"),
            (2, "2nd derivative penalty"),
        ]

        results = []

        for order, name in orders:
            logger.info(f"\nTesting: {name}")

            model = SmoothNN(hidden_layers=[32, 32, 32])

            if order == 0:
                loss_fn = SmoothnessLoss(smoothness_weight=0.0)
            else:
                loss_fn = SmoothnessLoss(smoothness_weight=0.01, derivative_order=order)

            trainer = Trainer(model, loss_fn, learning_rate=0.01)
            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred = model(self.x_test)
                test_mse = torch.mean((y_pred - self.y_test) ** 2).item()

            results.append(
                {
                    "name": name,
                    "model": model,
                    "test_mse": test_mse,
                    "history": trainer.history,
                }
            )

            logger.info(f"Test MSE: {test_mse:.6f}")

        self._plot_comparison(
            results, "Derivative Order Comparison", "derivative_orders.png"
        )
        return results

    def study_lambda_sensitivity(self):
        """Study sensitivity to smoothness weight λ."""
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 4: Lambda Sensitivity")
        logger.info("=" * 70)

        lambdas = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

        results = []
        test_mses = []
        train_mses = []

        for lam in lambdas:
            logger.info(f"\nTesting λ = {lam}")

            model = SmoothNN(hidden_layers=[32, 32, 32])
            loss_fn = SmoothnessLoss(smoothness_weight=lam, derivative_order=2)
            trainer = Trainer(model, loss_fn, learning_rate=0.01)

            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred_test = model(self.x_test)
                test_mse = torch.mean((y_pred_test - self.y_test) ** 2).item()

                y_pred_train = model(self.x_train)
                train_mse = torch.mean((y_pred_train - self.y_train) ** 2).item()

            test_mses.append(test_mse)
            train_mses.append(train_mse)

            results.append(
                {"lambda": lam, "test_mse": test_mse, "train_mse": train_mse}
            )

            logger.info(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")

        # Plot sensitivity
        self._plot_lambda_sensitivity(lambdas, train_mses, test_mses)

        return results

    def _plot_comparison(self, results, title, filename):
        """Plot comparison of different configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

        # Plot 1: Predictions
        ax = axes[0, 0]
        x_dense = np.linspace(0, 2 * np.pi, 500)
        y_dense = self.data_gen.ground_truth(x_dense)
        ax.plot(x_dense, y_dense, "k-", linewidth=2, label="Ground Truth")
        ax.scatter(
            self.x_train.numpy(),
            self.y_train.numpy(),
            c="gray",
            s=30,
            alpha=0.5,
            label="Training Data",
        )

        for idx, result in enumerate(results):
            model = result["model"]
            model.eval()
            with torch.no_grad():
                y_pred = model(self.x_test).numpy()
            ax.plot(
                self.x_test.numpy(),
                y_pred,
                linewidth=2,
                color=colors[idx],
                label=result["name"],
                alpha=0.8,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Predictions Comparison")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Test MSE comparison
        ax = axes[0, 1]
        names = [r["name"] for r in results]
        mses = [r["test_mse"] for r in results]
        bars = ax.bar(range(len(names)), mses, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Test MSE")
        ax.set_title("Test MSE Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Training curves
        ax = axes[1, 0]
        for idx, result in enumerate(results):
            history = result["history"]
            epochs = range(1, len(history["total_loss"]) + 1)
            ax.plot(
                epochs,
                history["total_loss"],
                linewidth=2,
                color=colors[idx],
                label=result["name"],
                alpha=0.8,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training Loss Curves")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: MSE vs Smoothness tradeoff
        ax = axes[1, 1]
        for idx, result in enumerate(results):
            history = result["history"]
            final_mse = history["mse_loss"][-1]
            final_smooth = history["smoothness_penalty"][-1]
            ax.scatter(
                final_smooth,
                final_mse,
                s=200,
                color=colors[idx],
                label=result["name"],
                alpha=0.8,
            )
        ax.set_xlabel("Smoothness Penalty")
        ax.set_ylabel("MSE Loss")
        ax.set_title("MSE vs Smoothness Tradeoff")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved: {self.output_dir / filename}")
        plt.close()

    def _plot_lambda_sensitivity(self, lambdas, train_mses, test_mses):
        """Plot lambda sensitivity analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        ax = axes[0]
        ax.plot(lambdas, train_mses, "o-", linewidth=2, markersize=8, label="Train MSE")
        ax.plot(lambdas, test_mses, "s-", linewidth=2, markersize=8, label="Test MSE")
        ax.set_xlabel("Smoothness Weight (λ)", fontweight="bold")
        ax.set_ylabel("MSE", fontweight="bold")
        ax.set_title("Lambda Sensitivity (Linear Scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Log scale
        ax = axes[1]
        ax.plot(lambdas, train_mses, "o-", linewidth=2, markersize=8, label="Train MSE")
        ax.plot(lambdas, test_mses, "s-", linewidth=2, markersize=8, label="Test MSE")
        ax.set_xlabel("Smoothness Weight (λ)", fontweight="bold")
        ax.set_ylabel("MSE", fontweight="bold")
        ax.set_title("Lambda Sensitivity (Log-Log Scale)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "lambda_sensitivity.png", dpi=300, bbox_inches="tight"
        )
        logger.info(f"✓ Saved: {self.output_dir / 'lambda_sensitivity.png'}")
        plt.close()

    def run_all_studies(self):
        """Run all ablation studies."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE ABLATION STUDY")
        logger.info("=" * 70)

        results = {}

        results["depth"] = self.study_network_depth()
        results["activation"] = self.study_activation_functions()
        results["derivative_order"] = self.study_derivative_orders()
        results["lambda"] = self.study_lambda_sensitivity()

        # Save summary
        self._save_summary(results)

        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.output_dir}")

        return results

    def _save_summary(self, results):
        """Save text summary of results."""
        summary_path = self.output_dir / "ablation_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("ABLATION STUDY SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            # Network depth
            f.write("1. NETWORK DEPTH\n")
            f.write("-" * 70 + "\n")
            for r in results["depth"]:
                f.write(f"{r['name']:30s} Test MSE: {r['test_mse']:.6f}\n")
            f.write("\n")

            # Activation functions
            f.write("2. ACTIVATION FUNCTIONS\n")
            f.write("-" * 70 + "\n")
            for r in results["activation"]:
                f.write(f"{r['name']:30s} Test MSE: {r['test_mse']:.6f}\n")
            f.write("\n")

            # Derivative orders
            f.write("3. DERIVATIVE ORDERS\n")
            f.write("-" * 70 + "\n")
            for r in results["derivative_order"]:
                f.write(f"{r['name']:30s} Test MSE: {r['test_mse']:.6f}\n")
            f.write("\n")

            # Lambda sensitivity
            f.write("4. LAMBDA SENSITIVITY\n")
            f.write("-" * 70 + "\n")
            for r in results["lambda"]:
                f.write(
                    f"λ = {r['lambda']:.4f}  Train: {r['train_mse']:.6f}  "
                    f"Test: {r['test_mse']:.6f}\n"
                )

        logger.info(f"✓ Summary saved: {summary_path}")


def main():
    """Run ablation study."""
    study = AblationStudy(output_dir="results/ablation")
    results = study.run_all_studies()

    return results


if __name__ == "__main__":
    main()

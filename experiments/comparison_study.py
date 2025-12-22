"""
Comparison Study
================

Compare different smoothness approaches and model architectures.
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
import time

from src.data.data_generator import NoisyDataGenerator
from src.models.smooth_nn import SmoothNN
from src.models.residual_smooth_nn import (
    ResidualSmoothNN,
    FourierResidualNN,
    AdaptiveResidualNN,
)
from src.models.bayesian_smooth_nn import (
    BayesianSmoothNN,
    MCDropoutSmoothNN,
    EnsembleSmoothNN,
    BayesianLoss,
)
from src.losses.smoothness_loss import SmoothnessLoss
from src.losses.curvature_loss import CurvatureLoss, TotalVariationLoss
from src.losses.physics_informed_loss import PhysicsInformedLoss
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ComparisonStudy:
    """
    Compare different approaches for smooth function learning.
    """

    def __init__(self, output_dir="results/comparison"):
        """
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate data
        self.data_gen = NoisyDataGenerator(n_points=40, noise_std=0.2, seed=42)
        self.x_train, self.y_train = self.data_gen.generate_training_data()
        self.x_test, self.y_test = self.data_gen.generate_test_data(200)

        self.results = []

    def compare_loss_functions(self):
        """Compare different loss function types."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON 1: Loss Functions")
        logger.info("=" * 70)

        configs = [
            (
                "Smoothness (2nd order)",
                SmoothNN([32, 32, 32]),
                SmoothnessLoss(smoothness_weight=0.01, derivative_order=2),
            ),
            (
                "Curvature Loss",
                SmoothNN([32, 32, 32]),
                CurvatureLoss(curvature_weight=0.01),
            ),
            (
                "Total Variation",
                SmoothNN([32, 32, 32]),
                TotalVariationLoss(tv_weight=0.001),
            ),
            (
                "Physics-Informed (PDE)",
                SmoothNN([32, 32, 32]),
                PhysicsInformedLoss(pde_weight=0.1, smoothness_weight=0.01),
            ),
        ]

        for name, model, loss_fn in configs:
            logger.info(f"\nTraining: {name}")

            start_time = time.time()
            trainer = Trainer(model, loss_fn, learning_rate=0.01)
            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )
            train_time = time.time() - start_time

            # Evaluate
            test_mse = self._evaluate_model(model)

            self.results.append(
                {
                    "category": "Loss Function",
                    "name": name,
                    "model": model,
                    "test_mse": test_mse,
                    "train_time": train_time,
                    "history": trainer.history,
                }
            )

            logger.info(f"Test MSE: {test_mse:.6f}, Time: {train_time:.2f}s")

    def compare_architectures(self):
        """Compare different neural network architectures."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON 2: Architectures")
        logger.info("=" * 70)

        loss_fn = SmoothnessLoss(smoothness_weight=0.01, derivative_order=2)

        configs = [
            ("Standard MLP", SmoothNN([32, 32, 32])),
            ("Residual Network", ResidualSmoothNN(hidden_size=64, num_blocks=3)),
            ("Fourier Features", FourierResidualNN(hidden_size=64, num_frequencies=10)),
            ("Adaptive Depth", AdaptiveResidualNN(hidden_size=64, num_blocks=4)),
        ]

        for name, model in configs:
            logger.info(f"\nTraining: {name}")

            start_time = time.time()
            trainer = Trainer(model, loss_fn, learning_rate=0.01)
            trainer.train(
                self.x_train, self.y_train, epochs=3000, verbose_interval=1000
            )
            train_time = time.time() - start_time

            # Evaluate
            test_mse = self._evaluate_model(model)

            self.results.append(
                {
                    "category": "Architecture",
                    "name": name,
                    "model": model,
                    "test_mse": test_mse,
                    "train_time": train_time,
                    "history": trainer.history,
                }
            )

            logger.info(f"Test MSE: {test_mse:.6f}, Time: {train_time:.2f}s")

    def compare_uncertainty_methods(self):
        """Compare uncertainty quantification approaches."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON 3: Uncertainty Quantification")
        logger.info("=" * 70)

        # Standard model (no uncertainty)
        logger.info("\nTraining: Standard (No Uncertainty)")
        standard_model = SmoothNN([32, 32, 32])
        loss_fn = SmoothnessLoss(smoothness_weight=0.01, derivative_order=2)
        trainer = Trainer(standard_model, loss_fn, learning_rate=0.01)
        trainer.train(self.x_train, self.y_train, epochs=3000, verbose_interval=1000)

        standard_mse = self._evaluate_model(standard_model)
        logger.info(f"Test MSE: {standard_mse:.6f}")

        # Bayesian model
        logger.info("\nTraining: Bayesian NN")
        bayesian_model = BayesianSmoothNN([32, 32, 32], prior_std=1.0)
        bayesian_mse = self._train_bayesian_model(bayesian_model)

        # MC Dropout
        logger.info("\nTraining: MC Dropout")
        dropout_model = MCDropoutSmoothNN([32, 32, 32], dropout_rate=0.1)
        trainer = Trainer(dropout_model, loss_fn, learning_rate=0.01)
        trainer.train(self.x_train, self.y_train, epochs=3000, verbose_interval=1000)
        dropout_mse = self._evaluate_model(dropout_model)
        logger.info(f"Test MSE: {dropout_mse:.6f}")

        # Ensemble
        logger.info("\nTraining: Ensemble (5 models)")
        ensemble_model = EnsembleSmoothNN(num_models=5, hidden_layers=[32, 32, 32])
        ensemble_mse = self._train_ensemble(ensemble_model, loss_fn)

        # Store results
        uncertainty_methods = [
            ("Standard (No Uncertainty)", standard_model, standard_mse),
            ("Bayesian NN", bayesian_model, bayesian_mse),
            ("MC Dropout", dropout_model, dropout_mse),
            ("Ensemble", ensemble_model, ensemble_mse),
        ]

        for name, model, mse in uncertainty_methods:
            self.results.append(
                {
                    "category": "Uncertainty",
                    "name": name,
                    "model": model,
                    "test_mse": mse,
                    "train_time": 0.0,  # Not tracked for simplicity
                    "history": None,
                }
            )

        # Visualize uncertainty
        self._visualize_uncertainty(uncertainty_methods)

    def _train_bayesian_model(self, model):
        """Train Bayesian model with ELBO loss."""
        bayesian_loss = BayesianLoss(num_data_points=len(self.x_train), kl_weight=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(3000):
            model.train()
            x_train_grad = self.x_train.clone().requires_grad_(True)

            # Forward pass
            predictions = model(x_train_grad, sample=True)

            # Compute ELBO loss
            total_loss, likelihood, kl = bayesian_loss(predictions, self.y_train, model)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                logger.info(
                    f"Epoch {epoch+1}: Loss = {total_loss:.6f}, "
                    f"Likelihood = {likelihood:.6f}, KL = {kl:.6f}"
                )

        # Evaluate
        model.eval()
        with torch.no_grad():
            mean_pred, _ = model.predict_with_uncertainty(self.x_test, num_samples=50)
            test_mse = torch.mean((mean_pred - self.y_test) ** 2).item()

        return test_mse

    def _train_ensemble(self, ensemble_model, loss_fn):
        """Train ensemble of models."""
        for i in range(ensemble_model.num_models):
            logger.info(f"  Training model {i+1}/{ensemble_model.num_models}")
            model = ensemble_model.models[i]
            trainer = Trainer(model, loss_fn, learning_rate=0.01)
            trainer.train(
                self.x_train, self.y_train, epochs=2000, verbose_interval=2000
            )

        # Evaluate ensemble
        mean_pred, _ = ensemble_model.predict_with_uncertainty(self.x_test)
        test_mse = torch.mean((mean_pred - self.y_test) ** 2).item()

        return test_mse

    def _evaluate_model(self, model):
        """Evaluate model on test set."""
        model.eval()
        with torch.no_grad():
            y_pred = model(self.x_test)
            test_mse = torch.mean((y_pred - self.y_test) ** 2).item()
        return test_mse

    def _visualize_uncertainty(self, methods):
        """Visualize predictions with uncertainty bands."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        x_dense = np.linspace(0, 2 * np.pi, 500)
        y_dense = self.data_gen.ground_truth(x_dense)

        for idx, (name, model, mse) in enumerate(methods):
            ax = axes[idx]

            # Ground truth and data
            ax.plot(
                x_dense, y_dense, "k-", linewidth=2, label="Ground Truth", alpha=0.7
            )
            ax.scatter(
                self.x_train.numpy(),
                self.y_train.numpy(),
                c="gray",
                s=30,
                alpha=0.5,
                label="Training Data",
            )

            # Predictions with uncertainty
            if "Bayesian" in name or "Dropout" in name:
                # These models need num_samples parameter
                mean, std = model.predict_with_uncertainty(self.x_test, num_samples=50)
                mean = mean.detach().numpy()
                std = std.detach().numpy()

                ax.plot(
                    self.x_test.numpy(), mean, "b-", linewidth=2, label="Prediction"
                )
                ax.fill_between(
                    self.x_test.numpy().flatten(),
                    (mean - 2 * std).flatten(),
                    (mean + 2 * std).flatten(),
                    alpha=0.3,
                    label="95% Confidence",
                )

            elif "Ensemble" in name:
                # Ensemble doesn't need num_samples - it uses all models
                mean, std = model.predict_with_uncertainty(self.x_test)
                mean = mean.detach().numpy()
                std = std.detach().numpy()

                ax.plot(
                    self.x_test.numpy(), mean, "b-", linewidth=2, label="Prediction"
                )
                ax.fill_between(
                    self.x_test.numpy().flatten(),
                    (mean - 2 * std).flatten(),
                    (mean + 2 * std).flatten(),
                    alpha=0.3,
                    label="95% Confidence",
                )

            else:
                # Standard model without uncertainty
                with torch.no_grad():
                    y_pred = model(self.x_test).numpy()
                ax.plot(
                    self.x_test.numpy(), y_pred, "b-", linewidth=2, label="Prediction"
                )

            ax.set_title(f"{name}\nTest MSE: {mse:.4f}", fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Uncertainty Quantification Comparison", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "uncertainty_comparison.png", dpi=300, bbox_inches="tight"
        )
        logger.info(f"✓ Saved: {self.output_dir / 'uncertainty_comparison.png'}")
        plt.close()

    def generate_summary_plots(self):
        """Generate comprehensive comparison plots."""
        logger.info("\n" + "=" * 70)
        logger.info("Generating Summary Visualizations")
        logger.info("=" * 70)

        fig = plt.figure(figsize=(18, 12))

        # Group results by category
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        # Plot 1: MSE comparison by category
        ax1 = plt.subplot(2, 3, 1)
        for cat_idx, (cat_name, cat_results) in enumerate(categories.items()):
            names = [r["name"] for r in cat_results]
            mses = [r["test_mse"] for r in cat_results]
            x_pos = np.arange(len(names)) + cat_idx * (len(names) + 1)
            ax1.bar(x_pos, mses, label=cat_name)

        ax1.set_ylabel("Test MSE")
        ax1.set_title("Test MSE Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Training time comparison
        ax2 = plt.subplot(2, 3, 2)
        all_names = [r["name"] for r in self.results if r["train_time"] > 0]
        all_times = [r["train_time"] for r in self.results if r["train_time"] > 0]
        if all_times:
            ax2.barh(range(len(all_names)), all_times)
            ax2.set_yticks(range(len(all_names)))
            ax2.set_yticklabels(all_names, fontsize=8)
            ax2.set_xlabel("Training Time (s)")
            ax2.set_title("Training Time Comparison")
            ax2.grid(True, alpha=0.3, axis="x")

        # Plot 3-6: Predictions from best models in each category
        plot_idx = 3
        for cat_name, cat_results in categories.items():
            if plot_idx > 6:
                break

            ax = plt.subplot(2, 3, plot_idx)

            # Find best model in category
            best_result = min(cat_results, key=lambda x: x["test_mse"])
            model = best_result["model"]

            # Plot
            x_dense = np.linspace(0, 2 * np.pi, 500)
            y_dense = self.data_gen.ground_truth(x_dense)
            ax.plot(x_dense, y_dense, "k-", linewidth=2, label="Ground Truth")
            ax.scatter(
                self.x_train.numpy(), self.y_train.numpy(), c="gray", s=30, alpha=0.5
            )

            model.eval()
            with torch.no_grad():
                y_pred = model(self.x_test).numpy()
            ax.plot(
                self.x_test.numpy(),
                y_pred,
                "b-",
                linewidth=2,
                label=f'{best_result["name"]}\nMSE: {best_result["test_mse"]:.4f}',
            )

            ax.set_title(f"Best {cat_name}", fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        plt.suptitle("Comprehensive Method Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comprehensive_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        logger.info(f"✓ Saved: {self.output_dir / 'comprehensive_comparison.png'}")
        plt.close()

    def save_comparison_report(self):
        """Save detailed comparison report."""
        report_path = self.output_dir / "comparison_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("METHOD COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Group by category
            categories = {}
            for r in self.results:
                cat = r["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(r)

            for cat_name, cat_results in categories.items():
                f.write(f"\n{cat_name.upper()}\n")
                f.write("-" * 70 + "\n")

                # Sort by MSE
                sorted_results = sorted(cat_results, key=lambda x: x["test_mse"])

                for rank, r in enumerate(sorted_results, 1):
                    f.write(f"{rank}. {r['name']}\n")
                    f.write(f"   Test MSE: {r['test_mse']:.6f}\n")
                    if r["train_time"] > 0:
                        f.write(f"   Training Time: {r['train_time']:.2f}s\n")
                    f.write("\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("CONCLUSION\n")
            f.write("=" * 70 + "\n")

            # Find overall best
            best = min(self.results, key=lambda x: x["test_mse"])
            f.write(f"\nBest Overall Method: {best['name']}\n")
            f.write(f"Category: {best['category']}\n")
            f.write(f"Test MSE: {best['test_mse']:.6f}\n")

        logger.info(f"✓ Report saved: {report_path}")

    def run_all_comparisons(self):
        """Run all comparison studies."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE METHOD COMPARISON")
        logger.info("=" * 70)

        self.compare_loss_functions()
        self.compare_architectures()
        self.compare_uncertainty_methods()

        self.generate_summary_plots()
        self.save_comparison_report()

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON STUDY COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """Run comparison study."""
    study = ComparisonStudy(output_dir="results/comparison")
    study.run_all_comparisons()


if __name__ == "__main__":
    main()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from matplotlib.gridspec import GridSpec


class Visualizer:
    """Creates comprehensive visualizations of results."""

    @staticmethod
    def plot_results(
        models_dict, x_train, y_train, x_test, y_test, data_generator, histories
    ):
        """
        Create comprehensive visualization comparing different models.

        Args:
            models_dict: Dictionary of {name: model}
            x_train, y_train: Training data
            x_test, y_test: Test data
            data_generator: Data generator instance
            histories: Dictionary of {name: training_history}
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Color scheme
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        # 1. Main comparison plot
        ax1 = fig.add_subplot(gs[0, :])

        # Plot ground truth
        x_dense = np.linspace(0, 2 * np.pi, 500)
        y_dense = data_generator.ground_truth(x_dense)
        ax1.plot(
            x_dense,
            y_dense,
            "k-",
            linewidth=2.5,
            label="Ground Truth: sin(x)",
            zorder=5,
        )

        # Plot training data
        ax1.scatter(
            x_train.numpy(),
            y_train.numpy(),
            c="gray",
            s=50,
            alpha=0.6,
            label="Noisy Training Data",
            zorder=3,
        )

        # Plot model predictions
        for idx, (name, model) in enumerate(models_dict.items()):
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test).numpy()
            ax1.plot(
                x_test.numpy(),
                y_pred,
                linewidth=2,
                label=name,
                color=colors[idx],
                alpha=0.8,
            )

        ax1.set_xlabel("x", fontsize=12, fontweight="bold")
        ax1.set_ylabel("y", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Model Comparison: Fitting Noisy Sinusoidal Data",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 2 * np.pi])

        # 2-4. Individual model plots with error analysis
        for idx, (name, model) in enumerate(models_dict.items()):
            ax = fig.add_subplot(gs[1, idx])

            # Plot ground truth and data
            ax.plot(
                x_dense, y_dense, "k-", linewidth=2, label="Ground Truth", alpha=0.7
            )
            ax.scatter(x_train.numpy(), y_train.numpy(), c="gray", s=30, alpha=0.5)

            # Plot prediction
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test).numpy()
            ax.plot(
                x_test.numpy(),
                y_pred,
                linewidth=2.5,
                color=colors[idx],
                label="Prediction",
            )

            # Compute MSE on clean test data
            mse = np.mean((y_pred.flatten() - y_test.numpy().flatten()) ** 2)

            ax.set_title(f"{name}\nTest MSE: {mse:.4f}", fontsize=11, fontweight="bold")
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 5-7. Training loss curves
        for idx, (name, history) in enumerate(histories.items()):
            ax = fig.add_subplot(gs[2, idx])

            epochs = range(1, len(history["total_loss"]) + 1)

            ax.plot(
                epochs,
                history["total_loss"],
                linewidth=2,
                color=colors[idx],
                label="Total Loss",
            )
            ax.plot(
                epochs,
                history["mse_loss"],
                linewidth=2,
                linestyle="--",
                color=colors[idx],
                alpha=0.6,
                label="MSE Loss",
            )

            ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
            ax.set_ylabel("Loss", fontsize=10, fontweight="bold")
            ax.set_title(f"{name}: Training Curves", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        plt.suptitle(
            "Reinforced Smoothing: Neural Network Training Results",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig("reinforced_smoothing_results.png", dpi=300, bbox_inches="tight")
        # plt.show()

    @staticmethod
    def plot_derivative_analysis(models_dict, x_test, colors):
        """Plot derivative analysis to visualize smoothness."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x_test_grad = x_test.clone().requires_grad_(True)

        for idx, (name, model) in enumerate(models_dict.items()):
            model.eval()
            y_pred = model(x_test_grad)

            # Compute first derivative
            dy_dx = torch.autograd.grad(
                outputs=y_pred,
                inputs=x_test_grad,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute second derivative
            d2y_dx2 = torch.autograd.grad(
                outputs=dy_dx,
                inputs=x_test_grad,
                grad_outputs=torch.ones_like(dy_dx),
                retain_graph=True,
            )[0]

            axes[idx].plot(
                x_test.detach().numpy(),
                dy_dx.detach().numpy(),
                linewidth=2,
                color=colors[idx],
                label="1st Derivative",
            )
            axes[idx].plot(
                x_test.detach().numpy(),
                d2y_dx2.detach().numpy(),
                linewidth=2,
                linestyle="--",
                color=colors[idx],
                alpha=0.6,
                label="2nd Derivative",
            )

            axes[idx].set_title(
                f"{name}\nDerivative Analysis", fontsize=12, fontweight="bold"
            )
            axes[idx].set_xlabel("x", fontsize=11)
            axes[idx].set_ylabel("Derivative Value", fontsize=11)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(
            "Smoothness Analysis: Model Derivatives", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("derivative_analysis.png", dpi=300, bbox_inches="tight")
        # plt.show()

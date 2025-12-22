# Reinforced Smoothing: Neural Network with Custom Loss Functions

A comprehensive implementation of neural networks with various smoothness regularization techniques for scientific machine learning applications.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements and compares multiple approaches to learning smooth functions from noisy data, a critical challenge in scientific computing and physics-informed machine learning.

### Key Features

- **Custom Loss Functions**: Smoothness, curvature, physics-informed, and total variation losses
- **Advanced Architectures**: Residual networks, Fourier features, adaptive depth
- **Uncertainty Quantification**: Bayesian NNs, MC Dropout, and ensemble methods
- **Comprehensive Evaluation**: Ablation studies and method comparisons
- **Professional Visualization**: Publication-ready plots and analysis tools

## 📁 Project Structure

```
reinforced-smoothing/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_generator.py         # Noisy data generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── smooth_nn.py              # Standard MLP
│   │   ├── residual_smooth_nn.py     # ResNet variants
│   │   └── bayesian_smooth_nn.py     # Uncertainty quantification
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── smoothness_loss.py        # Derivative penalties
│   │   ├── curvature_loss.py         # Geometric smoothness
│   │   └── physics_informed_loss.py  # PDE constraints
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                # Training loops
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py             # Plotting utilities
├── experiments/
│   ├── __init__.py
│   ├── basic_experiment.py           # Core experiment runner
│   ├── ablation_study.py             # Component analysis
│   └── comparison_study.py           # Method comparison
├── configs/
│   └── default_config.yaml           # Configuration files
├── results/                          # Generated outputs
├── tests/                            # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MrAsimZahid/reinforced-smoothing.git
cd reinforced-smoothing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run basic experiment with default settings
python experiments/basic_experiment.py

# Run with custom configuration
python experiments/basic_experiment.py --config configs/custom.yaml

# Quick test with single lambda
python experiments/basic_experiment.py --lambda 0.01 --epochs 3000

# Fast testing mode
python experiments/basic_experiment.py --quick
```

## 📊 Experiments

### 1. Basic Experiment

Compares different smoothness penalties (λ = 0, 0.001, 0.01):

```bash
python experiments/basic_experiment.py
```

**Output**: 
- Trained models with different λ values
- Comparison visualizations
- Training logs and metrics

### 2. Ablation Study

Systematically tests each component:

```bash
python experiments/ablation_study.py
```

**Tests**:
- Network depth (1-4 layers)
- Activation functions (Tanh, ReLU, ELU, Sigmoid)
- Derivative orders (1st vs 2nd)
- Lambda sensitivity (0.0001 - 0.1)

### 3. Comparison Study

Compares all methods comprehensively:

```bash
python experiments/comparison_study.py
```

**Compares**:
- Loss functions (Smoothness, Curvature, TV, Physics-Informed)
- Architectures (MLP, ResNet, Fourier Features, Adaptive)
- Uncertainty methods (Bayesian, MC Dropout, Ensemble)

## 🧪 Implemented Methods

### Loss Functions

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| **Smoothness Loss** | Penalizes derivatives | `smoothness_weight`, `derivative_order` |
| **Curvature Loss** | Geometric curvature penalty | `curvature_weight` |
| **Total Variation** | Piecewise smoothness | `tv_weight` |
| **Physics-Informed** | Enforces PDE: y'' + y = 0 | `pde_weight` |
| **Adaptive Curvature** | Density-aware smoothing | `base_weight`, `adaptation_rate` |
| **Multi-Objective** | Dynamic weight balancing | `data_weight`, `pde_weight` |

### Neural Network Architectures

| Architecture | Description | Advantages |
|--------------|-------------|------------|
| **Standard MLP** | Fully connected layers | Simple, fast |
| **Residual Network** | Skip connections | Better gradient flow |
| **Dense Residual** | Feature concatenation | Rich representations |
| **Adaptive Depth** | Learned gating | Variable effective depth |
| **Fourier Features** | Sinusoidal encoding | Excellent for periodic data |

### Uncertainty Quantification

| Method | Description | Computational Cost |
|--------|-------------|-------------------|
| **Bayesian NN** | Variational inference | High (sampling) |
| **MC Dropout** | Test-time dropout | Medium (multiple passes) |
| **Ensemble** | Multiple models | High (N models) |

## 📈 Results

### Smoothness Comparison

With λ = 0.01, models achieve:
- **Test MSE**: ~0.04 (vs 0.06 without smoothing)
- **Smoothness**: 70% reduction in derivative variance
- **Generalization**: Better extrapolation beyond training domain

### Uncertainty Quantification

Bayesian methods provide:
- **Calibrated uncertainty** in data-sparse regions
- **Confidence intervals** for predictions
- **Risk-aware** decision making

### Architecture Comparison

| Architecture | Test MSE | Training Time | Parameters |
|--------------|----------|---------------|------------|
| Standard MLP | 0.0412 | 3.2s | 3.1K |
| ResNet | 0.0398 | 4.1s | 4.8K |
| Fourier Features | 0.0365 | 4.5s | 5.2K |

## 🎨 Visualization Examples

The project generates publication-quality visualizations:

1. **Comparison Plots**: Model predictions vs ground truth
2. **Training Curves**: Loss evolution over epochs
3. **Derivative Analysis**: Smoothness characteristics
4. **Uncertainty Bands**: Confidence intervals
5. **Ablation Results**: Component-wise performance

## 🔧 Configuration

Use YAML files for reproducible experiments:

```yaml
# configs/custom_config.yaml
data:
  n_train_points: 40
  noise_std: 0.2
  seed: 42

model:
  hidden_layers: [32, 32, 32]
  activation: "tanh"

loss:
  smoothness_weights: [0.0, 0.001, 0.01]
  derivative_order: 2

training:
  learning_rate: 0.01
  epochs: 5000
```

## 📝 Command Line Options

### Basic Experiment

```bash
# Configuration
--config PATH          # Path to YAML config
--lambda FLOAT         # Single lambda to test
--epochs INT          # Training epochs
--learning-rate FLOAT # Learning rate
--seed INT            # Random seed

# Modes
--quick               # Fast test mode
--no-save             # Don't save models/logs
```

## 🧮 Mathematical Formulation

### Smoothness Loss

Total Loss = MSE + λ × Smoothness Penalty

```
L_total = ||y_pred - y_true||² + λ × ||d²y/dx²||²
```

### Physics-Informed Loss

For sine wave, enforce: y'' + y = 0

```
L_total = ||y_pred - y_true||² + λ_pde × ||y'' + y||²
```

### Curvature Loss

Geometric curvature: κ = |y''| / (1 + (y')²)^(3/2)

```
L_total = ||y_pred - y_true||² + λ_curv × ||κ||²
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Test coverage:

```bash
pytest --cov=src tests/
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{reinforced_smoothing,
  title={Reinforced Smoothing: Neural Networks with Custom Loss Functions},
  author={Asim Zahid},
  year={2025},
  url={https://github.com/MrAsimZahid/reinforced-smoothing}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Scientific ML community for inspiration
- Physics-Informed Neural Networks (PINNs) literature

## 📞 Contact

- **Author**: Asim Zahid
- **Email**: asimzahid02@gmail.com
- **GitHub**: [@MrAsimZahid](https://github.com/MrAsimZahid)

## 🔗 Related Projects

- [Physics-Informed Neural Networks](https://github.com/maziarraissi/PINNs)
- [Neural Operators](https://github.com/neuraloperator/neuraloperator)
- [DeepXDE](https://github.com/lululxvi/deepxde)

---

**Made with ❤️ for scientific machine learning**
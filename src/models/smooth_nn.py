import torch.nn as nn

class SmoothNN(nn.Module):
    """Neural network with configurable architecture for function approximation."""

    def __init__(self, hidden_layers=[32, 32, 32], activation=nn.Tanh()):
        """
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use
        """
        super(SmoothNN, self).__init__()

        layers = []
        input_size = 1

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation)
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.network(x)

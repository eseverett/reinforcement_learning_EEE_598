import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN_Model(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int]) -> None:
        """
        DQN model with adjustable depth and width via a list of hidden layer sizes.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_layers (List[int]): List defining the width of each hidden layer.
            Example: [128, 64, 32] -> 3 hidden layers.
        """
        super().__init__()

        # Combine input, hidden, and output dimensions into one sequence
        layer_sizes = [input_dim] + hidden_layers + [output_dim]

        # Build all linear layers dynamically
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply ReLU to all but the last layer
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

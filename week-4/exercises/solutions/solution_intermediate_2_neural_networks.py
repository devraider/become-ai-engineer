"""
Week 4 Exercise 2 (Intermediate): Neural Networks - SOLUTIONS
============================================================
"""

import torch
import torch.nn as nn
from typing import Tuple, List


def create_linear_layer(in_features: int, out_features: int) -> nn.Linear:
    """Create a linear (fully connected) layer."""
    return nn.Linear(in_features, out_features)


def create_sequential_model(layer_sizes: List[int]) -> nn.Sequential:
    """Create a sequential model with ReLU activations."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        # Add ReLU after all but the last layer
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SimpleClassifier(nn.Module):
    """Simple classifier neural network."""

    def __init__(
        self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron (MLP)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_activation_function(name: str) -> nn.Module:
    """Return the activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def apply_weight_init(model: nn.Module, init_type: str = "xavier") -> None:
    """Apply weight initialization to all linear layers."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight)
            elif init_type == "normal":
                nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """Freeze specific layers (disable gradient computation)."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if name.startswith(layer_name):
                param.requires_grad = False


def forward_with_intermediates(
    model: nn.Sequential, x: torch.Tensor
) -> List[torch.Tensor]:
    """Run forward pass and collect intermediate activations."""
    intermediates = [x.clone()]
    for layer in model:
        x = layer(x)
        intermediates.append(x.clone())
    return intermediates


def compute_output_shape(
    model: nn.Module, input_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute the output shape of a model given input shape."""
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        output = model(dummy_input)
    return tuple(output.shape[1:])

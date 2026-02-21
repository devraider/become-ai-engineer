"""
Week 4 Exercise 2 (Intermediate): Neural Networks
=================================================

Build neural networks using PyTorch nn.Module.

Run this file:
    python exercise_intermediate_2_neural_networks.py

Run tests:
    python -m pytest tests/test_exercise_intermediate_2_neural_networks.py -v
"""

import torch
import torch.nn as nn
from typing import Tuple, List


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def create_linear_layer(in_features: int, out_features: int) -> nn.Linear:
    """
    Task 1: Create a linear (fully connected) layer.

    Args:
        in_features: Number of input features
        out_features: Number of output features

    Returns:
        nn.Linear layer
    """
    # TODO: Implement
    pass


def create_sequential_model(layer_sizes: List[int]) -> nn.Sequential:
    """
    Task 2: Create a sequential model with ReLU activations.

    Args:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]

    Returns:
        nn.Sequential model with Linear layers and ReLU between them
        (no activation after the last layer)

    Example:
        layer_sizes = [10, 32, 16, 2]
        Creates: Linear(10,32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,2)
    """
    # TODO: Implement
    pass


class SimpleClassifier(nn.Module):
    """
    Task 3: Implement a simple classifier neural network.

    Architecture:
        - Linear layer: input_size -> hidden_size
        - ReLU activation
        - Dropout with given probability
        - Linear layer: hidden_size -> num_classes
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # TODO: Initialize layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        pass


class MultiLayerPerceptron(nn.Module):
    """
    Task 4: Implement a multi-layer perceptron (MLP).

    Architecture:
        - Multiple hidden layers with configurable sizes
        - ReLU activation after each hidden layer
        - BatchNorm after each linear layer (before activation)
        - Optional dropout after each activation
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: Initialize layers
        # Hint: Use nn.ModuleList for variable number of layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        # TODO: Implement
        pass


def count_parameters(model: nn.Module) -> int:
    """
    Task 5: Count the total number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters

    Hints:
        - Iterate over model.parameters()
        - Use .numel() to count elements
        - Only count if requires_grad is True
    """
    # TODO: Implement
    pass


def get_activation_function(name: str) -> nn.Module:
    """
    Task 6: Return the activation function by name.

    Args:
        name: One of 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'gelu'

    Returns:
        Corresponding nn.Module activation function

    Raises:
        ValueError: If name is not recognized
    """
    # TODO: Implement
    pass


def apply_weight_init(model: nn.Module, init_type: str = "xavier") -> None:
    """
    Task 7: Apply weight initialization to all linear layers.

    Args:
        model: PyTorch model
        init_type: 'xavier', 'kaiming', or 'normal'

    Hints:
        - Iterate over model.modules()
        - Check if module is nn.Linear
        - Use nn.init functions
    """
    # TODO: Implement
    pass


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Task 8: Freeze specific layers (disable gradient computation).

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze

    Hints:
        - Use model.named_parameters()
        - Set requires_grad = False for matching parameters
    """
    # TODO: Implement
    pass


def forward_with_intermediates(
    model: nn.Sequential, x: torch.Tensor
) -> List[torch.Tensor]:
    """
    Task 9: Run forward pass and collect intermediate activations.

    Args:
        model: Sequential model
        x: Input tensor

    Returns:
        List of tensors: [input, after_layer1, after_layer2, ..., output]
    """
    # TODO: Implement
    pass


def compute_output_shape(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Task 10: Compute the output shape of a model given input shape.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (excluding batch dimension)

    Returns:
        Shape of output tensor (excluding batch dimension)

    Hints:
        - Create a dummy tensor with batch size 1
        - Run forward pass
        - Return output shape without batch dimension
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Exercise 2: Neural Networks")
    print("=" * 60)

    print("\n1. Creating linear layer...")
    layer = create_linear_layer(10, 5)
    if layer:
        print(f"   Layer: {layer}")

    print("\n2. Creating sequential model...")
    seq_model = create_sequential_model([784, 256, 128, 10])
    if seq_model:
        print(f"   Model: {seq_model}")

    print("\n3. Simple classifier...")
    clf = SimpleClassifier(784, 128, 10, dropout=0.2)
    if hasattr(clf, "forward"):
        x = torch.randn(32, 784)
        try:
            out = clf(x)
            print(f"   Input: {x.shape} -> Output: {out.shape}")
        except Exception as e:
            print(f"   Not implemented yet: {e}")

    print("\n4. MLP...")
    mlp = MultiLayerPerceptron(784, [256, 128], 10, dropout=0.1)
    if hasattr(mlp, "forward"):
        try:
            out = mlp(torch.randn(16, 784))
            print(f"   Output shape: {out.shape}")
        except Exception as e:
            print(f"   Not implemented yet: {e}")

    print("\n5. Counting parameters...")
    model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    count = count_parameters(model)
    if count:
        print(f"   Parameters: {count}")

    print("\nComplete all TODOs and run tests to verify!")

"""
Week 4 Exercise 1 (Basic): PyTorch Tensors - SOLUTIONS
======================================================
"""

import torch
import numpy as np
from typing import Tuple


def create_tensor_from_list(data: list) -> torch.Tensor:
    """Create a tensor from a Python list."""
    return torch.tensor(data)


def create_zeros_tensor(rows: int, cols: int) -> torch.Tensor:
    """Create a tensor filled with zeros."""
    return torch.zeros(rows, cols)


def create_random_tensor(shape: Tuple[int, ...], seed: int = 42) -> torch.Tensor:
    """Create a tensor with random values from standard normal distribution."""
    torch.manual_seed(seed)
    return torch.randn(shape)


def tensor_from_numpy(np_array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    return torch.from_numpy(np_array)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.numpy()


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """Get information about a tensor."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "device": tensor.device,
        "requires_grad": tensor.requires_grad,
    }


def reshape_tensor(tensor: torch.Tensor, new_shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshape a tensor to a new shape."""
    return tensor.view(new_shape)


def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Perform matrix multiplication."""
    return a @ b


def element_wise_operations(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Perform element-wise operations."""
    return {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b,
    }


def compute_statistics(tensor: torch.Tensor) -> dict:
    """Compute basic statistics of a tensor."""
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "sum": tensor.sum().item(),
    }


def broadcasting_example(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Demonstrate broadcasting by adding a vector to each row of a matrix."""
    return matrix + vector

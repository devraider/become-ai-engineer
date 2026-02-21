"""
Week 4 Exercise 1 (Basic): PyTorch Tensors
==========================================

Learn to create and manipulate PyTorch tensors.

Run this file:
    python exercise_basic_1_tensors.py

Run tests:
    python -m pytest tests/test_exercise_basic_1_tensors.py -v
"""

import torch
import numpy as np
from typing import Tuple


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def create_tensor_from_list(data: list) -> torch.Tensor:
    """
    Task 1: Create a tensor from a Python list.

    Args:
        data: A Python list of numbers

    Returns:
        torch.Tensor created from the list

    Example:
        >>> t = create_tensor_from_list([1, 2, 3])
        >>> t
        tensor([1, 2, 3])
    """
    # TODO: Implement
    pass


def create_zeros_tensor(rows: int, cols: int) -> torch.Tensor:
    """
    Task 2: Create a tensor filled with zeros.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
        Zero tensor of shape (rows, cols)
    """
    # TODO: Implement
    pass


def create_random_tensor(shape: Tuple[int, ...], seed: int = 42) -> torch.Tensor:
    """
    Task 3: Create a tensor with random values from standard normal distribution.

    Args:
        shape: Tuple defining tensor shape
        seed: Random seed for reproducibility

    Returns:
        Random tensor of given shape

    Hints:
        - Use torch.manual_seed() for reproducibility
        - Use torch.randn() for standard normal
    """
    # TODO: Implement
    pass


def tensor_from_numpy(np_array: np.ndarray) -> torch.Tensor:
    """
    Task 4: Convert a NumPy array to a PyTorch tensor.

    Args:
        np_array: NumPy array

    Returns:
        PyTorch tensor

    Hints:
        - Use torch.from_numpy()
        - The tensor shares memory with the array!
    """
    # TODO: Implement
    pass


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Task 5: Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        NumPy array

    Hints:
        - Use .numpy() method
        - Tensor must be on CPU
    """
    # TODO: Implement
    pass


def get_tensor_info(tensor: torch.Tensor) -> dict:
    """
    Task 6: Get information about a tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary with keys:
        - 'shape': tensor shape as tuple
        - 'dtype': tensor data type
        - 'device': tensor device (cpu/cuda)
        - 'requires_grad': whether gradient tracking is enabled
    """
    # TODO: Implement
    pass


def reshape_tensor(tensor: torch.Tensor, new_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Task 7: Reshape a tensor to a new shape.

    Args:
        tensor: Input tensor
        new_shape: Target shape

    Returns:
        Reshaped tensor

    Hints:
        - Use .view() or .reshape()
        - Total elements must match
    """
    # TODO: Implement
    pass


def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Task 8: Perform matrix multiplication.

    Args:
        a: First matrix (m x n)
        b: Second matrix (n x p)

    Returns:
        Result matrix (m x p)

    Hints:
        - Use @ operator or torch.matmul()
    """
    # TODO: Implement
    pass


def element_wise_operations(a: torch.Tensor, b: torch.Tensor) -> dict:
    """
    Task 9: Perform element-wise operations.

    Args:
        a: First tensor
        b: Second tensor (same shape as a)

    Returns:
        Dictionary with keys:
        - 'add': a + b
        - 'subtract': a - b
        - 'multiply': a * b
        - 'divide': a / b
    """
    # TODO: Implement
    pass


def compute_statistics(tensor: torch.Tensor) -> dict:
    """
    Task 10: Compute basic statistics of a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary with keys:
        - 'mean': mean value
        - 'std': standard deviation
        - 'min': minimum value
        - 'max': maximum value
        - 'sum': sum of all elements

    Hints:
        - Use .item() to convert single-element tensors to Python numbers
    """
    # TODO: Implement
    pass


def broadcasting_example(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Task 11: Demonstrate broadcasting by adding a vector to each row of a matrix.

    Args:
        vector: 1D tensor of shape (n,)
        matrix: 2D tensor of shape (m, n)

    Returns:
        Result of broadcasting addition (m, n)

    Example:
        vector = [1, 2, 3]
        matrix = [[1, 1, 1],
                  [2, 2, 2]]
        result = [[2, 3, 4],
                  [3, 4, 5]]
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Exercise 1: PyTorch Tensors")
    print("=" * 60)

    # Test your implementations
    print("\n1. Creating tensor from list...")
    t = create_tensor_from_list([1, 2, 3, 4, 5])
    if t is not None:
        print(f"   Result: {t}")

    print("\n2. Creating zeros tensor...")
    z = create_zeros_tensor(3, 4)
    if z is not None:
        print(f"   Shape: {z.shape}")

    print("\n3. Creating random tensor...")
    r = create_random_tensor((2, 3), seed=42)
    if r is not None:
        print(f"   Shape: {r.shape}, Mean: {r.mean():.4f}")

    print("\n4. NumPy conversion...")
    np_arr = np.array([1.0, 2.0, 3.0])
    tensor = tensor_from_numpy(np_arr)
    if tensor is not None:
        print(f"   Tensor: {tensor}")

    print("\n5. Tensor info...")
    info = get_tensor_info(torch.randn(3, 4))
    if info:
        print(f"   Info: {info}")

    print("\n6. Matrix multiplication...")
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    c = matrix_multiply(a, b)
    if c is not None:
        print(f"   (2x3) @ (3x4) = {c.shape}")

    print("\n7. Statistics...")
    stats = compute_statistics(torch.randn(100))
    if stats:
        print(f"   Mean: {stats.get('mean', 'N/A'):.4f}")

    print("\nComplete all TODOs and run tests to verify!")

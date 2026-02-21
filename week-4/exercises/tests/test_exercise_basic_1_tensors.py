"""
Tests for Week 4 Exercise 1: PyTorch Tensors
"""

import pytest
import torch
import numpy as np
from exercise_basic_1_tensors import (
    create_tensor_from_list,
    create_zeros_tensor,
    create_random_tensor,
    tensor_from_numpy,
    tensor_to_numpy,
    get_tensor_info,
    reshape_tensor,
    matrix_multiply,
    element_wise_operations,
    compute_statistics,
    broadcasting_example,
)


class TestCreateTensorFromList:
    def test_simple_list(self):
        result = create_tensor_from_list([1, 2, 3])
        assert torch.is_tensor(result)
        assert result.tolist() == [1, 2, 3]

    def test_nested_list(self):
        result = create_tensor_from_list([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_float_list(self):
        result = create_tensor_from_list([1.5, 2.5, 3.5])
        assert result.dtype == torch.float32 or result.dtype == torch.float64


class TestCreateZerosTensor:
    def test_shape(self):
        result = create_zeros_tensor(3, 4)
        assert result.shape == (3, 4)

    def test_all_zeros(self):
        result = create_zeros_tensor(2, 2)
        assert torch.all(result == 0)

    def test_dtype(self):
        result = create_zeros_tensor(1, 1)
        assert result.dtype == torch.float32


class TestCreateRandomTensor:
    def test_shape(self):
        result = create_random_tensor((2, 3, 4), seed=42)
        assert result.shape == (2, 3, 4)

    def test_reproducibility(self):
        t1 = create_random_tensor((5,), seed=42)
        t2 = create_random_tensor((5,), seed=42)
        assert torch.allclose(t1, t2)

    def test_different_seeds(self):
        t1 = create_random_tensor((5,), seed=42)
        t2 = create_random_tensor((5,), seed=123)
        assert not torch.allclose(t1, t2)


class TestTensorFromNumpy:
    def test_conversion(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = tensor_from_numpy(arr)
        assert torch.is_tensor(result)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_shape_preserved(self):
        arr = np.random.randn(3, 4, 5)
        result = tensor_from_numpy(arr)
        assert result.shape == (3, 4, 5)


class TestTensorToNumpy:
    def test_conversion(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestGetTensorInfo:
    def test_info_keys(self):
        tensor = torch.randn(3, 4, requires_grad=True)
        info = get_tensor_info(tensor)
        assert "shape" in info
        assert "dtype" in info
        assert "device" in info
        assert "requires_grad" in info

    def test_info_values(self):
        tensor = torch.randn(3, 4, requires_grad=True)
        info = get_tensor_info(tensor)
        assert info["shape"] == (3, 4) or info["shape"] == torch.Size([3, 4])
        assert info["requires_grad"] == True


class TestReshapeTensor:
    def test_reshape(self):
        tensor = torch.randn(12)
        result = reshape_tensor(tensor, (3, 4))
        assert result.shape == (3, 4)

    def test_reshape_with_negative(self):
        tensor = torch.randn(2, 6)
        result = reshape_tensor(tensor, (-1, 3))
        assert result.shape == (4, 3)


class TestMatrixMultiply:
    def test_multiplication(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_multiply(a, b)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected)

    def test_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        result = matrix_multiply(a, b)
        assert result.shape == (3, 5)


class TestElementWiseOperations:
    def test_operations(self):
        a = torch.tensor([2.0, 4.0, 6.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        result = element_wise_operations(a, b)
        
        assert torch.allclose(result["add"], torch.tensor([3.0, 6.0, 9.0]))
        assert torch.allclose(result["subtract"], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(result["multiply"], torch.tensor([2.0, 8.0, 18.0]))
        assert torch.allclose(result["divide"], torch.tensor([2.0, 2.0, 2.0]))


class TestComputeStatistics:
    def test_statistics(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(tensor)
        
        assert abs(stats["mean"] - 3.0) < 0.01
        assert abs(stats["sum"] - 15.0) < 0.01
        assert abs(stats["min"] - 1.0) < 0.01
        assert abs(stats["max"] - 5.0) < 0.01


class TestBroadcastingExample:
    def test_broadcasting(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        matrix = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        result = broadcasting_example(vector, matrix)
        
        expected = torch.tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        assert torch.allclose(result, expected)

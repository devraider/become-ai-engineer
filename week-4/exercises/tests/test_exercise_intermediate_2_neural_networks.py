"""
Tests for Week 4 Exercise 2: Neural Networks
"""

import pytest
import torch
import torch.nn as nn
from exercise_intermediate_2_neural_networks import (
    create_linear_layer,
    create_sequential_model,
    SimpleClassifier,
    MultiLayerPerceptron,
    count_parameters,
    get_activation_function,
    apply_weight_init,
    freeze_layers,
    forward_with_intermediates,
    compute_output_shape,
)


class TestCreateLinearLayer:
    def test_layer_creation(self):
        layer = create_linear_layer(10, 5)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 10
        assert layer.out_features == 5

    def test_forward_pass(self):
        layer = create_linear_layer(10, 5)
        x = torch.randn(32, 10)
        output = layer(x)
        assert output.shape == (32, 5)


class TestCreateSequentialModel:
    def test_model_creation(self):
        model = create_sequential_model([10, 32, 16, 2])
        assert isinstance(model, nn.Sequential)

    def test_forward_pass(self):
        model = create_sequential_model([10, 32, 16, 2])
        x = torch.randn(8, 10)
        output = model(x)
        assert output.shape == (8, 2)

    def test_contains_relu(self):
        model = create_sequential_model([10, 32, 2])
        has_relu = any(isinstance(m, nn.ReLU) for m in model.modules())
        assert has_relu


class TestSimpleClassifier:
    def test_initialization(self):
        model = SimpleClassifier(784, 128, 10, dropout=0.2)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        model = SimpleClassifier(784, 128, 10, dropout=0.2)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_training_mode(self):
        model = SimpleClassifier(784, 128, 10, dropout=0.5)
        model.train()
        x = torch.randn(32, 784)
        # Should run without error
        output = model(x)


class TestMultiLayerPerceptron:
    def test_initialization(self):
        model = MultiLayerPerceptron(784, [256, 128], 10, dropout=0.1)
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        model = MultiLayerPerceptron(784, [256, 128], 10, dropout=0.1)
        x = torch.randn(16, 784)
        output = model(x)
        assert output.shape == (16, 10)

    def test_single_hidden(self):
        model = MultiLayerPerceptron(100, [50], 10)
        x = torch.randn(8, 100)
        output = model(x)
        assert output.shape == (8, 10)


class TestCountParameters:
    def test_simple_model(self):
        model = nn.Linear(10, 5)
        count = count_parameters(model)
        # 10 * 5 weights + 5 biases = 55
        assert count == 55

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        count = count_parameters(model)
        # (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        assert count == 325


class TestGetActivationFunction:
    def test_relu(self):
        act = get_activation_function("relu")
        assert isinstance(act, nn.ReLU)

    def test_sigmoid(self):
        act = get_activation_function("sigmoid")
        assert isinstance(act, nn.Sigmoid)

    def test_tanh(self):
        act = get_activation_function("tanh")
        assert isinstance(act, nn.Tanh)

    def test_invalid(self):
        with pytest.raises(ValueError):
            get_activation_function("invalid")


class TestApplyWeightInit:
    def test_xavier_init(self):
        model = nn.Linear(10, 5)
        old_weights = model.weight.clone()
        apply_weight_init(model, "xavier")
        # Weights should be modified
        assert not torch.allclose(model.weight, old_weights)

    def test_kaiming_init(self):
        model = nn.Linear(10, 5)
        apply_weight_init(model, "kaiming")
        # Should run without error


class TestFreezeLayers:
    def test_freeze_specific_layers(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        # Freeze first layer
        freeze_layers(model, ["0"])
        
        # Check that first layer is frozen
        for name, param in model.named_parameters():
            if name.startswith("0"):
                assert not param.requires_grad


class TestForwardWithIntermediates:
    def test_intermediates(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        x = torch.randn(1, 10)
        intermediates = forward_with_intermediates(model, x)
        
        assert len(intermediates) >= 2
        assert intermediates[0].shape == x.shape
        assert intermediates[-1].shape == (1, 2)


class TestComputeOutputShape:
    def test_simple_model(self):
        model = nn.Linear(10, 5)
        shape = compute_output_shape(model, (10,))
        assert shape == (5,)

    def test_sequential_model(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        shape = compute_output_shape(model, (100,))
        assert shape == (10,)

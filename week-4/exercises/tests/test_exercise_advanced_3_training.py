"""
Tests for Week 4 Exercise 3: Training Pipelines
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
import os
from exercise_advanced_3_training import (
    get_device,
    create_data_loaders,
    CustomDataset,
    train_one_epoch,
    validate,
    train_model,
    save_checkpoint,
    load_checkpoint,
    compute_class_weights,
    learning_rate_schedule,
)


class TestGetDevice:
    def test_returns_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_valid_device_type(self):
        device = get_device()
        assert device.type in ["cpu", "cuda", "mps"]


class TestCreateDataLoaders:
    def test_creates_loaders(self):
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))

        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_batch_size(self):
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))

        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=16
        )

        batch = next(iter(train_loader))
        assert batch[0].shape[0] == 16


class TestCustomDataset:
    def test_length(self):
        data = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 100)
        ds = CustomDataset(data, labels)
        assert len(ds) == 100

    def test_getitem(self):
        data = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 100)
        ds = CustomDataset(data, labels)

        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10,)

    def test_transform(self):
        data = np.ones((10, 5), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int64)
        transform = lambda x: x * 2

        ds = CustomDataset(data, labels, transform=transform)
        x, _ = ds[0]
        assert torch.allclose(x, torch.tensor([2.0] * 5))


class TestTrainOneEpoch:
    def test_returns_loss(self):
        model = nn.Linear(10, 2)
        train_loader = DataLoader(
            TensorDataset(torch.randn(64, 10), torch.randint(0, 2, (64,))),
            batch_size=16,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss >= 0


class TestValidate:
    def test_returns_loss_and_accuracy(self):
        model = nn.Linear(10, 2)
        val_loader = DataLoader(
            TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))),
            batch_size=16,
        )
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, accuracy = validate(model, val_loader, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestTrainModel:
    def test_returns_history(self):
        model = nn.Linear(10, 2)
        train_loader = DataLoader(
            TensorDataset(torch.randn(64, 10), torch.randint(0, 2, (64,))),
            batch_size=16,
        )
        val_loader = DataLoader(
            TensorDataset(torch.randn(16, 10), torch.randint(0, 2, (16,))),
            batch_size=16,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=3,
            early_stopping_patience=2,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_accuracy" in history


class TestCheckpoints:
    def test_save_and_load(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(model, optimizer, epoch=5, loss=0.123, filepath=filepath)

            new_model = nn.Linear(10, 5)
            checkpoint = load_checkpoint(filepath, new_model)

            assert checkpoint["epoch"] == 5
            assert abs(checkpoint["loss"] - 0.123) < 0.01


class TestComputeClassWeights:
    def test_imbalanced_classes(self):
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels)

        # Class 1 should have higher weight
        assert weights[1] > weights[0]

    def test_balanced_classes(self):
        labels = torch.tensor([0, 0, 1, 1])
        weights = compute_class_weights(labels)

        # Weights should be similar
        assert abs(weights[0] - weights[1]) < 0.5


class TestLearningRateSchedule:
    def test_initial_lr(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        lr = learning_rate_schedule(
            optimizer, epoch=0, initial_lr=0.1, decay_rate=0.1, decay_epochs=[30, 60]
        )
        assert abs(lr - 0.1) < 0.001

    def test_lr_decay(self):
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        lr = learning_rate_schedule(
            optimizer, epoch=35, initial_lr=0.1, decay_rate=0.1, decay_epochs=[30, 60]
        )
        assert abs(lr - 0.01) < 0.001

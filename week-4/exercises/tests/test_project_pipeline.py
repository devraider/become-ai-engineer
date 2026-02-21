"""
Tests for Week 4 Project: Neural Network Image Classifier
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from project_pipeline import (
    generate_synthetic_data,
    prepare_data,
    ImageClassifier,
    Trainer,
    evaluate_model,
    predict,
    run_experiment,
)


class TestGenerateSyntheticData:
    def test_shapes(self):
        X, y = generate_synthetic_data(n_samples=100, n_features=784, n_classes=10)
        assert X.shape == (100, 784)
        assert y.shape == (100,)

    def test_labels_range(self):
        X, y = generate_synthetic_data(n_samples=100, n_features=784, n_classes=10)
        assert y.min() >= 0
        assert y.max() < 10

    def test_features_normalized(self):
        X, y = generate_synthetic_data(n_samples=100, n_features=784, n_classes=10)
        assert X.min() >= 0
        assert X.max() <= 1

    def test_reproducibility(self):
        X1, y1 = generate_synthetic_data(seed=42)
        X2, y2 = generate_synthetic_data(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestPrepareData:
    def test_creates_data_loaders(self):
        X, y = generate_synthetic_data(n_samples=100, n_features=784, n_classes=10)
        train_loader, val_loader = prepare_data(X, y, train_ratio=0.8, batch_size=16)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_split_ratio(self):
        X, y = generate_synthetic_data(n_samples=100, n_features=784, n_classes=10)
        train_loader, val_loader = prepare_data(X, y, train_ratio=0.8, batch_size=1)

        train_size = sum(1 for _ in train_loader)
        val_size = sum(1 for _ in val_loader)

        assert train_size == 80
        assert val_size == 20


class TestImageClassifier:
    def test_initialization(self):
        model = ImageClassifier(
            input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout=0.2
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        model = ImageClassifier(
            input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout=0.2
        )
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_different_architectures(self):
        # Small model
        model1 = ImageClassifier(input_size=100, hidden_sizes=[50], num_classes=5)
        x1 = torch.randn(8, 100)
        assert model1(x1).shape == (8, 5)

        # Large model
        model2 = ImageClassifier(
            input_size=1024, hidden_sizes=[512, 256, 128, 64], num_classes=100
        )
        x2 = torch.randn(8, 1024)
        assert model2(x2).shape == (8, 100)


class TestTrainer:
    @pytest.fixture
    def setup_trainer(self):
        model = ImageClassifier(input_size=100, hidden_sizes=[50], num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")
        return Trainer(model, criterion, optimizer, device)

    def test_train_epoch(self, setup_trainer):
        trainer = setup_trainer
        X = torch.randn(64, 100)
        y = torch.randint(0, 10, (64,))
        train_loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=16)

        loss = trainer.train_epoch(train_loader)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate(self, setup_trainer):
        trainer = setup_trainer
        X = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        val_loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=16)

        loss, accuracy = trainer.validate(val_loader)
        assert isinstance(loss, float)
        assert 0 <= accuracy <= 1

    def test_fit(self, setup_trainer):
        trainer = setup_trainer
        X_train = torch.randn(64, 100)
        y_train = torch.randint(0, 10, (64,))
        X_val = torch.randn(16, 100)
        y_val = torch.randint(0, 10, (16,))

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), batch_size=16
        )
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val), batch_size=16
        )

        history = trainer.fit(
            train_loader, val_loader, num_epochs=3, early_stopping_patience=2
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_accuracy" in history


class TestEvaluateModel:
    def test_returns_metrics(self):
        model = ImageClassifier(input_size=100, hidden_sizes=[50], num_classes=10)
        X = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        test_loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=16)

        metrics = evaluate_model(model, test_loader, torch.device("cpu"))

        assert "accuracy" in metrics or "Accuracy" in str(metrics)


class TestPredict:
    def test_predictions(self):
        model = ImageClassifier(input_size=100, hidden_sizes=[50], num_classes=10)
        X = torch.randn(16, 100)

        predictions, probabilities = predict(model, X, torch.device("cpu"))

        assert predictions.shape == (16,)
        assert probabilities.shape == (16, 10)
        assert torch.all(predictions >= 0) and torch.all(predictions < 10)


class TestRunExperiment:
    def test_experiment_runs(self):
        results = run_experiment(
            n_samples=200,
            n_features=100,
            n_classes=5,
            hidden_sizes=[50],
            num_epochs=3,
            early_stopping_patience=2,
        )

        assert results is not None
        assert "history" in results
        assert "final_metrics" in results

    def test_reproducibility(self):
        results1 = run_experiment(n_samples=100, n_features=50, num_epochs=2, seed=42)
        results2 = run_experiment(n_samples=100, n_features=50, num_epochs=2, seed=42)

        # Final metrics should be identical with same seed
        if results1 and results2:
            assert (
                results1["history"]["train_loss"] == results2["history"]["train_loss"]
            )

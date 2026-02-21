"""
Week 4 Project: Neural Network Image Classifier
================================================

Build a complete neural network classifier for image classification.
This project combines all concepts from Week 4.

Dataset: We'll use a synthetic dataset or MNIST-like data.

Run this file:
    python project_pipeline.py

Run tests:
    python -m pytest tests/test_project_pipeline.py -v
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# PART 1: Data Preparation
# =============================================================================


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 784,
    n_classes: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TASK 1: Generate synthetic classification data.

    Create random data that simulates image classification:
    - Features should be normalized (0-1 range)
    - Labels should be integers from 0 to n_classes-1

    Args:
        n_samples: Number of samples
        n_features: Number of features (e.g., 784 for 28x28 images)
        n_classes: Number of classes
        seed: Random seed

    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    # TODO: Implement
    pass


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    TASK 2: Prepare training and validation data loaders.

    Steps:
    1. Convert numpy arrays to tensors
    2. Split into train/validation sets
    3. Create DataLoaders with proper shuffling

    Args:
        X: Feature array
        y: Label array
        train_ratio: Fraction of data for training
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # TODO: Implement
    pass


# =============================================================================
# PART 2: Model Architecture
# =============================================================================


class ImageClassifier(nn.Module):
    """
    TASK 3: Implement the image classifier neural network.

    Architecture:
        Input (784) -> Linear -> BatchNorm -> ReLU -> Dropout
                    -> Linear -> BatchNorm -> ReLU -> Dropout
                    -> Linear -> BatchNorm -> ReLU -> Dropout
                    -> Linear (output)

    The network should be flexible enough to handle different:
    - Input sizes
    - Number of hidden layers
    - Hidden layer sizes
    - Dropout rates
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: Initialize layers
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # TODO: Implement
        pass


# =============================================================================
# PART 3: Training Infrastructure
# =============================================================================


class Trainer:
    """
    TASK 4: Implement the training class.

    This class should handle:
    - Model training
    - Validation
    - Early stopping
    - Checkpoint saving/loading
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        # TODO: Initialize trainer
        pass

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        # TODO: Implement
        pass

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        # TODO: Implement
        pass

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        # TODO: Implement
        pass

    def save_checkpoint(self, filepath: str, epoch: int, best_val_loss: float) -> None:
        """Save training checkpoint."""
        # TODO: Implement
        pass

    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint."""
        # TODO: Implement
        pass


# =============================================================================
# PART 4: Evaluation and Inference
# =============================================================================


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    TASK 5: Evaluate model performance.

    Compute:
    - Accuracy
    - Per-class accuracy
    - Average loss

    Returns:
        Dictionary with evaluation metrics
    """
    # TODO: Implement
    pass


def predict(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TASK 6: Make predictions on new data.

    Args:
        model: Trained model
        X: Input features
        device: Device

    Returns:
        Tuple of (predicted_classes, probabilities)
    """
    # TODO: Implement
    pass


# =============================================================================
# PART 5: Main Pipeline
# =============================================================================


def run_experiment(
    n_samples: int = 5000,
    n_features: int = 784,
    n_classes: int = 10,
    hidden_sizes: List[int] = [512, 256, 128],
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    num_epochs: int = 50,
    early_stopping_patience: int = 5,
    seed: int = 42,
) -> Dict:
    """
    TASK 7: Run the complete training experiment.

    Steps:
    1. Set random seeds for reproducibility
    2. Generate/load data
    3. Create model
    4. Train model
    5. Evaluate model
    6. Return results

    Returns:
        Dictionary with:
        - 'history': training history
        - 'final_metrics': evaluation metrics
        - 'model_info': model architecture info
    """
    # TODO: Implement complete pipeline
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Project: Neural Network Image Classifier")
    print("=" * 60)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # Run the experiment
    results = run_experiment(
        n_samples=2000,
        n_features=784,
        n_classes=10,
        hidden_sizes=[256, 128],
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=20,
        early_stopping_patience=5,
    )

    if results:
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)

        if "final_metrics" in results:
            metrics = results["final_metrics"]
            print(f"\nFinal Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        if "history" in results:
            history = results["history"]
            print(
                f"\nTraining completed in {len(history.get('train_loss', []))} epochs"
            )

        if "model_info" in results:
            info = results["model_info"]
            print(f"\nModel Info:")
            for k, v in info.items():
                print(f"  {k}: {v}")
    else:
        print("\nComplete all TODOs to run the experiment!")

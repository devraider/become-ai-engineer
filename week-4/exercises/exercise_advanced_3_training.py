"""
Week 4 Exercise 3 (Advanced): Training Pipelines
================================================

Implement complete training pipelines with PyTorch.

Run this file:
    python exercise_advanced_3_training.py

Run tests:
    python -m pytest tests/test_exercise_advanced_3_training.py -v
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def get_device() -> torch.device:
    """
    Task 1: Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device for the best available hardware
    """
    # TODO: Implement
    pass


def create_data_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Task 2: Create training and validation data loaders.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader)

    Hints:
        - Use TensorDataset to combine X and y
        - Shuffle training data, don't shuffle validation
    """
    # TODO: Implement
    pass


class CustomDataset(Dataset):
    """
    Task 3: Implement a custom Dataset class.

    This dataset should:
    - Store features and labels
    - Apply an optional transform to features
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        # TODO: Initialize
        pass

    def __len__(self) -> int:
        # TODO: Return dataset length
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Return (feature, label) at index
        pass


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Task 4: Train model for one epoch.

    Args:
        model: Neural network
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    # TODO: Implement training loop
    pass


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Task 5: Validate model on validation set.

    Args:
        model: Neural network
        val_loader: Validation data loader
        criterion: Loss function
        device: Device

    Returns:
        Tuple of (average_loss, accuracy)
    """
    # TODO: Implement validation loop
    pass


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    early_stopping_patience: int = 3,
) -> Dict[str, List[float]]:
    """
    Task 6: Complete training loop with early stopping.

    Args:
        model: Neural network
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        num_epochs: Maximum number of epochs
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs

    Returns:
        Dictionary with training history:
        - 'train_loss': List of training losses per epoch
        - 'val_loss': List of validation losses per epoch
        - 'val_accuracy': List of validation accuracies per epoch
    """
    # TODO: Implement full training loop with early stopping
    pass


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """
    Task 7: Save training checkpoint.

    Save dictionary containing:
    - 'epoch': current epoch
    - 'model_state_dict': model weights
    - 'optimizer_state_dict': optimizer state
    - 'loss': current loss
    """
    # TODO: Implement
    pass


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict:
    """
    Task 8: Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dictionary
    """
    # TODO: Implement
    pass


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Task 9: Compute class weights for imbalanced datasets.

    Args:
        labels: Tensor of class labels

    Returns:
        Tensor of weights inversely proportional to class frequency

    Example:
        labels = [0, 0, 0, 1]  # 3 class-0, 1 class-1
        weights should give higher weight to class 1
    """
    # TODO: Implement
    pass


def learning_rate_schedule(
    optimizer: optim.Optimizer,
    epoch: int,
    initial_lr: float,
    decay_rate: float = 0.1,
    decay_epochs: List[int] = [30, 60],
) -> float:
    """
    Task 10: Implement step learning rate decay.

    Args:
        optimizer: Optimizer to update
        epoch: Current epoch
        initial_lr: Starting learning rate
        decay_rate: Factor to multiply LR by at decay epochs
        decay_epochs: Epochs at which to decay LR

    Returns:
        Current learning rate

    Example:
        initial_lr=0.1, decay_rate=0.1, decay_epochs=[30, 60]
        epoch 0-29: lr = 0.1
        epoch 30-59: lr = 0.01
        epoch 60+: lr = 0.001
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Exercise 3: Training Pipelines")
    print("=" * 60)

    print("\n1. Getting device...")
    device = get_device()
    if device:
        print(f"   Device: {device}")

    print("\n2. Creating data loaders...")
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    loaders = create_data_loaders(X[:800], y[:800], X[800:], y[800:], batch_size=32)
    if loaders:
        train_loader, val_loader = loaders
        print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n3. Custom dataset...")
    data = np.random.randn(100, 10).astype(np.float32)
    labels = np.random.randint(0, 2, 100)
    ds = CustomDataset(data, labels)
    if hasattr(ds, "__len__") and len(ds) > 0:
        print(f"   Dataset length: {len(ds)}")
        sample = ds[0]
        if sample:
            print(f"   Sample shape: {sample[0].shape}")

    print("\n4. Class weights...")
    labels = torch.tensor([0, 0, 0, 0, 1])
    weights = compute_class_weights(labels)
    if weights is not None:
        print(f"   Weights for imbalanced [0,0,0,0,1]: {weights}")

    print("\nComplete all TODOs and run tests to verify!")

"""
Week 4 Project: Neural Network Image Classifier - SOLUTIONS
============================================================
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
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders."""
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    n_train = int(len(X) * train_ratio)

    train_dataset = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_dataset = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# =============================================================================
# PART 2: Model Architecture
# =============================================================================


class ImageClassifier(nn.Module):
    """Image classifier neural network."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = [512, 256, 128],
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# PART 3: Training Infrastructure
# =============================================================================


class Trainer:
    """Training class."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return total_loss / len(val_loader), correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 5,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            if self.scheduler:
                self.scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        return history

    def save_checkpoint(self, filepath: str, epoch: int, best_val_loss: float) -> None:
        """Save training checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint


# =============================================================================
# PART 4: Evaluation and Inference
# =============================================================================


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model performance."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return {
        "accuracy": correct / total,
        "loss": total_loss / len(test_loader),
    }


def predict(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions on new data."""
    model.eval()
    X = X.to(device)

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return predictions.cpu(), probabilities.cpu()


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
    """Run the complete training experiment."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Generate data
    X, y = generate_synthetic_data(n_samples, n_features, n_classes, seed)
    train_loader, val_loader = prepare_data(
        X, y, train_ratio=0.8, batch_size=batch_size
    )

    # Create model
    model = ImageClassifier(
        input_size=n_features,
        hidden_sizes=hidden_sizes,
        num_classes=n_classes,
        dropout=dropout,
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, criterion, optimizer, device)

    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
    )

    # Evaluate
    final_metrics = evaluate_model(model, val_loader, device)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())

    return {
        "history": history,
        "final_metrics": final_metrics,
        "model_info": {
            "input_size": n_features,
            "hidden_sizes": hidden_sizes,
            "num_classes": n_classes,
            "total_parameters": total_params,
            "device": str(device),
        },
    }

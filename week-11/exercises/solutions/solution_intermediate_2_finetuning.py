"""
Solutions for Week 11 - Exercise 2 (Intermediate): Fine-tuning with Trainer API
================================================================================
"""

from typing import Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_scheduler,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
import evaluate


# =============================================================================
# TASK 1: Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    """Complete training configuration with validation."""

    model_name: str
    output_dir: str
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")

    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            fp16=self.fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            save_strategy=self.save_strategy,
            evaluation_strategy=self.evaluation_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# TASK 2: Dataset Preprocessor
# =============================================================================
class DatasetPreprocessor:
    """Preprocess datasets for fine-tuning."""

    def __init__(self, tokenizer, max_length: int = 512):
        """Initialize with tokenizer and max length."""
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_function(self, examples: dict) -> dict:
        """Tokenize a batch of examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Will be padded dynamically
        )

    def preprocess_dataset(
        self,
        dataset: Dataset | DatasetDict,
        text_column: str = "text",
        label_column: str = "label",
    ) -> Dataset | DatasetDict:
        """Preprocess dataset with tokenization."""
        # Rename columns if needed
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        if label_column != "label":
            dataset = dataset.rename_column(label_column, "label")

        # Tokenize
        tokenized = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[
                c
                for c in dataset.column_names
                if c not in ["label", "input_ids", "attention_mask"]
            ],
        )
        return tokenized

    def prepare_for_training(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> DatasetDict:
        """Create and preprocess a DatasetDict from raw data."""
        train_dataset = Dataset.from_list(train_data)
        tokenized_train = self.preprocess_dataset(train_dataset)

        if eval_data:
            eval_dataset = Dataset.from_list(eval_data)
            tokenized_eval = self.preprocess_dataset(eval_dataset)
            return DatasetDict({"train": tokenized_train, "eval": tokenized_eval})

        return DatasetDict({"train": tokenized_train})


# =============================================================================
# TASK 3: Metrics Computer
# =============================================================================
class MetricsComputer:
    """Compute and manage evaluation metrics."""

    def __init__(self, metric_names: list[str] = None):
        """Load specified metrics."""
        if metric_names is None:
            metric_names = ["accuracy"]
        self.metrics = {name: evaluate.load(name) for name in metric_names}

    def compute(self, eval_pred) -> dict:
        """Compute all metrics for evaluation predictions."""
        predictions, labels = eval_pred
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=-1)

        results = {}
        for name, metric in self.metrics.items():
            result = metric.compute(predictions=predictions, references=labels)
            if isinstance(result, dict):
                results.update(result)
            else:
                results[name] = result
        return results

    def add_metric(self, name: str) -> None:
        """Add a metric to compute."""
        if name not in self.metrics:
            self.metrics[name] = evaluate.load(name)

    def get_compute_function(self) -> Callable:
        """Return a compute function compatible with Trainer."""
        return self.compute


# =============================================================================
# TASK 4: Model Trainer Wrapper
# =============================================================================
class ModelTrainer:
    """Wrapper around HuggingFace Trainer with additional utilities."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels
        )
        self.trainer: Optional[Trainer] = None
        self.training_history: list[dict] = []

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """Configure the Trainer object."""
        training_args = self.config.to_training_args()

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks or [],
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Run training and return results."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")

        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.training_history.append(
            {"timestamp": time.time(), "metrics": train_result.metrics}
        )
        return train_result.metrics

    def evaluate(self) -> dict:
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        return self.trainer.evaluate()

    def save(self, path: Optional[str] = None) -> None:
        """Save model and tokenizer."""
        save_path = path or self.config.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


# =============================================================================
# TASK 5: Early Stopping Handler
# =============================================================================
class EarlyStoppingHandler:
    """Configure and manage early stopping behavior."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """Initialize early stopping parameters."""
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.greater_is_better = greater_is_better

        self.best_metric: Optional[float] = None
        self.counter = 0

    def get_callback(self) -> EarlyStoppingCallback:
        """Return HuggingFace EarlyStoppingCallback."""
        return EarlyStoppingCallback(
            early_stopping_patience=self.patience,
            early_stopping_threshold=self.threshold,
        )

    def should_stop(self, current_metric: float) -> bool:
        """Manually check if training should stop."""
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        improved = (
            current_metric > self.best_metric + self.threshold
            if self.greater_is_better
            else current_metric < self.best_metric - self.threshold
        )

        if improved:
            self.best_metric = current_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_metric = None
        self.counter = 0


# =============================================================================
# TASK 6: Learning Rate Scheduler Configuration
# =============================================================================
@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate schedulers."""

    scheduler_type: str = "linear"  # linear, cosine, cosine_with_restarts, polynomial
    num_warmup_steps: int = 0
    num_training_steps: int = 1000
    num_cycles: float = 0.5  # For cosine_with_restarts
    power: float = 1.0  # For polynomial

    def create_scheduler(self, optimizer) -> Any:
        """Create the configured scheduler."""
        return get_scheduler(
            name=self.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

    @classmethod
    def for_training(
        cls, scheduler_type: str, num_training_steps: int, warmup_ratio: float = 0.1
    ) -> "LRSchedulerConfig":
        """Create config for typical training setup."""
        return cls(
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=int(num_training_steps * warmup_ratio),
        )


# =============================================================================
# TASK 7: Dynamic Padding Collator
# =============================================================================
class DynamicPaddingCollator:
    """Data collator with dynamic padding for efficient training."""

    def __init__(
        self, tokenizer, padding: str = "longest", max_length: Optional[int] = None
    ):
        """Initialize collator."""
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        """Collate features into a batch with dynamic padding."""
        # Separate labels from features
        labels = [f.pop("label", None) for f in features]

        # Pad inputs
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Add labels back
        if labels[0] is not None:
            batch["labels"] = torch.tensor(labels)

        # Restore features
        for f, label in zip(features, labels):
            if label is not None:
                f["label"] = label

        return batch

    def get_pad_token_id(self) -> int:
        """Return the pad token ID."""
        return self.tokenizer.pad_token_id


# =============================================================================
# TASK 8: Training Logger
# =============================================================================
class TrainingLogger:
    """Log training progress and metrics."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize logger with output directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list[dict] = []
        self.start_time: Optional[float] = None

    def start_training(self) -> None:
        """Mark training start."""
        self.start_time = time.time()
        self.log_event("training_started", {})

    def log_event(self, event_type: str, data: dict) -> None:
        """Log a training event."""
        entry = {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time if self.start_time else 0,
            "event_type": event_type,
            "data": data,
        }
        self.logs.append(entry)

    def log_step(self, step: int, metrics: dict) -> None:
        """Log metrics at a training step."""
        self.log_event("step", {"step": step, "metrics": metrics})

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log metrics at end of epoch."""
        self.log_event("epoch", {"epoch": epoch, "metrics": metrics})

    def end_training(self, final_metrics: dict) -> None:
        """Mark training end and save logs."""
        self.log_event("training_ended", {"final_metrics": final_metrics})
        self.save()

    def save(self, filename: str = "training_log.json") -> None:
        """Save logs to file."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def get_metrics_history(self, metric_name: str) -> list[tuple[int, float]]:
        """Get history of a specific metric."""
        history = []
        for entry in self.logs:
            if entry["event_type"] in ["step", "epoch"]:
                metrics = entry["data"].get("metrics", {})
                if metric_name in metrics:
                    key = "step" if "step" in entry["data"] else "epoch"
                    history.append((entry["data"][key], metrics[metric_name]))
        return history


# =============================================================================
# TASK 9: Checkpoint Manager
# =============================================================================
class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_best_only: bool = False,
    ):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.checkpoints: list[dict] = []
        self.best_metric: Optional[float] = None

    def save_checkpoint(
        self,
        model,
        tokenizer,
        step: int,
        metrics: dict,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> Optional[str]:
        """Save a checkpoint if conditions are met."""
        current_metric = metrics.get(metric_for_best)

        # Check if this is the best model
        is_best = False
        if current_metric is not None:
            if self.best_metric is None:
                is_best = True
            elif greater_is_better and current_metric > self.best_metric:
                is_best = True
            elif not greater_is_better and current_metric < self.best_metric:
                is_best = True

            if is_best:
                self.best_metric = current_metric

        # Skip if save_best_only and not best
        if self.save_best_only and not is_best:
            return None

        # Create checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        # Save metadata
        metadata = {
            "step": step,
            "metrics": metrics,
            "is_best": is_best,
            "path": str(checkpoint_path),
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.checkpoints.append(metadata)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            # Keep best checkpoint, remove oldest non-best
            oldest = None
            for ckpt in self.checkpoints:
                if not ckpt["is_best"]:
                    oldest = ckpt
                    break

            if oldest is None:
                # All are best (shouldn't happen), remove oldest
                oldest = self.checkpoints[0]

            # Remove checkpoint directory
            ckpt_path = Path(oldest["path"])
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)

            self.checkpoints.remove(oldest)

    def load_best_checkpoint(self, model_class):
        """Load the best checkpoint."""
        best = None
        for ckpt in self.checkpoints:
            if ckpt["is_best"]:
                best = ckpt
                break

        if best is None and self.checkpoints:
            best = self.checkpoints[-1]

        if best:
            return model_class.from_pretrained(best["path"])
        return None

    def get_checkpoint_info(self) -> list[dict]:
        """Get information about all checkpoints."""
        return self.checkpoints.copy()


# =============================================================================
# TASK 10: Complete Training Pipeline
# =============================================================================
class TrainingPipeline:
    """End-to-end training pipeline combining all components."""

    def __init__(self, config: TrainingConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.preprocessor = DatasetPreprocessor(self.tokenizer, config.max_length)
        self.metrics_computer = MetricsComputer(["accuracy"])
        self.logger = TrainingLogger(f"{config.output_dir}/logs")
        self.checkpoint_manager = CheckpointManager(
            f"{config.output_dir}/checkpoints", max_checkpoints=3
        )
        self.early_stopping = EarlyStoppingHandler(patience=3)
        self.trainer_wrapper: Optional[ModelTrainer] = None

    def prepare_data(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> DatasetDict:
        """Prepare datasets for training."""
        return self.preprocessor.prepare_for_training(train_data, eval_data)

    def setup(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Set up all training components."""
        self.trainer_wrapper = ModelTrainer(self.config)

        callbacks = [self.early_stopping.get_callback()]

        self.trainer_wrapper.setup_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.metrics_computer.get_compute_function(),
            callbacks=callbacks,
        )

    def train(self) -> dict:
        """Run the full training process."""
        if self.trainer_wrapper is None:
            raise ValueError("Pipeline not set up. Call setup first.")

        self.logger.start_training()

        try:
            metrics = self.trainer_wrapper.train()
            self.logger.log_epoch(self.config.num_epochs, metrics)

            # Save final checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.trainer_wrapper.model,
                self.tokenizer,
                step=int(time.time()),
                metrics=metrics,
            )

            self.logger.end_training(metrics)
            return metrics

        except Exception as e:
            self.logger.log_event("error", {"error": str(e)})
            raise

    def evaluate(self) -> dict:
        """Evaluate the trained model."""
        if self.trainer_wrapper is None:
            raise ValueError("Pipeline not set up. Call setup first.")
        return self.trainer_wrapper.evaluate()

    def save(self) -> None:
        """Save all artifacts."""
        if self.trainer_wrapper:
            self.trainer_wrapper.save()
        self.config.save(f"{self.config.output_dir}/config.json")

    def run(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> dict:
        """Run the complete pipeline."""
        # Prepare data
        datasets = self.prepare_data(train_data, eval_data)

        # Setup
        self.setup(datasets["train"], datasets.get("eval"))

        # Train
        metrics = self.train()

        # Save
        self.save()

        return metrics


if __name__ == "__main__":
    print("Week 11 - Exercise 2 Solutions: Fine-tuning with Trainer API")
    print("=" * 60)

    # Test TrainingConfig
    print("\n1. TrainingConfig:")
    config = TrainingConfig(
        model_name="bert-base-uncased",
        output_dir="./output",
        num_labels=2,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=3,
    )
    print(f"   Config created: {config.model_name}")

    # Test to_dict
    config_dict = config.to_dict()
    print(f"   to_dict: {list(config_dict.keys())[:5]}...")

    # Test MetricsComputer
    print("\n2. MetricsComputer:")
    metrics = MetricsComputer(["accuracy"])
    print(f"   Metrics loaded: {list(metrics.metrics.keys())}")

    # Test LRSchedulerConfig
    print("\n3. LRSchedulerConfig:")
    lr_config = LRSchedulerConfig.for_training("cosine", 1000, warmup_ratio=0.1)
    print(
        f"   Scheduler: {lr_config.scheduler_type}, warmup: {lr_config.num_warmup_steps}"
    )

    # Test TrainingLogger
    print("\n4. TrainingLogger:")
    logger = TrainingLogger("./test_logs")
    logger.start_training()
    logger.log_step(100, {"loss": 0.5})
    print(f"   Logged {len(logger.logs)} events")

    print("\n✅ All solutions implemented!")

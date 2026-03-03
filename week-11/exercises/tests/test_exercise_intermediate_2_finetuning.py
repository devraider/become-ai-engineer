"""
Tests for Week 11 - Exercise 2: Fine-tuning with Trainer
"""

import os
import tempfile
from dataclasses import asdict
from importlib.util import find_spec
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

# Check for optional dependencies
HAS_TRANSFORMERS = find_spec("transformers") is not None

if HAS_TRANSFORMERS:
    from transformers import AutoTokenizer

# Import exercise classes
from exercise_intermediate_2_finetuning import (
    TrainingConfig,
    DatasetPreprocessor,
    MetricsComputer,
    ModelTrainer,
    EarlyStoppingHandler,
    LRSchedulerConfig,
    DynamicPaddingCollator,
    TrainingLogger,
    CheckpointManager,
    TrainingPipeline,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.batch_size == 16

    def test_to_training_args(self):
        """Test conversion to HF TrainingArguments."""
        config = TrainingConfig(output_dir="./test", learning_rate=1e-4, num_epochs=5)
        args = config.to_training_args()
        assert args.output_dir == "./test"
        assert args.learning_rate == 1e-4
        assert args.num_train_epochs == 5

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "model_name": "distilbert-base-uncased",
            "learning_rate": 3e-5,
            "batch_size": 32,
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.model_name == "distilbert-base-uncased"
        assert config.learning_rate == 3e-5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "learning_rate" in config_dict


class TestDatasetPreprocessor:
    """Tests for DatasetPreprocessor class."""

    def test_init_loads_tokenizer(self):
        """Test tokenizer is loaded."""
        preprocessor = DatasetPreprocessor("bert-base-uncased")
        assert preprocessor.tokenizer is not None

    def test_tokenize_function(self):
        """Test tokenization works."""
        preprocessor = DatasetPreprocessor("bert-base-uncased")
        examples = {"text": ["Hello world", "Test sentence"]}
        tokenized = preprocessor.tokenize_function(examples)
        assert "input_ids" in tokenized
        assert "attention_mask" in tokenized

    def test_prepare_dataset(self):
        """Test dataset preparation."""
        preprocessor = DatasetPreprocessor("bert-base-uncased")
        data = {"text": ["Sample text 1", "Sample text 2"], "label": [0, 1]}
        dataset = preprocessor.prepare_dataset(data)
        assert "input_ids" in dataset.column_names

    def test_create_splits(self):
        """Test train/val/test split creation."""
        preprocessor = DatasetPreprocessor("bert-base-uncased")
        data = {
            "text": ["Text " + str(i) for i in range(100)],
            "label": [i % 2 for i in range(100)],
        }
        dataset = preprocessor.prepare_dataset(data)
        splits = preprocessor.create_splits(dataset, 0.8, 0.1, 0.1)
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits


class TestMetricsComputer:
    """Tests for MetricsComputer class."""

    def test_init_loads_metrics(self):
        """Test metrics are loaded."""
        metrics = MetricsComputer(["accuracy", "f1"])
        assert len(metrics.metrics) == 2

    def test_call_computes_metrics(self):
        """Test __call__ computes metrics for Trainer."""
        metrics = MetricsComputer(["accuracy"])

        # Create mock EvalPrediction
        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1, 0, 1])
        eval_pred = (logits, labels)

        results = metrics(eval_pred)
        assert "accuracy" in results
        assert results["accuracy"] == 1.0

    def test_compute_metrics_direct(self):
        """Test direct metric computation."""
        metrics = MetricsComputer(["accuracy", "f1"])
        predictions = [0, 1, 1, 0]
        references = [0, 1, 0, 0]

        results = metrics.compute_metrics(predictions, references)
        assert "accuracy" in results
        assert "f1" in results


class TestModelTrainer:
    """Tests for ModelTrainer wrapper."""

    def test_init_creates_components(self):
        """Test initialization creates model and tokenizer."""
        config = TrainingConfig(model_name="distilbert-base-uncased")
        trainer = ModelTrainer(config, num_labels=2)
        assert trainer.model is not None
        assert trainer.tokenizer is not None

    def test_setup_trainer(self):
        """Test trainer setup."""
        config = TrainingConfig()
        trainer = ModelTrainer(config)

        # Mock datasets
        mock_train = Mock()
        mock_eval = Mock()

        hf_trainer = trainer.setup_trainer(mock_train, mock_eval)
        assert hf_trainer is not None

    def test_save_and_load_model(self):
        """Test model save and load."""
        config = TrainingConfig()
        trainer = ModelTrainer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            trainer.save_model(path)
            assert os.path.exists(path)

            trainer.load_model(path)
            assert trainer.model is not None


class TestEarlyStoppingHandler:
    """Tests for EarlyStoppingHandler class."""

    def test_get_callback(self):
        """Test callback creation."""
        handler = EarlyStoppingHandler(patience=3)
        callback = handler.get_callback()
        assert callback is not None

    def test_should_stop_after_patience(self):
        """Test stopping after patience exceeded."""
        handler = EarlyStoppingHandler(patience=2, metric="loss")

        # Improvement
        assert not handler.should_stop(0.5)
        # No improvement
        assert not handler.should_stop(0.5)
        assert not handler.should_stop(0.5)
        # Should stop now
        assert handler.should_stop(0.5)

    def test_reset(self):
        """Test reset clears state."""
        handler = EarlyStoppingHandler(patience=2)
        handler.should_stop(0.5)
        handler.should_stop(0.5)

        handler.reset()
        assert handler.best_value is None
        assert handler.counter == 0


class TestLRSchedulerConfig:
    """Tests for LRSchedulerConfig class."""

    def test_available_schedulers(self):
        """Test available schedulers list."""
        schedulers = LRSchedulerConfig.available_schedulers()
        assert "linear" in schedulers
        assert "cosine" in schedulers

    def test_get_scheduler_params_linear(self):
        """Test linear scheduler parameters."""
        config = LRSchedulerConfig("linear", warmup_ratio=0.1)
        params = config.get_scheduler_params(1000)
        assert "num_warmup_steps" in params
        assert params["num_warmup_steps"] == 100

    def test_get_scheduler_params_cosine(self):
        """Test cosine scheduler parameters."""
        config = LRSchedulerConfig("cosine", warmup_ratio=0.1, num_cycles=0.5)
        params = config.get_scheduler_params(1000)
        assert "num_cycles" in params or "num_warmup_steps" in params


class TestDynamicPaddingCollator:
    """Tests for DynamicPaddingCollator class."""

    def test_call_pads_batch(self):
        """Test batch collation with padding."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = DynamicPaddingCollator(tokenizer)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
        ]

        batch = collator(features)
        assert batch["input_ids"].shape[1] == 3  # Padded to longest

    def test_handles_labels(self):
        """Test labels are handled correctly."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collator = DynamicPaddingCollator(tokenizer)

        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 0},
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": 1},
        ]

        batch = collator(features)
        assert "labels" in batch


class TestTrainingLogger:
    """Tests for TrainingLogger class."""

    def test_log_step(self):
        """Test step logging."""
        logger = TrainingLogger()
        logger.log_step(step=100, loss=0.5, learning_rate=1e-5)
        assert len(logger.step_logs) == 1
        assert logger.step_logs[0]["step"] == 100

    def test_log_epoch(self):
        """Test epoch logging."""
        logger = TrainingLogger()
        logger.log_epoch(
            epoch=1, train_metrics={"loss": 0.4}, eval_metrics={"accuracy": 0.9}
        )
        assert len(logger.epoch_logs) == 1

    def test_get_best_epoch(self):
        """Test finding best epoch."""
        logger = TrainingLogger()
        logger.log_epoch(1, eval_metrics={"eval_accuracy": 0.8})
        logger.log_epoch(2, eval_metrics={"eval_accuracy": 0.9})
        logger.log_epoch(3, eval_metrics={"eval_accuracy": 0.85})

        best = logger.get_best_epoch("eval_accuracy")
        assert best["epoch"] == 2


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=2)

            mock_model = Mock()
            mock_model.save_pretrained = Mock()

            path = manager.save_checkpoint(
                model=mock_model, epoch=1, metrics={"accuracy": 0.9}
            )

            assert path is not None
            mock_model.save_pretrained.assert_called()

    def test_cleanup_old_checkpoints(self):
        """Test old checkpoint cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=2)

            mock_model = Mock()
            mock_model.save_pretrained = Mock()

            # Save 3 checkpoints
            manager.save_checkpoint(mock_model, epoch=1, metrics={"accuracy": 0.7})
            manager.save_checkpoint(mock_model, epoch=2, metrics={"accuracy": 0.8})
            manager.save_checkpoint(mock_model, epoch=3, metrics={"accuracy": 0.9})

            manager.cleanup()
            # Should keep only 2 best
            assert len(manager.checkpoints) <= 2


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""

    def test_init(self):
        """Test pipeline initialization."""
        config = TrainingConfig()
        pipeline = TrainingPipeline(config)
        assert pipeline.config == config

    def test_prepare_data(self):
        """Test data preparation."""
        config = TrainingConfig()
        pipeline = TrainingPipeline(config)

        data = {"text": ["Sample 1", "Sample 2"], "label": [0, 1]}

        datasets = pipeline.prepare_data(data=data)
        assert datasets is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

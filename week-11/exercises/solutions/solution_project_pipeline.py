"""
Solutions for Week 11 - Project: Multi-Modal Fine-tuning Pipeline
==================================================================
"""

from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict, load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import evaluate


# =============================================================================
# PART 1: Configuration Classes
# =============================================================================
class ModelType(Enum):
    """Supported model types."""

    SEQUENCE_CLASSIFICATION = "sequence_classification"
    CAUSAL_LM = "causal_lm"
    TOKEN_CLASSIFICATION = "token_classification"


class TrainingMethod(Enum):
    """Training method selection."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class PipelineConfig:
    """Complete configuration for the fine-tuning pipeline."""

    # Model configuration
    model_name: str
    model_type: ModelType
    num_labels: int = 2

    # Training method
    training_method: TrainingMethod = TrainingMethod.FULL

    # LoRA configuration (if using LoRA/QLoRA)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list[str]] = None

    # Quantization (if using QLoRA)
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    gradient_accumulation_steps: int = 1

    # Training settings
    fp16: bool = False
    bf16: bool = False

    # Output
    output_dir: str = "./output"

    # Evaluation
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Early stopping
    early_stopping_patience: int = 3

    def __post_init__(self):
        """Validate and set defaults."""
        if self.training_method == TrainingMethod.QLORA:
            self.load_in_4bit = True

        if self.lora_target_modules is None:
            # Default target modules based on model type
            self.lora_target_modules = ["q_proj", "v_proj"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "num_labels": self.num_labels,
            "training_method": self.training_method.value,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "output_dir": self.output_dir,
            "evaluation_strategy": self.evaluation_strategy,
            "save_strategy": self.save_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "early_stopping_patience": self.early_stopping_patience,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create from dictionary."""
        config_dict = config_dict.copy()
        if "model_type" in config_dict:
            config_dict["model_type"] = ModelType(config_dict["model_type"])
        if "training_method" in config_dict:
            config_dict["training_method"] = TrainingMethod(
                config_dict["training_method"]
            )
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# PART 2: Data Module
# =============================================================================
class DataModule(ABC):
    """Abstract base class for data handling."""

    def __init__(self, config: PipelineConfig):
        """Initialize data module."""
        self.config = config
        self.tokenizer = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def set_tokenizer(self, tokenizer) -> None:
        """Set the tokenizer for preprocessing."""
        self.tokenizer = tokenizer

    @abstractmethod
    def load_data(self) -> None:
        """Load raw data."""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess data for training."""
        pass

    def get_train_dataset(self) -> Dataset:
        """Get training dataset."""
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data and preprocess first.")
        return self.train_dataset

    def get_eval_dataset(self) -> Optional[Dataset]:
        """Get evaluation dataset."""
        return self.eval_dataset

    def get_test_dataset(self) -> Optional[Dataset]:
        """Get test dataset."""
        return self.test_dataset


class TextClassificationDataModule(DataModule):
    """Data module for text classification tasks."""

    def __init__(
        self,
        config: PipelineConfig,
        dataset_name: Optional[str] = None,
        train_data: Optional[list[dict]] = None,
        eval_data: Optional[list[dict]] = None,
        text_column: str = "text",
        label_column: str = "label",
    ):
        """Initialize text classification data module."""
        super().__init__(config)
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.eval_data = eval_data
        self.text_column = text_column
        self.label_column = label_column
        self._raw_train = None
        self._raw_eval = None

    def load_data(self) -> None:
        """Load raw data from source."""
        if self.dataset_name:
            dataset = load_dataset(self.dataset_name)
            self._raw_train = dataset["train"]
            if "validation" in dataset:
                self._raw_eval = dataset["validation"]
            elif "test" in dataset:
                self._raw_eval = dataset["test"]
        elif self.train_data:
            self._raw_train = Dataset.from_list(self.train_data)
            if self.eval_data:
                self._raw_eval = Dataset.from_list(self.eval_data)
        else:
            raise ValueError("Either dataset_name or train_data must be provided")

    def _tokenize_function(self, examples: dict) -> dict:
        """Tokenize a batch of examples."""
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )

    def preprocess(self) -> None:
        """Preprocess data for training."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")

        if self._raw_train is None:
            raise ValueError("Data not loaded. Call load_data first.")

        # Rename columns if needed
        train = self._raw_train
        if self.text_column != "text":
            train = train.rename_column(self.text_column, "text")
        if self.label_column != "label":
            train = train.rename_column(self.label_column, "label")
        self.text_column = "text"
        self.label_column = "label"

        # Tokenize
        self.train_dataset = train.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[c for c in train.column_names if c not in ["label"]],
        )

        if self._raw_eval:
            eval_data = self._raw_eval
            if (
                "text" not in eval_data.column_names
                and self.text_column in eval_data.column_names
            ):
                eval_data = eval_data.rename_column(self.text_column, "text")
            if (
                "label" not in eval_data.column_names
                and self.label_column in eval_data.column_names
            ):
                eval_data = eval_data.rename_column(self.label_column, "label")

            self.eval_dataset = eval_data.map(
                self._tokenize_function,
                batched=True,
                remove_columns=[
                    c for c in eval_data.column_names if c not in ["label"]
                ],
            )


# =============================================================================
# PART 3: Model Factory
# =============================================================================
class ModelFactory:
    """Factory for creating and configuring models."""

    def __init__(self, config: PipelineConfig):
        """Initialize factory with configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization config if needed."""
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _get_lora_config(self) -> LoraConfig:
        """Create LoRA configuration."""
        task_type = (
            TaskType.SEQ_CLS
            if self.config.model_type == ModelType.SEQUENCE_CLASSIFICATION
            else TaskType.CAUSAL_LM
        )

        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=task_type,
            bias="none",
        )

    def create_model(self):
        """Create and configure the model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get quantization config
        quant_config = self._get_quantization_config()

        # Load model based on type
        model_kwargs = {}
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        if self.config.model_type == ModelType.SEQUENCE_CLASSIFICATION:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                **model_kwargs,
            )
            # Set pad token id for models that need it
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

        # Prepare for k-bit training if quantized
        if quant_config and self.config.training_method in [
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
        ]:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA if needed
        if self.config.training_method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            lora_config = self._get_lora_config()
            self.model = get_peft_model(self.model, lora_config)

        return self.model, self.tokenizer

    def get_trainable_params(self) -> dict:
        """Get trainable parameter statistics."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model first.")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        return {
            "trainable": trainable,
            "total": total,
            "percentage": 100 * trainable / total,
        }


# =============================================================================
# PART 4: Trainer Module
# =============================================================================
class TrainerModule:
    """Module for training management."""

    def __init__(self, config: PipelineConfig, model, tokenizer):
        """Initialize trainer module."""
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.trainer: Optional[Trainer] = None
        self.training_history: list[dict] = []

    def _create_training_args(self) -> TrainingArguments:
        """Create TrainingArguments from config."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_steps=10,
        )

    def setup(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> None:
        """Set up the trainer."""
        training_args = self._create_training_args()

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> dict:
        """Run training."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup first.")

        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.training_history.append(
            {"timestamp": time.time(), "metrics": result.metrics}
        )
        return result.metrics

    def evaluate(self) -> dict:
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup first.")
        return self.trainer.evaluate()

    def save_model(self, path: Optional[str] = None) -> None:
        """Save the model."""
        save_path = path or self.config.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


# =============================================================================
# PART 5: Evaluation Module
# =============================================================================
@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metrics: dict
    predictions: Optional[list] = None
    labels: Optional[list] = None
    timestamp: float = field(default_factory=time.time)


class EvaluationModule:
    """Module for model evaluation."""

    def __init__(self, metric_names: Optional[list[str]] = None):
        """Initialize evaluation module."""
        if metric_names is None:
            metric_names = ["accuracy"]
        self.metrics = {name: evaluate.load(name) for name in metric_names}

    def compute_metrics(self, eval_pred) -> dict:
        """Compute metrics for trainer callback."""
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

    def evaluate(
        self, model, tokenizer, eval_dataset: Dataset, batch_size: int = 16
    ) -> EvaluationResult:
        """Run full evaluation."""
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        dataloader = DataLoader(
            eval_dataset, batch_size=batch_size, collate_fn=data_collator
        )

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels")
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        metrics = {}
        for name, metric in self.metrics.items():
            result = metric.compute(predictions=all_predictions, references=all_labels)
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[name] = result

        return EvaluationResult(
            metrics=metrics, predictions=all_predictions, labels=all_labels
        )

    def add_metric(self, name: str) -> None:
        """Add a metric to compute."""
        if name not in self.metrics:
            self.metrics[name] = evaluate.load(name)


# =============================================================================
# PART 6: Model Exporter
# =============================================================================
class ModelExporter:
    """Export trained models in various formats."""

    def __init__(self, model, tokenizer, config: PipelineConfig):
        """Initialize exporter."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def export_huggingface(self, output_dir: str) -> None:
        """Export in HuggingFace format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # If PEFT model, handle merging
        if self.config.training_method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            if hasattr(self.model, "merge_and_unload"):
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(str(output_path))
            else:
                self.model.save_pretrained(str(output_path))
        else:
            self.model.save_pretrained(str(output_path))

        self.tokenizer.save_pretrained(str(output_path))

    def export_adapter_only(self, output_dir: str) -> None:
        """Export only the adapter weights (for PEFT models)."""
        if self.config.training_method not in [
            TrainingMethod.LORA,
            TrainingMethod.QLORA,
        ]:
            raise ValueError("Adapter export only available for LoRA/QLoRA models")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_path))

    def export_onnx(self, output_dir: str) -> None:
        """Export to ONNX format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge if PEFT
        model = self.model
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

        # Export with torch.onnx
        dummy_input = self.tokenizer(
            "Sample text for export", return_tensors="pt", padding=True
        )

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(output_path / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
        )

    def create_model_card(self, output_dir: str, **kwargs) -> None:
        """Create a model card for the exported model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        card_content = f"""---
language: en
license: apache-2.0
tags:
- fine-tuned
- {self.config.model_type.value}
---

# Fine-tuned Model

## Model Description

Base model: {self.config.model_name}
Training method: {self.config.training_method.value}
Task: {self.config.model_type.value}

## Training Details

- Learning rate: {self.config.learning_rate}
- Batch size: {self.config.batch_size}
- Epochs: {self.config.num_epochs}
"""

        if self.config.training_method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            card_content += f"""
### LoRA Configuration

- Rank (r): {self.config.lora_r}
- Alpha: {self.config.lora_alpha}
- Dropout: {self.config.lora_dropout}
- Target modules: {self.config.lora_target_modules}
"""

        with open(output_path / "README.md", "w") as f:
            f.write(card_content)


# =============================================================================
# PART 7: Experiment Tracker
# =============================================================================
class ExperimentTracker:
    """Track experiments and results."""

    def __init__(self, experiment_dir: str = "./experiments"):
        """Initialize tracker."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[str] = None
        self.experiments: dict[str, dict] = {}

    def start_experiment(self, name: str, config: PipelineConfig) -> str:
        """Start a new experiment."""
        experiment_id = f"{name}_{int(time.time())}"
        self.current_experiment = experiment_id

        self.experiments[experiment_id] = {
            "name": name,
            "config": config.to_dict(),
            "start_time": time.time(),
            "metrics": {},
            "checkpoints": [],
            "status": "running",
        }

        # Save config
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        config.save(str(exp_dir / "config.json"))

        return experiment_id

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics for current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")

        key = f"step_{step}" if step else f"time_{int(time.time())}"
        self.experiments[self.current_experiment]["metrics"][key] = metrics

    def log_checkpoint(self, path: str, metrics: dict) -> None:
        """Log a checkpoint."""
        if self.current_experiment is None:
            raise ValueError("No active experiment.")

        self.experiments[self.current_experiment]["checkpoints"].append(
            {"path": path, "metrics": metrics, "timestamp": time.time()}
        )

    def end_experiment(self, final_metrics: dict, status: str = "completed") -> None:
        """End the current experiment."""
        if self.current_experiment is None:
            return

        exp = self.experiments[self.current_experiment]
        exp["end_time"] = time.time()
        exp["duration"] = exp["end_time"] - exp["start_time"]
        exp["final_metrics"] = final_metrics
        exp["status"] = status

        # Save experiment data
        exp_dir = self.experiment_dir / self.current_experiment
        with open(exp_dir / "results.json", "w") as f:
            json.dump(exp, f, indent=2)

        self.current_experiment = None

    def get_experiment(self, experiment_id: str) -> dict:
        """Get experiment data."""
        return self.experiments.get(experiment_id, {})

    def list_experiments(self) -> list[str]:
        """List all experiments."""
        return list(self.experiments.keys())

    def compare_experiments(
        self, experiment_ids: list[str], metric: str
    ) -> dict[str, float]:
        """Compare experiments by a metric."""
        results = {}
        for exp_id in experiment_ids:
            exp = self.experiments.get(exp_id, {})
            final_metrics = exp.get("final_metrics", {})
            if metric in final_metrics:
                results[exp_id] = final_metrics[metric]
        return results


# =============================================================================
# PART 8: Main Pipeline
# =============================================================================
class FineTuningPipeline:
    """Complete fine-tuning pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline."""
        self.config = config
        self.model_factory: Optional[ModelFactory] = None
        self.data_module: Optional[DataModule] = None
        self.trainer_module: Optional[TrainerModule] = None
        self.evaluation_module: Optional[EvaluationModule] = None
        self.experiment_tracker: Optional[ExperimentTracker] = None
        self.model = None
        self.tokenizer = None

    def setup_data(self, data_module: DataModule) -> None:
        """Set up data module."""
        self.data_module = data_module
        self.data_module.load_data()

    def setup_model(self) -> None:
        """Set up model and tokenizer."""
        self.model_factory = ModelFactory(self.config)
        self.model, self.tokenizer = self.model_factory.create_model()

        if self.data_module:
            self.data_module.set_tokenizer(self.tokenizer)
            self.data_module.preprocess()

    def setup_evaluation(self, metrics: Optional[list[str]] = None) -> None:
        """Set up evaluation module."""
        self.evaluation_module = EvaluationModule(metrics or ["accuracy"])

    def setup_tracking(self, experiment_dir: str = "./experiments") -> None:
        """Set up experiment tracking."""
        self.experiment_tracker = ExperimentTracker(experiment_dir)

    def train(self, experiment_name: str = "experiment") -> dict:
        """Run the training pipeline."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model first.")
        if self.data_module is None:
            raise ValueError("Data not set up. Call setup_data first.")

        # Start tracking
        if self.experiment_tracker:
            self.experiment_tracker.start_experiment(experiment_name, self.config)

        try:
            # Set up trainer
            self.trainer_module = TrainerModule(self.config, self.model, self.tokenizer)

            compute_metrics = None
            if self.evaluation_module:
                compute_metrics = self.evaluation_module.compute_metrics

            self.trainer_module.setup(
                train_dataset=self.data_module.get_train_dataset(),
                eval_dataset=self.data_module.get_eval_dataset(),
                compute_metrics=compute_metrics,
            )

            # Train
            metrics = self.trainer_module.train()

            # Log metrics
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics(metrics)

            # Evaluate
            eval_metrics = {}
            if self.data_module.get_eval_dataset():
                eval_metrics = self.trainer_module.evaluate()
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(eval_metrics, step=-1)

            # End tracking
            final_metrics = {**metrics, **eval_metrics}
            if self.experiment_tracker:
                self.experiment_tracker.end_experiment(final_metrics)

            return final_metrics

        except Exception as e:
            if self.experiment_tracker:
                self.experiment_tracker.end_experiment({}, status="failed")
            raise

    def evaluate(self) -> EvaluationResult:
        """Run evaluation."""
        if self.evaluation_module is None:
            self.setup_evaluation()

        eval_dataset = self.data_module.get_eval_dataset()
        if eval_dataset is None:
            raise ValueError("No evaluation dataset available")

        return self.evaluation_module.evaluate(
            self.model, self.tokenizer, eval_dataset, self.config.batch_size
        )

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save model and artifacts."""
        save_dir = output_dir or self.config.output_dir

        exporter = ModelExporter(self.model, self.tokenizer, self.config)

        if self.config.training_method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            # Save adapter only
            exporter.export_adapter_only(f"{save_dir}/adapter")

        # Save full model
        exporter.export_huggingface(f"{save_dir}/model")

        # Save config
        self.config.save(f"{save_dir}/config.json")

        # Create model card
        exporter.create_model_card(f"{save_dir}/model")

    def run(
        self,
        data_module: DataModule,
        experiment_name: str = "experiment",
        save: bool = True,
    ) -> dict:
        """Run the complete pipeline."""
        # Setup
        self.setup_data(data_module)
        self.setup_model()
        self.setup_evaluation()
        self.setup_tracking()

        # Train
        metrics = self.train(experiment_name)

        # Save
        if save:
            self.save()

        return metrics


# =============================================================================
# PART 9: Pipeline Factory
# =============================================================================
class PipelineFactory:
    """Factory for creating pre-configured pipelines."""

    @staticmethod
    def create_classification_pipeline(
        model_name: str,
        num_labels: int,
        training_method: TrainingMethod = TrainingMethod.FULL,
        output_dir: str = "./output",
    ) -> FineTuningPipeline:
        """Create a text classification pipeline."""
        config = PipelineConfig(
            model_name=model_name,
            model_type=ModelType.SEQUENCE_CLASSIFICATION,
            num_labels=num_labels,
            training_method=training_method,
            output_dir=output_dir,
        )
        return FineTuningPipeline(config)

    @staticmethod
    def create_lora_pipeline(
        model_name: str,
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        output_dir: str = "./output",
    ) -> FineTuningPipeline:
        """Create a LoRA fine-tuning pipeline."""
        config = PipelineConfig(
            model_name=model_name,
            model_type=ModelType.SEQUENCE_CLASSIFICATION,
            num_labels=num_labels,
            training_method=TrainingMethod.LORA,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            output_dir=output_dir,
        )
        return FineTuningPipeline(config)

    @staticmethod
    def create_qlora_pipeline(
        model_name: str,
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        output_dir: str = "./output",
    ) -> FineTuningPipeline:
        """Create a QLoRA fine-tuning pipeline."""
        config = PipelineConfig(
            model_name=model_name,
            model_type=ModelType.SEQUENCE_CLASSIFICATION,
            num_labels=num_labels,
            training_method=TrainingMethod.QLORA,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            load_in_4bit=True,
            output_dir=output_dir,
        )
        return FineTuningPipeline(config)


if __name__ == "__main__":
    print("Week 11 - Project Solutions: Multi-Modal Fine-tuning Pipeline")
    print("=" * 60)

    # Test PipelineConfig
    print("\n1. PipelineConfig:")
    config = PipelineConfig(
        model_name="bert-base-uncased",
        model_type=ModelType.SEQUENCE_CLASSIFICATION,
        num_labels=2,
        training_method=TrainingMethod.LORA,
        lora_r=8,
        output_dir="./test_output",
    )
    print(f"   Config: {config.model_name}, method={config.training_method.value}")

    # Test factory
    print("\n2. PipelineFactory:")
    pipeline = PipelineFactory.create_lora_pipeline(
        model_name="bert-base-uncased", num_labels=2, lora_r=8
    )
    print(f"   Created LoRA pipeline: {pipeline.config.training_method.value}")

    # Test ExperimentTracker
    print("\n3. ExperimentTracker:")
    tracker = ExperimentTracker("./test_experiments")
    exp_id = tracker.start_experiment("test", config)
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.85})
    tracker.end_experiment({"final_loss": 0.3, "final_accuracy": 0.92})
    print(f"   Experiment tracked: {exp_id}")

    # Test EvaluationModule
    print("\n4. EvaluationModule:")
    eval_module = EvaluationModule(["accuracy"])
    print(f"   Metrics loaded: {list(eval_module.metrics.keys())}")

    print("\n✅ All pipeline solutions implemented!")

"""
Solutions for Week 11 - Exercise 3 (Advanced): PEFT and Quantization
=====================================================================
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import json
import time
import gc

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)


# =============================================================================
# TASK 1: LoRA Configuration Builder
# =============================================================================
class LoRAConfigBuilder:
    """Builder pattern for creating LoRA configurations."""

    def __init__(self):
        """Initialize with default values."""
        self._r = 8
        self._lora_alpha = 16
        self._target_modules: Optional[list[str]] = None
        self._lora_dropout = 0.1
        self._task_type = TaskType.CAUSAL_LM
        self._bias = "none"
        self._modules_to_save: Optional[list[str]] = None

    def rank(self, r: int) -> "LoRAConfigBuilder":
        """Set the rank of LoRA matrices."""
        self._r = r
        return self

    def alpha(self, alpha: int) -> "LoRAConfigBuilder":
        """Set the alpha scaling factor."""
        self._lora_alpha = alpha
        return self

    def target_modules(self, modules: list[str]) -> "LoRAConfigBuilder":
        """Set the target modules for LoRA."""
        self._target_modules = modules
        return self

    def dropout(self, dropout: float) -> "LoRAConfigBuilder":
        """Set dropout rate."""
        self._lora_dropout = dropout
        return self

    def task_type(self, task: TaskType) -> "LoRAConfigBuilder":
        """Set the task type."""
        self._task_type = task
        return self

    def bias(self, bias: str) -> "LoRAConfigBuilder":
        """Set bias handling (none, all, lora_only)."""
        self._bias = bias
        return self

    def modules_to_save(self, modules: list[str]) -> "LoRAConfigBuilder":
        """Set modules to fully train (not LoRA)."""
        self._modules_to_save = modules
        return self

    def build(self) -> LoraConfig:
        """Build the LoRA configuration."""
        return LoraConfig(
            r=self._r,
            lora_alpha=self._lora_alpha,
            target_modules=self._target_modules,
            lora_dropout=self._lora_dropout,
            task_type=self._task_type,
            bias=self._bias,
            modules_to_save=self._modules_to_save,
        )

    @classmethod
    def for_llama(cls, r: int = 8) -> LoraConfig:
        """Create optimized config for Llama models."""
        return (
            cls()
            .rank(r)
            .alpha(r * 2)
            .target_modules(["q_proj", "k_proj", "v_proj", "o_proj"])
            .dropout(0.05)
            .task_type(TaskType.CAUSAL_LM)
            .build()
        )

    @classmethod
    def for_gpt2(cls, r: int = 8) -> LoraConfig:
        """Create optimized config for GPT-2 models."""
        return (
            cls()
            .rank(r)
            .alpha(r * 2)
            .target_modules(["c_attn", "c_proj"])
            .dropout(0.1)
            .task_type(TaskType.CAUSAL_LM)
            .build()
        )

    @classmethod
    def for_bert(cls, r: int = 8) -> LoraConfig:
        """Create optimized config for BERT models."""
        return (
            cls()
            .rank(r)
            .alpha(r * 2)
            .target_modules(["query", "key", "value"])
            .dropout(0.1)
            .task_type(TaskType.SEQ_CLS)
            .build()
        )


# =============================================================================
# TASK 2: Quantization Configuration
# =============================================================================
@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def to_bnb_config(self) -> BitsAndBytesConfig:
        """Convert to BitsAndBytesConfig."""
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )

    @classmethod
    def get_4bit_config(cls) -> "QuantizationConfig":
        """Get standard 4-bit NF4 quantization config."""
        return cls(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    @classmethod
    def get_8bit_config(cls) -> "QuantizationConfig":
        """Get standard 8-bit quantization config."""
        return cls(load_in_8bit=True)

    def estimate_memory(self, model_params: int) -> dict[str, float]:
        """Estimate memory usage in GB."""
        params_billions = model_params / 1e9

        if self.load_in_4bit:
            model_memory = params_billions * 0.5  # ~4 bits per param
            overhead = 0.5
        elif self.load_in_8bit:
            model_memory = params_billions * 1.0  # ~8 bits per param
            overhead = 0.3
        else:
            model_memory = params_billions * 4.0  # FP32
            overhead = 0.2

        return {
            "model_memory_gb": model_memory,
            "overhead_gb": overhead,
            "total_gb": model_memory + overhead,
            "bytes_per_param": (
                0.5 if self.load_in_4bit else (1.0 if self.load_in_8bit else 4.0)
            ),
        }


# =============================================================================
# TASK 3: PEFT Model Manager
# =============================================================================
class PEFTModelManager:
    """Manage PEFT models: loading, saving, and adapter operations."""

    def __init__(
        self,
        base_model_name: str,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        """Initialize with base model."""
        self.base_model_name = base_model_name
        self.quantization_config = quantization_config
        self.base_model = None
        self.peft_model = None
        self.tokenizer = None
        self.active_adapters: list[str] = []

    def load_base_model(self, for_training: bool = False) -> None:
        """Load the base model with optional quantization."""
        kwargs = {"trust_remote_code": True}

        if self.quantization_config:
            kwargs["quantization_config"] = self.quantization_config.to_bnb_config()
            kwargs["device_map"] = "auto"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if for_training and self.quantization_config:
            self.base_model = prepare_model_for_kbit_training(self.base_model)

    def apply_lora(self, lora_config: LoraConfig) -> None:
        """Apply LoRA to the base model."""
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")

        self.peft_model = get_peft_model(self.base_model, lora_config)
        self.active_adapters.append("default")

    def save_adapter(self, path: str, adapter_name: str = "default") -> None:
        """Save a specific adapter."""
        if self.peft_model is None:
            raise ValueError("No PEFT model. Call apply_lora first.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(save_path))

    def load_adapter(self, path: str, adapter_name: str) -> None:
        """Load an adapter from disk."""
        if self.peft_model is not None:
            self.peft_model.load_adapter(path, adapter_name)
        else:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model, path, adapter_name=adapter_name
            )
        self.active_adapters.append(adapter_name)

    def set_active_adapter(self, adapter_name: str) -> None:
        """Set the active adapter for inference."""
        if self.peft_model is None:
            raise ValueError("No PEFT model loaded.")
        self.peft_model.set_adapter(adapter_name)

    def merge_and_unload(self) -> None:
        """Merge LoRA weights into base model."""
        if self.peft_model is None:
            raise ValueError("No PEFT model to merge.")

        self.base_model = self.peft_model.merge_and_unload()
        self.peft_model = None
        self.active_adapters.clear()

    def get_trainable_parameters(self) -> dict[str, int]:
        """Get count of trainable vs total parameters."""
        model = self.peft_model or self.base_model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "percentage": 100 * trainable / total,
        }


# =============================================================================
# TASK 4: QLoRA Trainer
# =============================================================================
class QLoRATrainer:
    """Train models using QLoRA (Quantization + LoRA)."""

    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        quantization_config: Optional[QuantizationConfig] = None,
        output_dir: str = "./qlora_output",
    ):
        """Initialize QLoRA trainer."""
        self.model_name = model_name
        self.lora_config = lora_config
        self.quantization_config = (
            quantization_config or QuantizationConfig.get_4bit_config()
        )
        self.output_dir = output_dir

        self.manager = PEFTModelManager(model_name, self.quantization_config)
        self.trainer: Optional[Trainer] = None

    def setup(self) -> None:
        """Set up model with quantization and LoRA."""
        self.manager.load_base_model(for_training=True)
        self.manager.apply_lora(self.lora_config)

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        training_args: Optional[TrainingArguments] = None,
    ) -> dict:
        """Train the model."""
        if self.manager.peft_model is None:
            raise ValueError("Model not set up. Call setup first.")

        if training_args is None:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=2e-4,
                fp16=True,
                save_strategy="epoch",
                logging_steps=10,
            )

        self.trainer = Trainer(
            model=self.manager.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.manager.tokenizer,
        )

        result = self.trainer.train()
        return result.metrics

    def save(self, path: Optional[str] = None) -> None:
        """Save the trained adapter."""
        save_path = path or self.output_dir
        self.manager.save_adapter(save_path)

    def get_model(self):
        """Get the PEFT model."""
        return self.manager.peft_model


# =============================================================================
# TASK 5: Adapter Combiner
# =============================================================================
class AdapterCombiner:
    """Combine multiple LoRA adapters."""

    def __init__(self, base_model_name: str):
        """Initialize with base model."""
        self.base_model_name = base_model_name
        self.manager = PEFTModelManager(base_model_name)
        self.loaded_adapters: dict[str, str] = {}  # name -> path

    def load_base(self) -> None:
        """Load the base model."""
        self.manager.load_base_model()

    def add_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """Add an adapter from disk."""
        self.manager.load_adapter(adapter_path, adapter_name)
        self.loaded_adapters[adapter_name] = adapter_path

    def combine_adapters(
        self, adapter_names: list[str], weights: list[float], new_adapter_name: str
    ) -> None:
        """Combine multiple adapters with weights (weighted average)."""
        if self.manager.peft_model is None:
            raise ValueError("No adapters loaded.")

        # Use PEFT's add_weighted_adapter if available
        try:
            self.manager.peft_model.add_weighted_adapter(
                adapters=adapter_names,
                weights=weights,
                adapter_name=new_adapter_name,
                combination_type="linear",
            )
            self.loaded_adapters[new_adapter_name] = "combined"
        except AttributeError:
            # Fallback: just use the first adapter
            self.manager.set_active_adapter(adapter_names[0])

    def switch_adapter(self, adapter_name: str) -> None:
        """Switch to a specific adapter."""
        self.manager.set_active_adapter(adapter_name)

    def list_adapters(self) -> list[str]:
        """List all loaded adapters."""
        return list(self.loaded_adapters.keys())


# =============================================================================
# TASK 6: Quantized Inference Pipeline
# =============================================================================
class QuantizedInferencePipeline:
    """Run inference with quantized models."""

    def __init__(
        self,
        model_name: str,
        quantization_config: Optional[QuantizationConfig] = None,
        adapter_path: Optional[str] = None,
    ):
        """Initialize inference pipeline."""
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model with quantization and optional adapter."""
        kwargs = {}
        if self.quantization_config:
            kwargs["quantization_config"] = self.quantization_config.to_bnb_config()
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_generate(
        self, prompts: list[str], max_new_tokens: int = 100, **kwargs
    ) -> list[str]:
        """Generate text for multiple prompts."""
        return [
            self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            for prompt in prompts
        ]


# =============================================================================
# TASK 7: LoRA Hyperparameter Search
# =============================================================================
class LoRAHyperparameterSearch:
    """Search for optimal LoRA hyperparameters."""

    def __init__(
        self,
        model_name: str,
        train_dataset,
        eval_dataset,
        output_dir: str = "./lora_search",
    ):
        """Initialize hyperparameter search."""
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def get_search_space(self) -> list[dict]:
        """Define hyperparameter search space."""
        return [
            {"r": 4, "alpha": 8, "dropout": 0.05},
            {"r": 8, "alpha": 16, "dropout": 0.05},
            {"r": 8, "alpha": 16, "dropout": 0.1},
            {"r": 16, "alpha": 32, "dropout": 0.05},
            {"r": 16, "alpha": 32, "dropout": 0.1},
            {"r": 32, "alpha": 64, "dropout": 0.1},
        ]

    def run_trial(self, params: dict, trial_num: int) -> dict:
        """Run a single hyperparameter trial."""
        lora_config = LoraConfig(
            r=params["r"],
            lora_alpha=params["alpha"],
            lora_dropout=params["dropout"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        )

        trial_dir = self.output_dir / f"trial_{trial_num}"

        trainer = QLoRATrainer(
            model_name=self.model_name,
            lora_config=lora_config,
            output_dir=str(trial_dir),
        )
        trainer.setup()

        training_args = TrainingArguments(
            output_dir=str(trial_dir),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
        )

        metrics = trainer.train(self.train_dataset, self.eval_dataset, training_args)

        # Clean up
        del trainer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {"trial": trial_num, "params": params, "metrics": metrics}

    def search(self, n_trials: Optional[int] = None) -> list[dict]:
        """Run hyperparameter search."""
        search_space = self.get_search_space()
        if n_trials:
            search_space = search_space[:n_trials]

        for i, params in enumerate(search_space):
            result = self.run_trial(params, i)
            self.results.append(result)

        return self.results

    def get_best_params(self, metric: str = "eval_loss", minimize: bool = True) -> dict:
        """Get the best hyperparameters based on metric."""
        if not self.results:
            raise ValueError("No results. Run search first.")

        best = None
        best_value = None

        for result in self.results:
            value = result["metrics"].get(metric)
            if value is None:
                continue

            if best_value is None:
                best = result
                best_value = value
            elif minimize and value < best_value:
                best = result
                best_value = value
            elif not minimize and value > best_value:
                best = result
                best_value = value

        return best["params"] if best else {}


# =============================================================================
# TASK 8: Model Profiler
# =============================================================================
class ModelProfiler:
    """Profile model memory and compute requirements."""

    def __init__(self, model_name: str):
        """Initialize profiler."""
        self.model_name = model_name
        self.profile_results: dict = {}

    def profile_memory(
        self, quantization_config: Optional[QuantizationConfig] = None
    ) -> dict:
        """Profile memory usage."""
        # Load tokenizer to get model config
        config = AutoConfig.from_pretrained(self.model_name)

        # Estimate parameters
        hidden_size = getattr(config, "hidden_size", 768)
        num_layers = getattr(config, "num_hidden_layers", 12)
        vocab_size = getattr(config, "vocab_size", 30000)

        # Rough parameter estimate
        params = (
            vocab_size * hidden_size  # Embeddings
            + num_layers
            * (
                4 * hidden_size * hidden_size + 4 * hidden_size * hidden_size
            )  # Attention
            + num_layers * (4 * hidden_size * hidden_size * 4)  # FFN
        )

        if quantization_config:
            memory_info = quantization_config.estimate_memory(params)
        else:
            memory_info = {
                "model_memory_gb": params * 4 / 1e9,  # FP32
                "bytes_per_param": 4.0,
            }

        self.profile_results["memory"] = {"estimated_params": params, **memory_info}
        return self.profile_results["memory"]

    def profile_lora_overhead(self, lora_config: LoraConfig) -> dict:
        """Profile LoRA parameter overhead."""
        config = AutoConfig.from_pretrained(self.model_name)

        hidden_size = getattr(config, "hidden_size", 768)
        num_layers = getattr(config, "num_hidden_layers", 12)

        # LoRA params per layer (for attention)
        lora_params_per_module = 2 * hidden_size * lora_config.r
        num_target_modules = (
            len(lora_config.target_modules) if lora_config.target_modules else 4
        )

        total_lora_params = num_layers * num_target_modules * lora_params_per_module

        # Base model params (estimate)
        base_params = num_layers * 4 * hidden_size * hidden_size * 2

        self.profile_results["lora"] = {
            "lora_params": total_lora_params,
            "base_params_estimate": base_params,
            "percentage": 100 * total_lora_params / base_params,
        }
        return self.profile_results["lora"]

    def get_report(self) -> str:
        """Generate a profiling report."""
        lines = [f"Model Profile: {self.model_name}", "=" * 50]

        if "memory" in self.profile_results:
            mem = self.profile_results["memory"]
            lines.append(
                f"Estimated Parameters: {mem.get('estimated_params', 'N/A'):,}"
            )
            lines.append(f"Model Memory: {mem.get('model_memory_gb', 'N/A'):.2f} GB")

        if "lora" in self.profile_results:
            lora = self.profile_results["lora"]
            lines.append(f"LoRA Parameters: {lora.get('lora_params', 'N/A'):,}")
            lines.append(f"LoRA Overhead: {lora.get('percentage', 'N/A'):.2f}%")

        return "\n".join(lines)


# =============================================================================
# TASK 9: PEFT Model Exporter
# =============================================================================
class PEFTExporter:
    """Export PEFT models in various formats."""

    def __init__(self, peft_model, tokenizer):
        """Initialize exporter with model and tokenizer."""
        self.peft_model = peft_model
        self.tokenizer = tokenizer

    def export_merged(self, output_dir: str) -> None:
        """Export merged model (LoRA weights baked in)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge and save
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

    def export_adapter_only(self, output_dir: str) -> None:
        """Export only the adapter weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(output_path))

    def export_for_inference(self, output_dir: str, merge: bool = True) -> None:
        """Export optimized for inference."""
        if merge:
            self.export_merged(output_dir)
        else:
            self.export_adapter_only(output_dir)

        # Save inference config
        config = {
            "model_type": "merged" if merge else "adapter",
            "timestamp": time.time(),
        }
        with open(Path(output_dir) / "inference_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def get_export_size(self, merged: bool = False) -> dict:
        """Estimate export size."""
        if merged:
            params = sum(p.numel() for p in self.peft_model.parameters())
        else:
            params = sum(
                p.numel()
                for n, p in self.peft_model.named_parameters()
                if "lora" in n.lower()
            )

        return {
            "parameters": params,
            "fp32_size_mb": params * 4 / 1e6,
            "fp16_size_mb": params * 2 / 1e6,
        }


# =============================================================================
# TASK 10: Multi-Adapter Model
# =============================================================================
class AdapterType(Enum):
    """Types of adapters."""

    LORA = "lora"
    PREFIX = "prefix"
    PROMPT = "prompt"


class MultiAdapterModel:
    """Model with multiple switchable adapters for different tasks."""

    def __init__(self, base_model_name: str):
        """Initialize with base model."""
        self.base_model_name = base_model_name
        self.base_model = None
        self.tokenizer = None
        self.adapters: dict[str, dict] = {}  # name -> {type, config, path}
        self.current_adapter: Optional[str] = None
        self.peft_model = None

    def load_base(
        self, quantization_config: Optional[QuantizationConfig] = None
    ) -> None:
        """Load the base model."""
        kwargs = {}
        if quantization_config:
            kwargs["quantization_config"] = quantization_config.to_bnb_config()
            kwargs["device_map"] = "auto"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_adapter(
        self,
        name: str,
        adapter_type: AdapterType,
        config: LoraConfig,
        train: bool = False,
    ) -> None:
        """Add a new adapter."""
        if self.base_model is None:
            raise ValueError("Base model not loaded.")

        if self.peft_model is None:
            self.peft_model = get_peft_model(self.base_model, config, adapter_name=name)
        else:
            self.peft_model.add_adapter(name, config)

        self.adapters[name] = {"type": adapter_type, "config": config, "trained": train}
        self.current_adapter = name

    def load_adapter(self, path: str, name: str) -> None:
        """Load a pre-trained adapter."""
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model, path, adapter_name=name
            )
        else:
            self.peft_model.load_adapter(path, name)

        self.adapters[name] = {"type": AdapterType.LORA, "path": path, "trained": True}

    def switch_adapter(self, name: str) -> None:
        """Switch to a specific adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found.")

        if self.peft_model:
            self.peft_model.set_adapter(name)
        self.current_adapter = name

    def disable_adapters(self) -> None:
        """Disable all adapters (use base model)."""
        if self.peft_model:
            self.peft_model.disable_adapter_layers()
        self.current_adapter = None

    def enable_adapters(self) -> None:
        """Re-enable adapters."""
        if self.peft_model:
            self.peft_model.enable_adapter_layers()

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text using current adapter."""
        model = self.peft_model or self.base_model

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def list_adapters(self) -> list[str]:
        """List all available adapters."""
        return list(self.adapters.keys())

    def get_adapter_info(self, name: str) -> dict:
        """Get information about an adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found.")
        return self.adapters[name]


if __name__ == "__main__":
    print("Week 11 - Exercise 3 Solutions: PEFT and Quantization")
    print("=" * 60)

    # Test LoRAConfigBuilder
    print("\n1. LoRAConfigBuilder:")
    config = (
        LoRAConfigBuilder()
        .rank(8)
        .alpha(16)
        .target_modules(["q_proj", "v_proj"])
        .dropout(0.1)
        .build()
    )
    print(f"   Config: r={config.r}, alpha={config.lora_alpha}")

    # Test preset configs
    llama_config = LoRAConfigBuilder.for_llama(r=16)
    print(f"   Llama preset: r={llama_config.r}, targets={llama_config.target_modules}")

    # Test QuantizationConfig
    print("\n2. QuantizationConfig:")
    quant = QuantizationConfig.get_4bit_config()
    print(f"   4-bit config: load_in_4bit={quant.load_in_4bit}")

    memory = quant.estimate_memory(7_000_000_000)  # 7B params
    print(f"   7B model memory: {memory['model_memory_gb']:.1f} GB")

    # Test ModelProfiler
    print("\n3. ModelProfiler:")
    profiler = ModelProfiler("gpt2")
    mem_profile = profiler.profile_memory()
    print(f"   GPT-2 estimated params: {mem_profile['estimated_params']:,}")

    print("\n✅ All solutions implemented!")

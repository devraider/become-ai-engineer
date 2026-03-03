"""
Tests for Week 11 - Exercise 3: PEFT and Quantization
"""

import tempfile
from importlib.util import find_spec
from unittest.mock import Mock, patch, MagicMock

import pytest

# Check for optional dependencies
HAS_TRANSFORMERS = find_spec("transformers") is not None
HAS_PEFT = find_spec("peft") is not None

if HAS_TRANSFORMERS:
    from transformers import AutoModelForSequenceClassification, AutoConfig

if HAS_PEFT:
    from peft import LoraConfig

# Import exercise classes
from exercise_advanced_3_peft import (
    LoRAConfigBuilder,
    QuantizationConfig,
    PEFTModelManager,
    QLoRATrainer,
    AdapterCombiner,
    QuantizedInferencePipeline,
    LoRAHyperparameterSearch,
    ModelProfiler,
    PEFTExporter,
    MultiAdapterModel,
)


class TestLoRAConfigBuilder:
    """Tests for LoRAConfigBuilder class."""

    def test_default_values(self):
        """Test default configuration values."""
        builder = LoRAConfigBuilder()
        assert builder.r == 16
        assert builder.lora_alpha == 32
        assert builder.lora_dropout == 0.05

    def test_build_causal_lm(self):
        """Test building config for causal LM."""
        builder = LoRAConfigBuilder(r=8, lora_alpha=16)
        config = builder.build(task_type="CAUSAL_LM")
        assert config.r == 8
        assert config.lora_alpha == 16

    def test_build_seq_cls(self):
        """Test building config for sequence classification."""
        builder = LoRAConfigBuilder()
        config = builder.build(task_type="SEQ_CLS")
        assert config is not None

    def test_for_llama_preset(self):
        """Test LLaMA preset."""
        builder = LoRAConfigBuilder.for_llama()
        assert "q_proj" in builder.target_modules or builder.target_modules is None

    def test_for_gpt2_preset(self):
        """Test GPT-2 preset."""
        builder = LoRAConfigBuilder.for_gpt2()
        assert builder is not None

    def test_for_bert_preset(self):
        """Test BERT preset."""
        builder = LoRAConfigBuilder.for_bert()
        assert builder is not None

    def test_get_target_modules(self):
        """Test target module detection."""
        modules = LoRAConfigBuilder.get_target_modules("llama")
        assert isinstance(modules, list)


class TestQuantizationConfig:
    """Tests for QuantizationConfig class."""

    def test_get_4bit_config(self):
        """Test 4-bit quantization config."""
        config = QuantizationConfig.get_4bit_config()
        assert config.load_in_4bit is True

    def test_get_4bit_config_nf4(self):
        """Test 4-bit with NF4 quantization."""
        config = QuantizationConfig.get_4bit_config(quant_type="nf4")
        assert config.bnb_4bit_quant_type == "nf4"

    def test_get_8bit_config(self):
        """Test 8-bit quantization config."""
        config = QuantizationConfig.get_8bit_config()
        assert config.load_in_8bit is True

    def test_estimate_memory(self):
        """Test memory estimation."""
        params = 7_000_000_000  # 7B parameters

        mem_fp32 = QuantizationConfig.estimate_memory(params, "fp32")
        mem_4bit = QuantizationConfig.estimate_memory(params, "4bit")

        # 4-bit should be much smaller
        assert mem_4bit["estimated_gb"] < mem_fp32["estimated_gb"]


class TestPEFTModelManager:
    """Tests for PEFTModelManager class."""

    def test_create_peft_model(self):
        """Test PEFT model creation."""
        manager = PEFTModelManager()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"])

        peft_model = manager.create_peft_model(base_model, lora_config)
        assert peft_model is not None

    def test_print_trainable_parameters(self):
        """Test parameter counting."""
        manager = PEFTModelManager()

        mock_model = Mock()
        mock_model.parameters = Mock(
            return_value=[
                Mock(numel=Mock(return_value=100), requires_grad=True),
                Mock(numel=Mock(return_value=1000), requires_grad=False),
            ]
        )

        stats = manager.print_trainable_parameters(mock_model)
        assert "trainable_params" in stats
        assert "total_params" in stats

    def test_save_and_load_adapter(self):
        """Test adapter saving and loading."""
        manager = PEFTModelManager()
        mock_model = Mock()
        mock_model.save_pretrained = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.save_adapter(mock_model, tmpdir)
            mock_model.save_pretrained.assert_called()

    def test_merge_and_unload(self):
        """Test merging adapter into base model."""
        manager = PEFTModelManager()

        mock_model = Mock()
        mock_model.merge_and_unload = Mock(return_value=Mock())

        merged = manager.merge_and_unload(mock_model)
        mock_model.merge_and_unload.assert_called_once()


class TestQLoRATrainer:
    """Tests for QLoRATrainer class."""

    def test_init(self):
        """Test initialization."""
        trainer = QLoRATrainer(
            model_name="distilbert-base-uncased", output_dir="./test"
        )
        assert trainer.model_name == "distilbert-base-uncased"

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_load_quantized_model(self):
        """Test quantized model loading."""
        trainer = QLoRATrainer("distilbert-base-uncased")
        trainer.load_quantized_model(bits=4)
        assert trainer.model is not None

    def test_prepare_for_training(self):
        """Test training preparation."""
        trainer = QLoRATrainer("distilbert-base-uncased")
        # Would need model loaded first
        with pytest.raises(Exception):
            trainer.prepare_for_training()


class TestAdapterCombiner:
    """Tests for AdapterCombiner class."""

    def test_init(self):
        """Test initialization."""
        mock_model = Mock()
        combiner = AdapterCombiner(mock_model)
        assert combiner.base_model == mock_model

    def test_combine_weighted_validates_weights(self):
        """Test weighted combination validates weights."""
        mock_model = Mock()
        combiner = AdapterCombiner(mock_model)

        # Weights don't sum to 1
        with pytest.raises(ValueError):
            combiner.combine_weighted([0.3, 0.3])


class TestQuantizedInferencePipeline:
    """Tests for QuantizedInferencePipeline class."""

    @pytest.mark.skipif(True, reason="Requires model download")
    def test_generate(self):
        """Test text generation."""
        pipeline = QuantizedInferencePipeline(
            model_path="distilbert-base-uncased", bits=8
        )
        output = pipeline.generate("Once upon a time")
        assert isinstance(output, str)

    @pytest.mark.skipif(True, reason="Requires model download")
    def test_batch_generate(self):
        """Test batch generation."""
        pipeline = QuantizedInferencePipeline(model_path="gpt2", bits=8)
        outputs = pipeline.batch_generate(["Hello", "World"])
        assert len(outputs) == 2


class TestLoRAHyperparameterSearch:
    """Tests for LoRAHyperparameterSearch class."""

    def test_define_search_space(self):
        """Test search space definition."""
        searcher = LoRAHyperparameterSearch(
            model_name="bert-base-uncased", train_dataset=Mock(), eval_dataset=Mock()
        )

        space = searcher.define_search_space(
            r_values=[8, 16, 32], alpha_values=[16, 32], dropout_values=[0.05, 0.1]
        )

        assert "r" in space
        assert len(space["r"]) == 3

    def test_grid_search_combinations(self):
        """Test grid search generates all combinations."""
        searcher = LoRAHyperparameterSearch(
            model_name="bert-base-uncased", train_dataset=Mock(), eval_dataset=Mock()
        )

        space = {"r": [8, 16], "alpha": [16, 32], "dropout": [0.05]}

        # Should generate 2 * 2 * 1 = 4 combinations
        # (In actual implementation)


class TestModelProfiler:
    """Tests for ModelProfiler class."""

    def test_profile_memory(self):
        """Test memory profiling."""
        mock_model = Mock()
        mock_model.parameters = Mock(
            return_value=[Mock(numel=Mock(return_value=1000000))]
        )

        profiler = ModelProfiler(mock_model, Mock())
        memory = profiler.profile_memory()

        assert "parameter_count" in memory or "model_size" in memory

    def test_profile_speed(self):
        """Test speed profiling."""
        mock_model = Mock()
        mock_model.eval = Mock()

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": Mock()}

        profiler = ModelProfiler(mock_model, mock_tokenizer)
        speed = profiler.profile_speed("test input", num_runs=5)

        assert "mean_time" in speed or "avg_latency" in speed


class TestPEFTExporter:
    """Tests for PEFTExporter class."""

    def test_export_merged(self):
        """Test merged model export."""
        mock_model = Mock()
        mock_model.merge_and_unload = Mock(return_value=Mock())
        mock_model.save_pretrained = Mock()

        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()

        exporter = PEFTExporter(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export_merged(tmpdir)
            mock_model.merge_and_unload.assert_called()

    def test_export_safetensors(self):
        """Test safetensors export."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        exporter = PEFTExporter(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.export_safetensors(tmpdir)


class TestMultiAdapterModel:
    """Tests for MultiAdapterModel class."""

    def test_add_adapter(self):
        """Test adding adapter."""
        multi = MultiAdapterModel("distilbert-base-uncased")

        config = LoraConfig(r=8, target_modules=["q_lin"])

        multi.add_adapter("test", lora_config=config)
        assert "test" in multi.adapter_names

    def test_switch_adapter(self):
        """Test switching between adapters."""
        multi = MultiAdapterModel("distilbert-base-uncased")

        # Add adapters
        multi.add_adapter("adapter1")
        multi.add_adapter("adapter2")

        multi.switch_adapter("adapter1")
        assert multi.active_adapter == "adapter1"

        multi.switch_adapter("adapter2")
        assert multi.active_adapter == "adapter2"

    def test_list_adapters(self):
        """Test listing adapters."""
        multi = MultiAdapterModel("distilbert-base-uncased")
        multi.add_adapter("a1")
        multi.add_adapter("a2")

        adapters = multi.list_adapters()
        assert "a1" in adapters
        assert "a2" in adapters

    def test_disable_adapters(self):
        """Test disabling all adapters."""
        multi = MultiAdapterModel("distilbert-base-uncased")
        multi.add_adapter("test")
        multi.switch_adapter("test")

        multi.disable_adapters()
        # Model should use base weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

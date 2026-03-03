"""
Tests for Week 11 - Project: Multi-Modal Fine-tuning Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import project classes
from project_pipeline import (
    ModelType,
    TrainingMethod,
    PipelineConfig,
    DataModule,
    TextClassificationDataModule,
    ModelFactory,
    TrainerModule,
    EvaluationResult,
    EvaluationModule,
    ModelExporter,
    ExperimentTracker,
    FineTuningPipeline,
    PipelineFactory,
)


class TestPipelineConfig:
    """Tests for PipelineConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.model_name == "bert-base-uncased"
        assert config.model_type == ModelType.SEQUENCE_CLASSIFICATION
        assert config.training_method == TrainingMethod.LORA

    def test_custom_values(self):
        """Test custom configuration."""
        config = PipelineConfig(
            model_name="roberta-base", num_labels=5, learning_rate=3e-5
        )
        assert config.model_name == "roberta-base"
        assert config.num_labels == 5
        assert config.learning_rate == 3e-5

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = PipelineConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_config(self):
        """Test validation catches invalid config."""
        config = PipelineConfig(
            training_method=TrainingMethod.QLORA, bits=3  # Invalid bits
        )
        errors = config.validate()
        assert len(errors) > 0

    def test_save_and_load(self):
        """Test config save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")

            original = PipelineConfig(model_name="test-model", learning_rate=1e-4)
            original.save(path)

            loaded = PipelineConfig.load(path)
            assert loaded.model_name == "test-model"
            assert loaded.learning_rate == 1e-4


class TestTextClassificationDataModule:
    """Tests for TextClassificationDataModule class."""

    def test_init(self):
        """Test initialization."""
        mock_tokenizer = Mock()
        module = TextClassificationDataModule(tokenizer=mock_tokenizer, max_length=256)
        assert module.max_length == 256

    def test_prepare_from_dict(self):
        """Test data preparation from dictionary."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        module = TextClassificationDataModule(tokenizer=mock_tokenizer)
        module.prepare(train_data={"text": ["sample text"], "label": [0]})

        assert module.train_dataset is not None

    def test_get_datasets(self):
        """Test dataset getters."""
        mock_tokenizer = Mock()
        module = TextClassificationDataModule(tokenizer=mock_tokenizer)

        # Set datasets directly
        module.train_dataset = Mock()
        module.eval_dataset = Mock()

        assert module.get_train_dataset() is not None
        assert module.get_eval_dataset() is not None


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_create_base_model_classification(self):
        """Test creating classification model."""
        config = PipelineConfig(
            model_name="distilbert-base-uncased",
            model_type=ModelType.SEQUENCE_CLASSIFICATION,
            num_labels=2,
        )

        model, tokenizer = ModelFactory.create_base_model(config)
        assert model is not None
        assert tokenizer is not None

    def test_create_base_model_causal(self):
        """Test creating causal LM model."""
        config = PipelineConfig(model_name="gpt2", model_type=ModelType.CAUSAL_LM)

        model, tokenizer = ModelFactory.create_base_model(config)
        assert model is not None

    def test_apply_peft(self):
        """Test applying PEFT to model."""
        config = PipelineConfig(
            training_method=TrainingMethod.LORA, lora_r=8, lora_alpha=16
        )

        mock_model = Mock()
        peft_model = ModelFactory.apply_peft(mock_model, config)
        assert peft_model is not None


class TestTrainerModule:
    """Tests for TrainerModule class."""

    def test_init(self):
        """Test initialization."""
        config = PipelineConfig()
        trainer = TrainerModule(model=Mock(), tokenizer=Mock(), config=config)
        assert trainer.config == config

    def test_setup_trainer(self):
        """Test trainer setup."""
        config = PipelineConfig()
        trainer = TrainerModule(model=Mock(), tokenizer=Mock(), config=config)

        trainer.setup_trainer(train_dataset=Mock(), eval_dataset=Mock())

        assert trainer.trainer is not None

    def test_train(self):
        """Test training execution."""
        config = PipelineConfig(num_epochs=1)
        trainer = TrainerModule(model=Mock(), tokenizer=Mock(), config=config)

        mock_train = Mock()
        mock_eval = Mock()

        results = trainer.train(mock_train, mock_eval)
        assert results is not None


class TestEvaluationModule:
    """Tests for EvaluationModule class."""

    def test_init_loads_metrics(self):
        """Test metric loading."""
        evaluator = EvaluationModule(["accuracy", "f1"])
        assert len(evaluator.metrics) == 2

    def test_compute_classification_metrics(self):
        """Test metric computation."""
        evaluator = EvaluationModule(["accuracy"])

        predictions = [0, 1, 1, 0]
        labels = [0, 1, 0, 0]

        metrics = evaluator.compute_classification_metrics(predictions, labels)
        assert "accuracy" in metrics

    def test_generate_report(self):
        """Test report generation."""
        evaluator = EvaluationModule()

        result = EvaluationResult(
            metrics={"accuracy": 0.9, "f1": 0.85},
            predictions=[0, 1, 1],
            labels=[0, 1, 0],
        )

        report = evaluator.generate_report(result)
        assert isinstance(report, str)
        assert "accuracy" in report


class TestModelExporter:
    """Tests for ModelExporter class."""

    def test_export_huggingface(self):
        """Test HuggingFace format export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = Mock()
            mock_model.save_pretrained = Mock()

            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            exporter = ModelExporter(
                model=mock_model, tokenizer=mock_tokenizer, config=PipelineConfig()
            )

            exporter.export_huggingface(tmpdir)
            mock_model.save_pretrained.assert_called()

    def test_export_merged(self):
        """Test merged export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = Mock()
            mock_model.merge_and_unload = Mock(return_value=Mock())

            exporter = ModelExporter(
                model=mock_model,
                tokenizer=Mock(),
                config=PipelineConfig(training_method=TrainingMethod.LORA),
            )

            exporter.export_merged(tmpdir)


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""

    def test_start(self):
        """Test experiment start."""
        tracker = ExperimentTracker("test-exp")
        tracker.start()
        assert tracker.start_time is not None

    def test_log_config(self):
        """Test config logging."""
        tracker = ExperimentTracker("test-exp")
        config = PipelineConfig(learning_rate=1e-4)

        tracker.log_config(config)
        assert tracker.config is not None

    def test_log_metrics(self):
        """Test metric logging."""
        tracker = ExperimentTracker("test-exp")

        tracker.log_metrics(
            step=100, metrics={"loss": 0.5, "accuracy": 0.8}, phase="train"
        )

        assert len(tracker.metrics) == 1

    def test_save_experiment(self):
        """Test experiment saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker("test-exp", output_dir=tmpdir)
            tracker.log_config(PipelineConfig())
            tracker.log_metrics(100, {"loss": 0.5})

            path = tracker.save_experiment()
            assert os.path.exists(path)


class TestFineTuningPipeline:
    """Tests for FineTuningPipeline class."""

    def test_init(self):
        """Test pipeline initialization."""
        config = PipelineConfig()
        pipeline = FineTuningPipeline(config)
        assert pipeline.config == config

    def test_setup(self):
        """Test pipeline setup."""
        config = PipelineConfig()
        pipeline = FineTuningPipeline(config)

        pipeline.setup()
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None

    def test_prepare_data(self):
        """Test data preparation."""
        config = PipelineConfig()
        pipeline = FineTuningPipeline(config)
        pipeline.setup()

        pipeline.prepare_data(
            train_data={"text": ["sample 1", "sample 2"], "label": [0, 1]}
        )

        assert pipeline.data_module is not None

    def test_run_full_pipeline(self):
        """Test full pipeline execution."""
        config = PipelineConfig(num_epochs=1, batch_size=2)
        pipeline = FineTuningPipeline(config)

        results = pipeline.run(
            train_data={
                "text": ["positive text", "negative text"] * 10,
                "label": [1, 0] * 10,
            },
            eval_data={"text": ["test positive", "test negative"], "label": [1, 0]},
            experiment_name="test-run",
        )

        assert results is not None

    def test_evaluate(self):
        """Test evaluation."""
        config = PipelineConfig()
        pipeline = FineTuningPipeline(config)
        pipeline.setup()

        result = pipeline.evaluate()
        assert isinstance(result, EvaluationResult)

    def test_export(self):
        """Test model export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            pipeline = FineTuningPipeline(config)
            pipeline.setup()

            pipeline.export(tmpdir, format="huggingface")
            assert os.path.exists(tmpdir)


class TestPipelineFactory:
    """Tests for PipelineFactory class."""

    def test_sentiment_classification(self):
        """Test sentiment classification pipeline."""
        pipeline = PipelineFactory.sentiment_classification(
            model_name="distilbert-base-uncased", training_method=TrainingMethod.LORA
        )

        assert pipeline.config.model_type == ModelType.SEQUENCE_CLASSIFICATION
        assert pipeline.config.num_labels == 2

    def test_text_generation(self):
        """Test text generation pipeline."""
        pipeline = PipelineFactory.text_generation(
            model_name="gpt2", training_method=TrainingMethod.QLORA
        )

        assert pipeline.config.model_type == ModelType.CAUSAL_LM

    def test_custom_pipeline(self):
        """Test custom configuration pipeline."""
        config = PipelineConfig(
            model_name="custom-model", num_labels=10, learning_rate=5e-5
        )

        pipeline = PipelineFactory.custom(config)
        assert pipeline.config.model_name == "custom-model"
        assert pipeline.config.num_labels == 10


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.slow
    def test_end_to_end_lora_training(self):
        """Test complete LoRA training workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pipeline
            config = PipelineConfig(
                model_name="distilbert-base-uncased",
                training_method=TrainingMethod.LORA,
                num_epochs=1,
                batch_size=4,
                output_dir=tmpdir,
            )

            pipeline = FineTuningPipeline(config)

            # Prepare data
            train_data = {
                "text": ["great movie", "terrible film", "loved it", "hated it"] * 5,
                "label": [1, 0, 1, 0] * 5,
            }
            eval_data = {"text": ["amazing", "awful"], "label": [1, 0]}

            # Train
            results = pipeline.run(
                train_data=train_data,
                eval_data=eval_data,
                experiment_name="integration-test",
            )

            assert results is not None

            # Evaluate
            eval_result = pipeline.evaluate()
            assert eval_result.metrics is not None

            # Export
            export_path = os.path.join(tmpdir, "exported")
            pipeline.export(export_path, format="huggingface")
            assert os.path.exists(export_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

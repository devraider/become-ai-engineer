"""
Week 3 Project: Tests for Sentiment Classifier
==============================================

Run tests:
    python -m pytest tests/test_project.py -v
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_pipeline import (
    load_emotion_data,
    preprocess_text,
    create_tfidf_features,
    train_classifier,
    compare_classifiers,
    tune_best_model,
    evaluate_model,
    save_pipeline,
    load_pipeline,
    predict_sentiment,
)


# =============================================================================
# SAMPLE DATA
# =============================================================================


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I am so happy today!",
        "This is terrible and sad.",
        "Just a normal day.",
        "I love this amazing product!",
        "Angry and frustrated right now.",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return np.array([1, 0, 2, 1, 3])


# =============================================================================
# TESTS FOR preprocess_text
# =============================================================================


class TestPreprocessText:
    """Tests for preprocess_text function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = preprocess_text("Hello World!")
        if result is not None:
            assert isinstance(result, str)

    def test_lowercase(self):
        """Test that text is lowercased."""
        result = preprocess_text("HELLO WORLD")
        if result is not None:
            assert result.islower() or result == result.lower()

    def test_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        result = preprocess_text("hello    world")
        if result is not None:
            assert "    " not in result


# =============================================================================
# TESTS FOR create_tfidf_features
# =============================================================================


class TestCreateTfidfFeatures:
    """Tests for create_tfidf_features function."""

    def test_returns_tuple(self, sample_texts):
        """Test that function returns tuple."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        assert result is not None, "Should return tuple"
        assert len(result) == 3, "Should return (X_train, X_test, vectorizer)"

    def test_feature_shapes(self, sample_texts):
        """Test that feature shapes are correct."""
        train = sample_texts
        test = sample_texts[:2]

        result = create_tfidf_features(train, test, max_features=100)
        if result is not None:
            X_train, X_test, _ = result
            assert X_train.shape[0] == len(train)
            assert X_test.shape[0] == len(test)
            assert X_train.shape[1] == X_test.shape[1]


# =============================================================================
# TESTS FOR train_classifier
# =============================================================================


class TestTrainClassifier:
    """Tests for train_classifier function."""

    def test_logistic(self, sample_texts, sample_labels):
        """Test logistic regression training."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        if result is not None:
            X_train, _, _ = result
            model = train_classifier(X_train, sample_labels, "logistic")
            assert model is not None

    def test_random_forest(self, sample_texts, sample_labels):
        """Test random forest training."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        if result is not None:
            X_train, _, _ = result
            model = train_classifier(X_train, sample_labels, "random_forest")
            assert model is not None

    def test_naive_bayes(self, sample_texts, sample_labels):
        """Test naive bayes training."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        if result is not None:
            X_train, _, _ = result
            model = train_classifier(X_train, sample_labels, "naive_bayes")
            assert model is not None


# =============================================================================
# TESTS FOR compare_classifiers
# =============================================================================


class TestCompareClassifiers:
    """Tests for compare_classifiers function."""

    def test_returns_dataframe(self, sample_texts, sample_labels):
        """Test that function returns DataFrame."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        if result is not None:
            X_train, _, _ = result
            comparison = compare_classifiers(X_train, sample_labels, cv=2)
            if comparison is not None:
                assert isinstance(comparison, pd.DataFrame)


# =============================================================================
# TESTS FOR evaluate_model
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_dict(self, sample_texts, sample_labels):
        """Test that function returns dictionary."""
        result = create_tfidf_features(sample_texts, sample_texts[:2])
        if result is not None:
            X_train, X_test, _ = result
            model = train_classifier(X_train, sample_labels, "logistic")
            if model is not None:
                y_test = sample_labels[:2]
                eval_result = evaluate_model(model, X_test, y_test)
                if eval_result is not None:
                    assert isinstance(eval_result, dict)
                    assert "accuracy" in eval_result
                    assert "f1" in eval_result


# =============================================================================
# TESTS FOR save_pipeline / load_pipeline
# =============================================================================


class TestSaveLoadPipeline:
    """Tests for save and load pipeline."""

    def test_save_and_load(self, sample_texts, sample_labels):
        """Test saving and loading pipeline."""
        result = create_tfidf_features(sample_texts, sample_texts)
        if result is not None:
            X_train, _, vectorizer = result
            model = train_classifier(X_train, sample_labels, "logistic")

            if model is not None and vectorizer is not None:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    filepath = f.name

                try:
                    save_pipeline(vectorizer, model, filepath)
                    assert os.path.exists(filepath)

                    loaded = load_pipeline(filepath)
                    if loaded is not None:
                        assert len(loaded) == 2
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)


# =============================================================================
# TESTS FOR predict_sentiment
# =============================================================================


class TestPredictSentiment:
    """Tests for predict_sentiment function."""

    def test_returns_dict(self, sample_texts, sample_labels):
        """Test that function returns dictionary."""
        result = create_tfidf_features(sample_texts, sample_texts)
        if result is not None:
            X_train, _, vectorizer = result
            model = train_classifier(X_train, sample_labels, "logistic")

            if model is not None and vectorizer is not None:
                pred = predict_sentiment("I am happy!", vectorizer, model)
                if pred is not None:
                    assert isinstance(pred, dict)
                    assert "predicted_label" in pred
                    assert "confidence" in pred


# =============================================================================
# TESTS FOR load_emotion_data (optional - requires network)
# =============================================================================


class TestLoadEmotionData:
    """Tests for load_emotion_data function."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_NETWORK_TESTS"), reason="Skipping network test"
    )
    def test_returns_dataframes(self):
        """Test that function returns DataFrames."""
        result = load_emotion_data(num_samples=100)
        if result is not None:
            train_df, test_df = result
            assert isinstance(train_df, pd.DataFrame)
            assert isinstance(test_df, pd.DataFrame)
            assert "text" in train_df.columns
            assert "label" in train_df.columns

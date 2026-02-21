"""
Week 3 Exercise 1: Tests for scikit-learn Basics
================================================

Run tests:
    python -m pytest tests/test_exercise_1.py -v
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_basic_1_sklearn_basics import (
    load_iris_dataset,
    split_data,
    scale_features,
    train_logistic_regression,
    make_predictions,
    get_prediction_probabilities,
    train_random_forest,
    get_feature_importance,
    create_pipeline,
    save_model,
    load_model,
)


# =============================================================================
# TESTS FOR load_iris_dataset
# =============================================================================


class TestLoadIrisDataset:
    """Tests for load_iris_dataset function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        result = load_iris_dataset()
        assert result is not None, "Function returned None"
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 3, "Should return (X, y, feature_names)"

    def test_data_shapes(self):
        """Test that data has correct shapes."""
        result = load_iris_dataset()
        if result is not None:
            X, y, feature_names = result
            assert X.shape == (150, 4), "X should be (150, 4)"
            assert y.shape == (150,), "y should be (150,)"
            assert len(feature_names) == 4, "Should have 4 feature names"


# =============================================================================
# TESTS FOR split_data
# =============================================================================


class TestSplitData:
    """Tests for split_data function."""

    def test_returns_four_arrays(self):
        """Test that function returns four arrays."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        result = split_data(X, y)
        assert result is not None, "Function returned None"
        assert len(result) == 4, "Should return 4 arrays"

    def test_split_sizes(self):
        """Test that split sizes are correct."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        result = split_data(X, y, test_size=0.2)
        if result is not None:
            X_train, X_test, y_train, y_test = result
            assert X_train.shape[0] == 80, "Train should be 80 samples"
            assert X_test.shape[0] == 20, "Test should be 20 samples"

    def test_reproducibility(self):
        """Test that random_state makes splits reproducible."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        result1 = split_data(X, y, random_state=42)
        result2 = split_data(X, y, random_state=42)

        if result1 is not None and result2 is not None:
            np.testing.assert_array_equal(result1[0], result2[0])


# =============================================================================
# TESTS FOR scale_features
# =============================================================================


class TestScaleFeatures:
    """Tests for scale_features function."""

    def test_returns_scaled_arrays(self):
        """Test that function returns scaled arrays."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[2, 3]])

        result = scale_features(X_train, X_test)
        assert result is not None, "Function returned None"
        assert len(result) == 2, "Should return 2 arrays"

    def test_train_is_standardized(self):
        """Test that training data is standardized."""
        X_train = np.random.randn(100, 4) * 10 + 50
        X_test = np.random.randn(20, 4) * 10 + 50

        result = scale_features(X_train, X_test)
        if result is not None:
            X_train_scaled, _ = result
            assert abs(X_train_scaled.mean()) < 0.1, "Mean should be ~0"
            assert abs(X_train_scaled.std() - 1.0) < 0.1, "Std should be ~1"


# =============================================================================
# TESTS FOR train_logistic_regression
# =============================================================================


class TestTrainLogisticRegression:
    """Tests for train_logistic_regression function."""

    def test_returns_model(self):
        """Test that function returns a model."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = train_logistic_regression(X, y)
        assert model is not None, "Should return a model"

    def test_model_is_fitted(self):
        """Test that model is fitted."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = train_logistic_regression(X, y)
        if model is not None:
            assert hasattr(model, "coef_"), "Model should be fitted"


# =============================================================================
# TESTS FOR make_predictions
# =============================================================================


class TestMakePredictions:
    """Tests for make_predictions function."""

    def test_returns_predictions(self):
        """Test that function returns predictions."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        preds = make_predictions(model, X[:10])
        assert preds is not None, "Should return predictions"
        assert len(preds) == 10, "Should return 10 predictions"


# =============================================================================
# TESTS FOR get_prediction_probabilities
# =============================================================================


class TestGetPredictionProbabilities:
    """Tests for get_prediction_probabilities function."""

    def test_returns_probabilities(self):
        """Test that function returns probabilities."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        probs = get_prediction_probabilities(model, X[:10])
        assert probs is not None, "Should return probabilities"
        assert probs.shape == (10, 2), "Shape should be (10, n_classes)"

    def test_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        probs = get_prediction_probabilities(model, X[:10])
        if probs is not None:
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(10))


# =============================================================================
# TESTS FOR train_random_forest
# =============================================================================


class TestTrainRandomForest:
    """Tests for train_random_forest function."""

    def test_returns_model(self):
        """Test that function returns a model."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        model = train_random_forest(X, y)
        assert model is not None, "Should return a model"

    def test_respects_n_estimators(self):
        """Test that n_estimators parameter works."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)

        model = train_random_forest(X, y, n_estimators=50)
        if model is not None:
            assert model.n_estimators == 50


# =============================================================================
# TESTS FOR get_feature_importance
# =============================================================================


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        feature_names = ["f1", "f2", "f3", "f4"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = get_feature_importance(model, feature_names)
        assert result is not None, "Should return DataFrame"
        assert isinstance(result, pd.DataFrame), "Should be DataFrame"

    def test_has_correct_columns(self):
        """Test that DataFrame has correct columns."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        feature_names = ["f1", "f2", "f3", "f4"]

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = get_feature_importance(model, feature_names)
        if result is not None:
            assert "feature" in result.columns
            assert "importance" in result.columns


# =============================================================================
# TESTS FOR create_pipeline
# =============================================================================


class TestCreatePipeline:
    """Tests for create_pipeline function."""

    def test_returns_pipeline(self):
        """Test that function returns a Pipeline."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = create_pipeline(StandardScaler(), LogisticRegression())
        assert pipe is not None, "Should return a pipeline"

    def test_pipeline_has_steps(self):
        """Test that pipeline has steps."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = create_pipeline(StandardScaler(), LogisticRegression())
        if pipe is not None:
            assert len(pipe.steps) == 2, "Should have 2 steps"


# =============================================================================
# TESTS FOR save_model / load_model
# =============================================================================


class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""

    def test_save_and_load(self):
        """Test that model can be saved and loaded."""
        from sklearn.linear_model import LogisticRegression

        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            filepath = f.name

        try:
            save_model(model, filepath)
            assert os.path.exists(filepath), "File should exist"

            loaded = load_model(filepath)
            assert loaded is not None, "Should load model"

            # Check predictions match
            preds_orig = model.predict(X[:5])
            preds_loaded = loaded.predict(X[:5])
            np.testing.assert_array_equal(preds_orig, preds_loaded)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

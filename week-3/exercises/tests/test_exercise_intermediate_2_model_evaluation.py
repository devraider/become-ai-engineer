"""
Week 3 Exercise 2: Tests for Model Evaluation
=============================================

Run tests:
    python -m pytest tests/test_exercise_2.py -v
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_intermediate_2_model_evaluation import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    get_confusion_matrix,
    get_classification_report,
    cross_validate_model,
    compare_models,
    grid_search_cv,
    evaluate_model_full,
)


# =============================================================================
# SAMPLE DATA
# =============================================================================


@pytest.fixture
def binary_data():
    """Binary classification test data."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    return y_true, y_pred


@pytest.fixture
def multiclass_data():
    """Multiclass test data."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2])
    return y_true, y_pred


@pytest.fixture
def iris_data():
    """Iris dataset for model testing."""
    from sklearn.datasets import load_iris

    data = load_iris()
    return data.data, data.target


# =============================================================================
# TESTS FOR calculate_accuracy
# =============================================================================


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_returns_float(self, binary_data):
        """Test that function returns a float."""
        y_true, y_pred = binary_data
        result = calculate_accuracy(y_true, y_pred)
        assert result is not None, "Should return a value"
        assert isinstance(result, float), "Should return float"

    def test_correct_value(self, binary_data):
        """Test that accuracy is calculated correctly."""
        y_true, y_pred = binary_data
        result = calculate_accuracy(y_true, y_pred)
        if result is not None:
            # 6 correct out of 8
            assert abs(result - 0.75) < 0.01, "Accuracy should be 0.75"

    def test_perfect_accuracy(self):
        """Test perfect predictions."""
        y = np.array([0, 1, 2, 0, 1])
        result = calculate_accuracy(y, y)
        if result is not None:
            assert result == 1.0, "Perfect predictions should give 1.0"


# =============================================================================
# TESTS FOR calculate_precision_recall_f1
# =============================================================================


class TestCalculatePrecisionRecallF1:
    """Tests for calculate_precision_recall_f1 function."""

    def test_returns_dict(self, binary_data):
        """Test that function returns a dictionary."""
        y_true, y_pred = binary_data
        result = calculate_precision_recall_f1(y_true, y_pred)
        assert result is not None, "Should return a dict"
        assert isinstance(result, dict), "Should be a dictionary"

    def test_has_required_keys(self, binary_data):
        """Test that result has required keys."""
        y_true, y_pred = binary_data
        result = calculate_precision_recall_f1(y_true, y_pred)
        if result is not None:
            assert "precision" in result
            assert "recall" in result
            assert "f1" in result

    def test_values_in_range(self, binary_data):
        """Test that metrics are in [0, 1]."""
        y_true, y_pred = binary_data
        result = calculate_precision_recall_f1(y_true, y_pred)
        if result is not None:
            assert 0 <= result["precision"] <= 1
            assert 0 <= result["recall"] <= 1
            assert 0 <= result["f1"] <= 1


# =============================================================================
# TESTS FOR get_confusion_matrix
# =============================================================================


class TestGetConfusionMatrix:
    """Tests for get_confusion_matrix function."""

    def test_returns_array(self, binary_data):
        """Test that function returns numpy array."""
        y_true, y_pred = binary_data
        result = get_confusion_matrix(y_true, y_pred)
        assert result is not None, "Should return array"
        assert isinstance(result, np.ndarray), "Should be numpy array"

    def test_correct_shape(self, multiclass_data):
        """Test that shape is correct for multiclass."""
        y_true, y_pred = multiclass_data
        result = get_confusion_matrix(y_true, y_pred)
        if result is not None:
            assert result.shape == (3, 3), "Should be 3x3 for 3 classes"


# =============================================================================
# TESTS FOR get_classification_report
# =============================================================================


class TestGetClassificationReport:
    """Tests for get_classification_report function."""

    def test_returns_string(self, binary_data):
        """Test that function returns a string."""
        y_true, y_pred = binary_data
        result = get_classification_report(y_true, y_pred)
        assert result is not None, "Should return string"
        assert isinstance(result, str), "Should be a string"

    def test_contains_metrics(self, binary_data):
        """Test that report contains expected metrics."""
        y_true, y_pred = binary_data
        result = get_classification_report(y_true, y_pred)
        if result is not None:
            assert "precision" in result.lower()
            assert "recall" in result.lower()
            assert "f1" in result.lower()


# =============================================================================
# TESTS FOR cross_validate_model
# =============================================================================


class TestCrossValidateModel:
    """Tests for cross_validate_model function."""

    def test_returns_dict(self, iris_data):
        """Test that function returns a dictionary."""
        X, y = iris_data
        model = LogisticRegression(max_iter=200)

        result = cross_validate_model(model, X, y, cv=3)
        assert result is not None, "Should return dict"
        assert isinstance(result, dict), "Should be dictionary"

    def test_has_required_keys(self, iris_data):
        """Test that result has mean and std."""
        X, y = iris_data
        model = LogisticRegression(max_iter=200)

        result = cross_validate_model(model, X, y, cv=3)
        if result is not None:
            assert "mean" in result
            assert "std" in result


# =============================================================================
# TESTS FOR compare_models
# =============================================================================


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_dataframe(self, iris_data):
        """Test that function returns DataFrame."""
        X, y = iris_data
        models = {
            "LR": LogisticRegression(max_iter=200),
            "RF": RandomForestClassifier(n_estimators=10, random_state=42),
        }

        result = compare_models(models, X, y, cv=3)
        assert result is not None, "Should return DataFrame"
        assert isinstance(result, pd.DataFrame), "Should be DataFrame"

    def test_has_all_models(self, iris_data):
        """Test that all models are in results."""
        X, y = iris_data
        models = {
            "LR": LogisticRegression(max_iter=200),
            "RF": RandomForestClassifier(n_estimators=10, random_state=42),
        }

        result = compare_models(models, X, y, cv=3)
        if result is not None:
            assert len(result) == 2, "Should have 2 models"


# =============================================================================
# TESTS FOR grid_search_cv
# =============================================================================


class TestGridSearchCV:
    """Tests for grid_search_cv function."""

    def test_returns_dict(self, iris_data):
        """Test that function returns a dictionary."""
        X, y = iris_data
        param_grid = {"C": [0.1, 1]}

        result = grid_search_cv(
            LogisticRegression(max_iter=200), param_grid, X, y, cv=3
        )
        assert result is not None, "Should return dict"
        assert isinstance(result, dict)

    def test_has_required_keys(self, iris_data):
        """Test that result has required keys."""
        X, y = iris_data
        param_grid = {"C": [0.1, 1]}

        result = grid_search_cv(
            LogisticRegression(max_iter=200), param_grid, X, y, cv=3
        )
        if result is not None:
            assert "best_params" in result
            assert "best_score" in result
            assert "best_model" in result


# =============================================================================
# TESTS FOR evaluate_model_full
# =============================================================================


class TestEvaluateModelFull:
    """Tests for evaluate_model_full function."""

    def test_returns_dict(self, iris_data):
        """Test that function returns a dictionary."""
        X, y = iris_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        result = evaluate_model_full(model, X_test, y_test)
        assert result is not None, "Should return dict"
        assert isinstance(result, dict)

    def test_has_all_metrics(self, iris_data):
        """Test that result has all metrics."""
        X, y = iris_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        result = evaluate_model_full(model, X_test, y_test)
        if result is not None:
            assert "accuracy" in result
            assert "precision" in result
            assert "recall" in result
            assert "f1" in result

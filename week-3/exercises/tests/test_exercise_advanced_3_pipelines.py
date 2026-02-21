"""
Week 3 Exercise 3: Tests for Advanced ML Pipelines
==================================================

Run tests:
    python -m pytest tests/test_exercise_advanced_3_pipelines.py -v
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_advanced_3_pipelines import (
    create_preprocessing_pipeline,
    create_full_pipeline,
    add_polynomial_features,
    create_feature_selector,
    create_voting_classifier,
    create_stacking_classifier,
    nested_cross_validation,
    create_calibrated_classifier,
    learning_curve_analysis,
    create_threshold_classifier,
)


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def sample_models():
    """Create sample models for testing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    return {
        "lr": LogisticRegression(max_iter=200),
        "rf": RandomForestClassifier(n_estimators=10, random_state=42),
    }


# =============================================================================
# TESTS FOR create_preprocessing_pipeline
# =============================================================================


class TestCreatePreprocessingPipeline:
    """Tests for Task 1."""

    def test_returns_transformer(self):
        """Test that function returns a transformer."""
        result = create_preprocessing_pipeline(
            numeric_features=["age", "income"], categorical_features=["gender"]
        )
        assert result is not None, "Function returned None"
        assert hasattr(result, "fit_transform")


# =============================================================================
# TESTS FOR create_full_pipeline
# =============================================================================


class TestCreateFullPipeline:
    """Tests for Task 2."""

    def test_returns_pipeline(self):
        """Test that function returns a Pipeline."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        result = create_full_pipeline(StandardScaler(), LogisticRegression())
        assert result is not None, "Function returned None"
        assert hasattr(result, "fit")
        assert hasattr(result, "predict")


# =============================================================================
# TESTS FOR add_polynomial_features
# =============================================================================


class TestAddPolynomialFeatures:
    """Tests for Task 3."""

    def test_returns_array(self):
        """Test function returns array."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = add_polynomial_features(X, degree=2)
        assert result is not None, "Function returned None"
        assert isinstance(result, np.ndarray)

    def test_expands_features(self):
        """Test that features are expanded."""
        X = np.array([[1, 2], [3, 4]])
        result = add_polynomial_features(X, degree=2)
        if result is not None:
            # degree 2 with 2 features: 1 + 2 + 3 = 6 (with bias)
            assert result.shape[1] > 2


# =============================================================================
# TESTS FOR create_feature_selector
# =============================================================================


class TestCreateFeatureSelector:
    """Tests for Task 4."""

    def test_returns_selector(self):
        """Test function returns a selector."""
        result = create_feature_selector(k=5)
        assert result is not None, "Function returned None"
        assert hasattr(result, "fit_transform")


# =============================================================================
# TESTS FOR create_voting_classifier
# =============================================================================


class TestCreateVotingClassifier:
    """Tests for Task 5."""

    def test_returns_classifier(self, sample_models):
        """Test function returns VotingClassifier."""
        result = create_voting_classifier(sample_models)
        assert result is not None, "Function returned None"
        assert hasattr(result, "fit")
        assert hasattr(result, "predict")

    def test_can_fit(self, sample_models, sample_data):
        """Test that classifier can be fitted."""
        X, y = sample_data
        result = create_voting_classifier(sample_models)
        if result:
            result.fit(X, y)
            preds = result.predict(X[:5])
            assert len(preds) == 5


# =============================================================================
# TESTS FOR create_stacking_classifier
# =============================================================================


class TestCreateStackingClassifier:
    """Tests for Task 6."""

    def test_returns_classifier(self, sample_models):
        """Test function returns StackingClassifier."""
        from sklearn.linear_model import LogisticRegression

        final = LogisticRegression(max_iter=200)
        result = create_stacking_classifier(sample_models, final)
        assert result is not None, "Function returned None"


# =============================================================================
# TESTS FOR nested_cross_validation
# =============================================================================


class TestNestedCrossValidation:
    """Tests for Task 7."""

    def test_returns_dict(self, sample_data):
        """Test function returns dictionary."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        param_grid = {"C": [0.1, 1]}
        result = nested_cross_validation(
            X, y, LogisticRegression(max_iter=200), param_grid, outer_cv=3, inner_cv=2
        )
        assert result is not None, "Function returned None"
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_data):
        """Test required keys are present."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        param_grid = {"C": [0.1, 1]}
        result = nested_cross_validation(
            X, y, LogisticRegression(max_iter=200), param_grid, outer_cv=3, inner_cv=2
        )
        if result:
            assert "outer_scores" in result
            assert "mean_score" in result
            assert "std_score" in result


# =============================================================================
# TESTS FOR create_calibrated_classifier
# =============================================================================


class TestCreateCalibratedClassifier:
    """Tests for Task 8."""

    def test_returns_classifier(self):
        """Test function returns calibrated classifier."""
        from sklearn.linear_model import LogisticRegression

        result = create_calibrated_classifier(LogisticRegression(max_iter=200))
        assert result is not None, "Function returned None"


# =============================================================================
# TESTS FOR learning_curve_analysis
# =============================================================================


class TestLearningCurveAnalysis:
    """Tests for Task 9."""

    def test_returns_dict(self, sample_data):
        """Test function returns dictionary."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        result = learning_curve_analysis(LogisticRegression(max_iter=200), X, y, cv=3)
        assert result is not None, "Function returned None"
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_data):
        """Test required keys are present."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        result = learning_curve_analysis(LogisticRegression(max_iter=200), X, y, cv=3)
        if result:
            assert "train_sizes" in result
            assert "train_scores" in result
            assert "test_scores" in result


# =============================================================================
# TESTS FOR create_threshold_classifier
# =============================================================================


class TestCreateThresholdClassifier:
    """Tests for Task 10."""

    def test_returns_callable(self, sample_data):
        """Test function returns callable."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        result = create_threshold_classifier(model, threshold=0.7)
        assert result is not None, "Function returned None"
        assert callable(result)

    def test_predictions_work(self, sample_data):
        """Test that threshold predictions work."""
        from sklearn.linear_model import LogisticRegression

        X, y = sample_data
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        predict_fn = create_threshold_classifier(model, threshold=0.7)
        if predict_fn:
            preds = predict_fn(X[:10])
            assert len(preds) == 10

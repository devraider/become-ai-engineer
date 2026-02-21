"""
Week 3 Exercise 1: scikit-learn Basics
======================================

Learn the fundamentals of scikit-learn: data loading, preprocessing,
model training, and prediction.

Run this file:
    python exercise_1_sklearn_basics.py

Run tests:
    python -m pytest tests/test_exercise_1.py -v
"""

import numpy as np
import pandas as pd
from typing import Tuple


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Task 1: Load the Iris dataset from scikit-learn.

    Returns:
        Tuple of (X, y, feature_names)
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)
        - feature_names: List of feature names

    Hints:
        - from sklearn.datasets import load_iris
        - data = load_iris()
        - data.data, data.target, data.feature_names
    """
    # TODO: Implement
    pass


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Task 2: Split data into train and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Hints:
        - from sklearn.model_selection import train_test_split
        - Use stratify=y for balanced splits
    """
    # TODO: Implement
    pass


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 3: Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_scaled, X_test_scaled)

    Important:
        - Fit the scaler on X_train only!
        - Transform both X_train and X_test
        - This prevents data leakage

    Hints:
        - from sklearn.preprocessing import StandardScaler
        - scaler.fit_transform(X_train) for training
        - scaler.transform(X_test) for test (no fit!)
    """
    # TODO: Implement
    pass


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray):
    """
    Task 4: Train a Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained LogisticRegression model

    Hints:
        - from sklearn.linear_model import LogisticRegression
        - Use max_iter=200 to ensure convergence
        - model.fit(X_train, y_train)
    """
    # TODO: Implement
    pass


def make_predictions(model, X: np.ndarray) -> np.ndarray:
    """
    Task 5: Make predictions using a trained model.

    Args:
        model: Trained sklearn model
        X: Features to predict on

    Returns:
        Predicted labels

    Hints:
        - model.predict(X)
    """
    # TODO: Implement
    pass


def get_prediction_probabilities(model, X: np.ndarray) -> np.ndarray:
    """
    Task 6: Get prediction probabilities.

    Args:
        model: Trained sklearn model
        X: Features to predict on

    Returns:
        Probability matrix (n_samples, n_classes)

    Hints:
        - model.predict_proba(X)
    """
    # TODO: Implement
    pass


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
):
    """
    Task 7: Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier

    Hints:
        - from sklearn.ensemble import RandomForestClassifier
    """
    # TODO: Implement
    pass


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Task 8: Get feature importances from a tree-based model.

    Args:
        model: Trained tree-based model (e.g., RandomForest)
        feature_names: List of feature names

    Returns:
        DataFrame with columns ['feature', 'importance']
        sorted by importance descending

    Hints:
        - model.feature_importances_ gives importance array
        - Create DataFrame and sort by importance
    """
    # TODO: Implement
    pass


def create_pipeline(scaler, classifier):
    """
    Task 9: Create a sklearn Pipeline.

    Args:
        scaler: A preprocessing transformer (e.g., StandardScaler)
        classifier: A classifier model

    Returns:
        sklearn Pipeline

    Hints:
        - from sklearn.pipeline import Pipeline
        - Pipeline([('scaler', scaler), ('classifier', classifier)])
    """
    # TODO: Implement
    pass


def save_model(model, filepath: str) -> None:
    """
    Task 10: Save a trained model to file.

    Args:
        model: Trained sklearn model
        filepath: Path to save the model

    Hints:
        - import joblib
        - joblib.dump(model, filepath)
    """
    # TODO: Implement
    pass


def load_model(filepath: str):
    """
    Task 11: Load a saved model from file.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded sklearn model

    Hints:
        - import joblib
        - joblib.load(filepath)
    """
    # TODO: Implement
    pass


# =============================================================================
# QUICK CHECK - Run this file to test your implementations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 1: scikit-learn Basics")
    print("=" * 60)

    # Test 1: Load data
    print("\n--- Task 1: Load Iris Dataset ---")
    result = load_iris_dataset()
    if result:
        X, y, feature_names = result
        print(f"✅ X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Features: {feature_names}")
    else:
        print("❌ Not implemented")
        X, y, feature_names = None, None, None

    # Test 2: Split data
    print("\n--- Task 2: Split Data ---")
    if X is not None:
        splits = split_data(X, y)
        if splits:
            X_train, X_test, y_train, y_test = splits
            print(f"✅ Train: {X_train.shape[0]} samples")
            print(f"   Test: {X_test.shape[0]} samples")
        else:
            print("❌ Not implemented")
            X_train, X_test, y_train, y_test = None, None, None, None

    # Test 3: Scale features
    print("\n--- Task 3: Scale Features ---")
    if X_train is not None:
        scaled = scale_features(X_train, X_test)
        if scaled:
            X_train_scaled, X_test_scaled = scaled
            print(f"✅ Scaled train mean: {X_train_scaled.mean():.4f}")
            print(f"   Scaled train std: {X_train_scaled.std():.4f}")
        else:
            print("❌ Not implemented")
            X_train_scaled = X_train

    # Test 4: Train Logistic Regression
    print("\n--- Task 4: Train Logistic Regression ---")
    if X_train is not None:
        lr_model = train_logistic_regression(X_train_scaled, y_train)
        if lr_model:
            print(f"✅ Model trained: {type(lr_model).__name__}")
        else:
            print("❌ Not implemented")

    # Test 5: Make predictions
    print("\n--- Task 5: Make Predictions ---")
    if lr_model:
        preds = make_predictions(lr_model, X_test_scaled if scaled else X_test)
        if preds is not None:
            print(f"✅ Predictions: {preds[:10]}...")
        else:
            print("❌ Not implemented")

    # Test 6: Get probabilities
    print("\n--- Task 6: Prediction Probabilities ---")
    if lr_model:
        probs = get_prediction_probabilities(
            lr_model, X_test_scaled if scaled else X_test
        )
        if probs is not None:
            print(f"✅ Probabilities shape: {probs.shape}")
            print(f"   First sample: {probs[0]}")
        else:
            print("❌ Not implemented")

    # Test 7: Train Random Forest
    print("\n--- Task 7: Train Random Forest ---")
    if X_train is not None:
        rf_model = train_random_forest(X_train, y_train)
        if rf_model:
            print(f"✅ Model: {type(rf_model).__name__}")
            print(
                f"   Trees: {rf_model.n_estimators if hasattr(rf_model, 'n_estimators') else 'N/A'}"
            )
        else:
            print("❌ Not implemented")

    # Test 8: Feature importance
    print("\n--- Task 8: Feature Importance ---")
    if rf_model and feature_names:
        importance = get_feature_importance(rf_model, feature_names)
        if importance is not None:
            print(f"✅ Top features:")
            print(importance.head().to_string(index=False))
        else:
            print("❌ Not implemented")

    # Test 9: Create Pipeline
    print("\n--- Task 9: Create Pipeline ---")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = create_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    if pipe:
        print(f"✅ Pipeline created with {len(pipe.steps)} steps")
    else:
        print("❌ Not implemented")

    # Test 10 & 11: Save and Load
    print("\n--- Tasks 10-11: Save/Load Model ---")
    import tempfile
    import os

    if lr_model:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            filepath = f.name
        try:
            save_model(lr_model, filepath)
            loaded = load_model(filepath)
            if loaded:
                print(f"✅ Model saved and loaded successfully")
            else:
                print("❌ Load not implemented")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    print("\n" + "=" * 60)
    print("Run tests: python -m pytest tests/test_exercise_1.py -v")
    print("=" * 60)

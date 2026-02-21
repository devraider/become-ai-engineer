"""
Week 3 Exercise 1: scikit-learn Basics - SOLUTIONS
==================================================

These are the reference solutions. Try to complete the exercises yourself first!
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


# =============================================================================
# EXERCISE 1: Load the Iris Dataset
# =============================================================================


def load_iris_dataset():
    """
    Load the classic Iris dataset.

    Returns:
        tuple: (X, y, feature_names) where X is features, y is labels
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    return X, y, feature_names


# =============================================================================
# EXERCISE 2: Split Data into Train/Test Sets
# =============================================================================


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X: Features array
        y: Labels array
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# EXERCISE 3: Scale Features with StandardScaler
# =============================================================================


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        tuple: (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# =============================================================================
# EXERCISE 4: Train a Logistic Regression Model
# =============================================================================


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a logistic regression classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed

    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(max_iter=200, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# EXERCISE 5: Make Predictions
# =============================================================================


def make_predictions(model, X):
    """
    Use a trained model to make predictions.

    Args:
        model: Trained sklearn model
        X: Features to predict on

    Returns:
        np.ndarray: Predicted labels
    """
    predictions = model.predict(X)
    return predictions


# =============================================================================
# EXERCISE 6: Get Prediction Probabilities
# =============================================================================


def get_prediction_probabilities(model, X):
    """
    Get probability estimates for each class.

    Args:
        model: Trained sklearn model with predict_proba method
        X: Features to predict on

    Returns:
        np.ndarray: Probability matrix (n_samples, n_classes)
    """
    probabilities = model.predict_proba(X)
    return probabilities


# =============================================================================
# EXERCISE 7: Train a Random Forest Classifier
# =============================================================================


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees (default 100)
        random_state: Random seed

    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# EXERCISE 8: Get Feature Importance
# =============================================================================


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance'] sorted descending
    """
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


# =============================================================================
# EXERCISE 9: Create a Pipeline
# =============================================================================


def create_pipeline(scaler, classifier):
    """
    Create an sklearn Pipeline combining preprocessing and classification.

    Args:
        scaler: Preprocessing step (e.g., StandardScaler())
        classifier: Classification model

    Returns:
        Pipeline: sklearn Pipeline object
    """
    pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
    return pipeline


# =============================================================================
# EXERCISE 10: Save Model to Disk
# =============================================================================


def save_model(model, filepath):
    """
    Save a trained model to disk using joblib.

    Args:
        model: Trained sklearn model
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)


# =============================================================================
# EXERCISE 11: Load Model from Disk
# =============================================================================


def load_model(filepath):
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded sklearn model
    """
    model = joblib.load(filepath)
    return model


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 1: scikit-learn Basics - SOLUTIONS")
    print("=" * 60)

    # Load data
    X, y, feature_names = load_iris_dataset()
    print(f"\n1. Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"2. Split: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print(f"3. Scaled features - train mean: {X_train_scaled.mean():.4f}")

    # Train logistic regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    print(f"4. Trained LogisticRegression")

    # Make predictions
    predictions = make_predictions(lr_model, X_test_scaled)
    print(f"5. Predictions: {predictions[:5]}")

    # Get probabilities
    probs = get_prediction_probabilities(lr_model, X_test_scaled[:3])
    print(f"6. Probabilities shape: {probs.shape}")

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    print(f"7. Trained RandomForest with {rf_model.n_estimators} trees")

    # Feature importance
    importance = get_feature_importance(rf_model, feature_names)
    print(f"8. Top feature: {importance.iloc[0]['feature']}")

    # Create pipeline
    pipeline = create_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    pipeline.fit(X_train, y_train)
    print(f"9. Pipeline accuracy: {pipeline.score(X_test, y_test):.4f}")

    # Save and load (demo)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        save_model(lr_model, f.name)
        loaded = load_model(f.name)
        print(f"10-11. Model saved and loaded successfully")

    print("\n" + "=" * 60)
    print("All exercises completed!")

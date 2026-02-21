"""
Week 3 Exercise 3: Advanced ML Pipelines & Feature Engineering
==============================================================

Master scikit-learn pipelines, custom transformers, and advanced ML patterns
used in production systems.

Run this file:
    python exercise_advanced_3_pipelines.py

Run tests:
    python -m pytest tests/test_exercise_advanced_3_pipelines.py -v
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def create_preprocessing_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Any:
    """
    Task 1: Create a ColumnTransformer for mixed data types.
    
    Real-world use: Handling datasets with mixed numeric and categorical features,
    which is common in tabular ML tasks.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    
    Returns:
        ColumnTransformer that:
        - Scales numeric features with StandardScaler
        - One-hot encodes categorical features
    
    Hints:
        - from sklearn.compose import ColumnTransformer
        - from sklearn.preprocessing import StandardScaler, OneHotEncoder
        - Use Pipeline for each branch if needed
    """
    # TODO: Implement
    pass


def create_full_pipeline(preprocessor: Any, model: Any) -> Any:
    """
    Task 2: Create end-to-end pipeline with preprocessing and model.
    
    Args:
        preprocessor: ColumnTransformer or preprocessing pipeline
        model: sklearn estimator
    
    Returns:
        Pipeline combining preprocessing and model
    """
    # TODO: Implement
    pass


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Task 3: Add polynomial features for capturing non-linear relationships.
    
    Real-world use: Feature engineering for linear models to capture
    non-linear patterns.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        degree: Polynomial degree
    
    Returns:
        Expanded feature matrix with polynomial terms
    
    Hints:
        - from sklearn.preprocessing import PolynomialFeatures
    """
    # TODO: Implement
    pass


def create_feature_selector(k: int = 10) -> Any:
    """
    Task 4: Create a feature selector based on mutual information.
    
    Real-world use: Reducing dimensionality, removing irrelevant features,
    improving model interpretability.
    
    Args:
        k: Number of top features to select
    
    Returns:
        SelectKBest transformer with mutual_info_classif scoring
    
    Hints:
        - from sklearn.feature_selection import SelectKBest, mutual_info_classif
    """
    # TODO: Implement
    pass


def create_voting_classifier(models: Dict[str, Any]) -> Any:
    """
    Task 5: Create an ensemble voting classifier.
    
    Real-world use: Combining multiple models for better predictions,
    reducing variance, leveraging model diversity.
    
    Args:
        models: Dictionary of {'name': model} pairs
    
    Returns:
        VotingClassifier with soft voting
    
    Hints:
        - from sklearn.ensemble import VotingClassifier
        - estimators=[(name, model) for name, model in models.items()]
    """
    # TODO: Implement
    pass


def create_stacking_classifier(
    base_models: Dict[str, Any], final_model: Any
) -> Any:
    """
    Task 6: Create a stacking classifier.
    
    Real-world use: Advanced ensemble technique where base models' predictions
    become features for a meta-learner.
    
    Args:
        base_models: Dictionary of base estimators
        final_model: Meta-learner estimator
    
    Returns:
        StackingClassifier
    
    Hints:
        - from sklearn.ensemble import StackingClassifier
    """
    # TODO: Implement
    pass


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    param_grid: Dict[str, List],
    outer_cv: int = 5,
    inner_cv: int = 3,
) -> Dict[str, Any]:
    """
    Task 7: Perform nested cross-validation for unbiased evaluation.
    
    Real-world use: Getting honest performance estimates while tuning
    hyperparameters. Prevents overfitting to validation set.
    
    Args:
        X: Features
        y: Labels
        model: sklearn estimator
        param_grid: Parameters to search
        outer_cv: Outer fold count
        inner_cv: Inner fold count
    
    Returns:
        Dictionary with:
        - 'outer_scores': scores from outer CV
        - 'mean_score': mean of outer scores
        - 'std_score': std of outer scores
    
    Hints:
        - Outer loop: cross_val_score
        - Inner loop: GridSearchCV as the estimator
    """
    # TODO: Implement
    pass


def create_calibrated_classifier(model: Any, method: str = "sigmoid") -> Any:
    """
    Task 8: Create a probability-calibrated classifier.
    
    Real-world use: Getting reliable probability estimates for
    risk assessment, decision thresholds, uncertainty quantification.
    
    Args:
        model: Base classifier
        method: 'sigmoid' (Platt scaling) or 'isotonic'
    
    Returns:
        CalibratedClassifierCV
    
    Hints:
        - from sklearn.calibration import CalibratedClassifierCV
    """
    # TODO: Implement
    pass


def learning_curve_analysis(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Dict[str, np.ndarray]:
    """
    Task 9: Generate learning curve data for model diagnostics.
    
    Real-world use: Diagnosing bias/variance issues, determining if
    more data would help, detecting overfitting.
    
    Args:
        model: sklearn estimator
        X: Features
        y: Labels
        cv: Cross-validation folds
    
    Returns:
        Dictionary with:
        - 'train_sizes': array of training set sizes
        - 'train_scores': training scores at each size
        - 'test_scores': test scores at each size
    
    Hints:
        - from sklearn.model_selection import learning_curve
    """
    # TODO: Implement
    pass


def create_threshold_classifier(
    model: Any, threshold: float = 0.5
) -> callable:
    """
    Task 10: Create a classifier with custom probability threshold.
    
    Real-world use: Adjusting precision/recall tradeoff, handling
    class imbalance, setting decision boundaries for business rules.
    
    Args:
        model: Trained classifier with predict_proba
        threshold: Probability threshold for positive class
    
    Returns:
        Function that takes X and returns predictions with custom threshold
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 3: Advanced ML Pipelines")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    print("\n1. Testing polynomial features...")
    X_poly = add_polynomial_features(X[:5, :3], degree=2)
    if X_poly is not None:
        print(f"   Original: 3 features -> Poly: {X_poly.shape[1]} features")
    
    print("\n2. Testing voting classifier...")
    models = {
        'lr': LogisticRegression(max_iter=200),
        'rf': RandomForestClassifier(n_estimators=10)
    }
    voting = create_voting_classifier(models)
    if voting:
        print("   VotingClassifier created")
    
    print("\n3. Testing nested CV...")
    param_grid = {'C': [0.1, 1]}
    result = nested_cross_validation(
        X, y, LogisticRegression(max_iter=200), param_grid, outer_cv=3, inner_cv=2
    )
    if result:
        print(f"   Mean nested CV score: {result['mean_score']:.4f}")
    
    print("\nComplete all TODOs and run tests to verify!")

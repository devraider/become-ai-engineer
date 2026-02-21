"""
Week 3 Exercise 2: Model Evaluation
===================================

Master model evaluation metrics, cross-validation, and hyperparameter tuning.

Run this file:
    python exercise_2_model_evaluation.py

Run tests:
    python -m pytest tests/test_exercise_2.py -v
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Task 1: Calculate accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score (0.0 to 1.0)

    Hints:
        - from sklearn.metrics import accuracy_score
        - Or: (y_true == y_pred).mean()
    """
    # TODO: Implement
    pass


def calculate_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """
    Task 2: Calculate precision, recall, and F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Dictionary with 'precision', 'recall', 'f1' keys

    Hints:
        - from sklearn.metrics import precision_score, recall_score, f1_score
    """
    # TODO: Implement
    pass


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Task 3: Get confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as 2D numpy array

    Hints:
        - from sklearn.metrics import confusion_matrix
    """
    # TODO: Implement
    pass


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list = None
) -> str:
    """
    Task 4: Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names

    Returns:
        Classification report as string

    Hints:
        - from sklearn.metrics import classification_report
    """
    # TODO: Implement
    pass


def cross_validate_model(
    model, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy"
) -> Dict[str, float]:
    """
    Task 5: Perform cross-validation on a model.

    Args:
        model: sklearn estimator (not fitted)
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        scoring: Metric to use

    Returns:
        Dictionary with 'mean', 'std', 'scores' keys

    Hints:
        - from sklearn.model_selection import cross_val_score
        - scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    """
    # TODO: Implement
    pass


def compare_models(
    models: dict, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> pd.DataFrame:
    """
    Task 6: Compare multiple models using cross-validation.

    Args:
        models: Dictionary of {name: model} pairs
        X: Feature matrix
        y: Target vector
        cv: Number of folds

    Returns:
        DataFrame with columns ['model', 'mean_score', 'std_score']
        sorted by mean_score descending

    Example:
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier()
        }
    """
    # TODO: Implement
    pass


def grid_search_cv(
    model,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """
    Task 7: Perform grid search with cross-validation.

    Args:
        model: sklearn estimator
        param_grid: Dictionary of parameters to search
        X: Feature matrix
        y: Target vector
        cv: Number of folds
        scoring: Metric to optimize

    Returns:
        Dictionary with:
        - 'best_params': Best parameter combination
        - 'best_score': Best cross-validation score
        - 'best_model': Fitted model with best params

    Hints:
        - from sklearn.model_selection import GridSearchCV
        - grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        - grid.fit(X, y)
        - grid.best_params_, grid.best_score_, grid.best_estimator_
    """
    # TODO: Implement
    pass


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    save_path: str = None,
) -> None:
    """
    Task 8: Plot confusion matrix as heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure

    Hints:
        - import matplotlib.pyplot as plt
        - import seaborn as sns
        - cm = confusion_matrix(y_true, y_pred)
        - sns.heatmap(cm, annot=True, fmt='d')
    """
    # TODO: Implement
    pass


def plot_roc_curve(
    model, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None
) -> float:
    """
    Task 9: Plot ROC curve and return AUC score.

    Note: This is for binary classification only.

    Args:
        model: Trained classifier with predict_proba method
        X_test: Test features
        y_test: Test labels (binary: 0 or 1)
        save_path: Optional path to save figure

    Returns:
        AUC score

    Hints:
        - from sklearn.metrics import roc_curve, auc
        - probas = model.predict_proba(X_test)[:, 1]
        - fpr, tpr, _ = roc_curve(y_test, probas)
        - auc_score = auc(fpr, tpr)
    """
    # TODO: Implement
    pass


def evaluate_model_full(
    model, X_test: np.ndarray, y_test: np.ndarray, class_names: list = None
) -> dict:
    """
    Task 10: Complete model evaluation with all metrics.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        class_names: Optional class names

    Returns:
        Dictionary with:
        - 'accuracy': Accuracy score
        - 'precision': Weighted precision
        - 'recall': Weighted recall
        - 'f1': Weighted F1
        - 'confusion_matrix': Confusion matrix
        - 'report': Classification report string
    """
    # TODO: Implement
    pass


# =============================================================================
# QUICK CHECK - Run this file to test your implementations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 2: Model Evaluation")
    print("=" * 60)

    # Setup: Load data and train a model
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Test 1: Accuracy
    print("\n--- Task 1: Accuracy ---")
    acc = calculate_accuracy(y_test, y_pred)
    if acc is not None:
        print(f"✅ Accuracy: {acc:.4f}")
    else:
        print("❌ Not implemented")

    # Test 2: Precision, Recall, F1
    print("\n--- Task 2: Precision/Recall/F1 ---")
    metrics = calculate_precision_recall_f1(y_test, y_pred)
    if metrics:
        print(f"✅ Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
    else:
        print("❌ Not implemented")

    # Test 3: Confusion Matrix
    print("\n--- Task 3: Confusion Matrix ---")
    cm = get_confusion_matrix(y_test, y_pred)
    if cm is not None:
        print(f"✅ Confusion Matrix:\n{cm}")
    else:
        print("❌ Not implemented")

    # Test 4: Classification Report
    print("\n--- Task 4: Classification Report ---")
    report = get_classification_report(y_test, y_pred, data.target_names.tolist())
    if report:
        print(f"✅ Report:\n{report}")
    else:
        print("❌ Not implemented")

    # Test 5: Cross-validation
    print("\n--- Task 5: Cross-validation ---")
    cv_results = cross_validate_model(LogisticRegression(max_iter=200), X, y, cv=5)
    if cv_results:
        print(f"✅ Mean: {cv_results['mean']:.4f}")
        print(f"   Std: {cv_results['std']:.4f}")
    else:
        print("❌ Not implemented")

    # Test 6: Compare models
    print("\n--- Task 6: Compare Models ---")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    }
    comparison = compare_models(models, X, y)
    if comparison is not None:
        print(f"✅ Model Comparison:")
        print(comparison.to_string(index=False))
    else:
        print("❌ Not implemented")

    # Test 7: Grid Search
    print("\n--- Task 7: Grid Search CV ---")
    param_grid = {"C": [0.1, 1, 10], "max_iter": [100, 200]}
    gs_result = grid_search_cv(LogisticRegression(), param_grid, X, y)
    if gs_result:
        print(f"✅ Best params: {gs_result['best_params']}")
        print(f"   Best score: {gs_result['best_score']:.4f}")
    else:
        print("❌ Not implemented")

    # Test 8: Plot Confusion Matrix
    print("\n--- Task 8: Plot Confusion Matrix ---")
    try:
        plot_confusion_matrix(y_test, y_pred, data.target_names.tolist())
        print("✅ Confusion matrix plotted (check window)")
    except Exception as e:
        print(f"❌ Error or not implemented: {e}")

    # Test 9: ROC Curve (binary only)
    print("\n--- Task 9: ROC Curve ---")
    # Create binary problem
    y_binary = (y == 0).astype(int)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    model_binary = LogisticRegression(max_iter=200)
    model_binary.fit(X_train_b, y_train_b)

    auc_score = plot_roc_curve(model_binary, X_test_b, y_test_b)
    if auc_score is not None:
        print(f"✅ AUC Score: {auc_score:.4f}")
    else:
        print("❌ Not implemented")

    # Test 10: Full Evaluation
    print("\n--- Task 10: Full Model Evaluation ---")
    full_eval = evaluate_model_full(model, X_test, y_test)
    if full_eval:
        print(f"✅ Full Evaluation:")
        print(f"   Accuracy: {full_eval['accuracy']:.4f}")
        print(f"   F1: {full_eval['f1']:.4f}")
    else:
        print("❌ Not implemented")

    print("\n" + "=" * 60)
    print("Run tests: python -m pytest tests/test_exercise_2.py -v")
    print("=" * 60)

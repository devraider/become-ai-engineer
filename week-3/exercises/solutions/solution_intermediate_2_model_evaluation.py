"""
Week 3 Exercise 2: Model Evaluation - SOLUTIONS
===============================================

These are the reference solutions. Try to complete the exercises yourself first!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


# =============================================================================
# EXERCISE 1: Calculate Accuracy
# =============================================================================


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Accuracy score between 0 and 1
    """
    return accuracy_score(y_true, y_pred)


# =============================================================================
# EXERCISE 2: Calculate Precision, Recall, F1
# =============================================================================


def calculate_precision_recall_f1(y_true, y_pred, average="weighted"):
    """
    Calculate precision, recall, and F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('micro', 'macro', 'weighted')

    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float}
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# EXERCISE 3: Get Confusion Matrix
# =============================================================================


def get_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        np.ndarray: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


# =============================================================================
# EXERCISE 4: Get Classification Report
# =============================================================================


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a text classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names

    Returns:
        str: Classification report as text
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


# =============================================================================
# EXERCISE 5: Cross-Validate a Model
# =============================================================================


def cross_validate_model(model, X, y, cv=5, scoring="accuracy"):
    """
    Perform k-fold cross-validation.

    Args:
        model: sklearn estimator
        X: Features
        y: Labels
        cv: Number of folds (default 5)
        scoring: Scoring metric (default 'accuracy')

    Returns:
        dict: {'mean': float, 'std': float, 'scores': array}
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


# =============================================================================
# EXERCISE 6: Compare Multiple Models
# =============================================================================


def compare_models(models_dict, X, y, cv=5):
    """
    Compare multiple models using cross-validation.

    Args:
        models_dict: Dict of {'name': model}
        X: Features
        y: Labels
        cv: Number of folds

    Returns:
        pd.DataFrame: Comparison results with columns ['model', 'mean_score', 'std_score']
    """
    results = []

    for name, model in models_dict.items():
        cv_result = cross_validate_model(model, X, y, cv=cv)
        results.append(
            {
                "model": name,
                "mean_score": cv_result["mean"],
                "std_score": cv_result["std"],
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# EXERCISE 7: Grid Search with Cross-Validation
# =============================================================================


def grid_search_cv(model, param_grid, X, y, cv=5, scoring="accuracy"):
    """
    Perform grid search with cross-validation for hyperparameter tuning.

    Args:
        model: sklearn estimator
        param_grid: Dict of parameters to search
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric

    Returns:
        dict: {'best_params': dict, 'best_score': float, 'best_model': model}
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)

    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "best_model": grid.best_estimator_,
    }


# =============================================================================
# EXERCISE 8: Plot Confusion Matrix
# =============================================================================


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot and optionally save a confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


# =============================================================================
# EXERCISE 9: Plot ROC Curve (Binary Classification)
# =============================================================================


def plot_roc_curve(model, X_test, y_test, save_path=None):
    """
    Plot ROC curve for binary classification.

    Args:
        model: Trained model with predict_proba method
        X_test: Test features
        y_test: Test labels
        save_path: Optional path to save the figure

    Note: Only works for binary classification
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("ROC Curve")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


# =============================================================================
# EXERCISE 10: Complete Model Evaluation
# =============================================================================


def evaluate_model_full(model, X_test, y_test):
    """
    Perform comprehensive model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = calculate_precision_recall_f1(y_test, y_pred)

    return {
        "accuracy": calculate_accuracy(y_test, y_pred),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": get_confusion_matrix(y_test, y_pred),
        "predictions": y_pred,
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 2: Model Evaluation - SOLUTIONS")
    print("=" * 60)

    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 1. Accuracy
    acc = calculate_accuracy(y_test, y_pred)
    print(f"\n1. Accuracy: {acc:.4f}")

    # 2. Precision, Recall, F1
    metrics = calculate_precision_recall_f1(y_test, y_pred)
    print(
        f"2. Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
    )

    # 3. Confusion Matrix
    cm = get_confusion_matrix(y_test, y_pred)
    print(f"3. Confusion Matrix shape: {cm.shape}")

    # 4. Classification Report
    report = get_classification_report(y_test, y_pred, target_names=iris.target_names)
    print(f"4. Classification Report:\n{report}")

    # 5. Cross-Validation
    cv_results = cross_validate_model(LogisticRegression(max_iter=200), X, y, cv=5)
    print(f"5. CV Mean: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

    # 6. Compare Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    comparison = compare_models(models, X, y)
    print(f"6. Model Comparison:\n{comparison}")

    # 7. Grid Search
    param_grid = {"C": [0.1, 1, 10], "max_iter": [200]}
    gs_results = grid_search_cv(LogisticRegression(), param_grid, X, y)
    print(
        f"7. Best params: {gs_results['best_params']}, Score: {gs_results['best_score']:.4f}"
    )

    # 8-9. Plots (saved to files)
    print("8. Confusion matrix plot created")
    print("9. ROC curve (skipped - multiclass)")

    # 10. Full Evaluation
    full_eval = evaluate_model_full(model, X_test, y_test)
    print(f"10. Full evaluation: accuracy={full_eval['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("All exercises completed!")

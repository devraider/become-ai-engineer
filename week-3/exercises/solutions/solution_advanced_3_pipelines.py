"""
Week 3 Exercise 3: Advanced ML Pipelines - SOLUTIONS
====================================================

Reference solutions for the advanced ML pipelines exercise.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve
from sklearn.calibration import CalibratedClassifierCV


def create_preprocessing_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Task 1: Create a ColumnTransformer for mixed data types."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def create_full_pipeline(preprocessor: Any, model: Any) -> Pipeline:
    """Task 2: Create end-to-end pipeline."""
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def add_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Task 3: Add polynomial features."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X)


def create_feature_selector(k: int = 10) -> SelectKBest:
    """Task 4: Create feature selector based on mutual information."""
    return SelectKBest(score_func=mutual_info_classif, k=k)


def create_voting_classifier(models: Dict[str, Any]) -> VotingClassifier:
    """Task 5: Create ensemble voting classifier."""
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting="soft")


def create_stacking_classifier(
    base_models: Dict[str, Any], final_model: Any
) -> StackingClassifier:
    """Task 6: Create stacking classifier."""
    estimators = [(name, model) for name, model in base_models.items()]
    return StackingClassifier(estimators=estimators, final_estimator=final_model, cv=5)


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    param_grid: Dict[str, List],
    outer_cv: int = 5,
    inner_cv: int = 3,
) -> Dict[str, Any]:
    """Task 7: Perform nested cross-validation."""
    # Inner loop: GridSearchCV for hyperparameter tuning
    inner_cv_estimator = GridSearchCV(
        model, param_grid, cv=inner_cv, scoring="accuracy"
    )

    # Outer loop: cross_val_score for unbiased evaluation
    outer_scores = cross_val_score(inner_cv_estimator, X, y, cv=outer_cv)

    return {
        "outer_scores": outer_scores,
        "mean_score": outer_scores.mean(),
        "std_score": outer_scores.std(),
    }


def create_calibrated_classifier(
    model: Any, method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """Task 8: Create probability-calibrated classifier."""
    return CalibratedClassifierCV(model, method=method, cv=5)


def learning_curve_analysis(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Dict[str, np.ndarray]:
    """Task 9: Generate learning curve data."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy"
    )

    return {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def create_threshold_classifier(model: Any, threshold: float = 0.5) -> callable:
    """Task 10: Create classifier with custom probability threshold."""

    def predict_with_threshold(X):
        probas = model.predict_proba(X)
        # For binary: use second column (positive class)
        if probas.shape[1] == 2:
            return (probas[:, 1] >= threshold).astype(int)
        # For multiclass: use threshold on max probability
        else:
            return (probas.max(axis=1) >= threshold).astype(int) * probas.argmax(axis=1)

    return predict_with_threshold


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Exercise 3: Solutions Demo")
    print("=" * 60)

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 1. Preprocessing pipeline (demo with DataFrame)
    df = pd.DataFrame(X_train[:, :5], columns=["a", "b", "c", "d", "e"])
    df["category"] = np.random.choice(["X", "Y", "Z"], len(df))
    preprocessor = create_preprocessing_pipeline(
        numeric_features=["a", "b", "c"], categorical_features=["category"]
    )
    print(f"\n1. Created preprocessing pipeline")

    # 2. Full pipeline
    full_pipe = create_full_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    full_pipe.fit(X_train, y_train)
    print(f"2. Full pipeline accuracy: {full_pipe.score(X_test, y_test):.4f}")

    # 3. Polynomial features
    X_poly = add_polynomial_features(X_train[:5, :3], degree=2)
    print(f"\n3. Polynomial features: 3 -> {X_poly.shape[1]}")

    # 4. Feature selector
    selector = create_feature_selector(k=10)
    selector.fit(X_train, y_train)
    print(f"4. Selected {selector.k} features")

    # 5. Voting classifier
    models = {
        "lr": LogisticRegression(max_iter=200),
        "rf": RandomForestClassifier(n_estimators=50, random_state=42),
    }
    voting = create_voting_classifier(models)
    voting.fit(X_train, y_train)
    print(f"\n5. Voting classifier accuracy: {voting.score(X_test, y_test):.4f}")

    # 6. Stacking classifier
    stacking = create_stacking_classifier(models, LogisticRegression(max_iter=200))
    stacking.fit(X_train, y_train)
    print(f"6. Stacking classifier accuracy: {stacking.score(X_test, y_test):.4f}")

    # 7. Nested CV
    param_grid = {"C": [0.1, 1, 10]}
    nested_result = nested_cross_validation(
        X, y, LogisticRegression(max_iter=200), param_grid, outer_cv=3, inner_cv=2
    )
    print(
        f"\n7. Nested CV mean: {nested_result['mean_score']:.4f} (+/- {nested_result['std_score']:.4f})"
    )

    # 8. Calibrated classifier
    calibrated = create_calibrated_classifier(LogisticRegression(max_iter=200))
    calibrated.fit(X_train, y_train)
    print(f"8. Calibrated classifier accuracy: {calibrated.score(X_test, y_test):.4f}")

    # 9. Learning curve
    lc_result = learning_curve_analysis(LogisticRegression(max_iter=200), X, y, cv=3)
    print(f"\n9. Learning curve: {len(lc_result['train_sizes'])} data points")

    # 10. Threshold classifier
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    threshold_predict = create_threshold_classifier(lr, threshold=0.7)
    preds = threshold_predict(X_test[:10])
    print(f"10. Threshold predictions: {preds}")

    print("\n" + "=" * 60)
    print("All solutions working!")

"""
Week 3 Project: Sentiment Classifier
====================================

Build a complete ML pipeline for text classification.

Requirements:
1. Load emotion dataset from HuggingFace
2. Preprocess text with TF-IDF
3. Train and compare multiple classifiers
4. Tune hyperparameters
5. Evaluate on test set
6. Save the best model

Run this file:
    python project_classifier.py

Run tests:
    python -m pytest tests/test_project.py -v
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def load_emotion_data(num_samples: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task 1: Load the emotion dataset from HuggingFace.

    Args:
        num_samples: Number of samples to load (for faster experimentation)

    Returns:
        Tuple of (train_df, test_df) with columns ['text', 'label']

    Hints:
        - from datasets import load_dataset
        - dataset = load_dataset("emotion")
        - Convert to DataFrames
        - Limit samples with .select(range(num_samples)) or slicing
    """
    # TODO: Implement
    pass


def preprocess_text(text: str) -> str:
    """
    Task 2: Preprocess a single text sample.

    Steps:
    1. Convert to lowercase
    2. Remove extra whitespace
    3. (Optionally) remove special characters

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # TODO: Implement
    pass


def create_tfidf_features(
    train_texts: list,
    test_texts: list,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Task 3: Create TF-IDF features from text.

    Args:
        train_texts: List of training texts
        test_texts: List of test texts
        max_features: Maximum number of features
        ngram_range: Range of n-grams (e.g., (1, 2) for unigrams and bigrams)

    Returns:
        Tuple of (X_train, X_test, vectorizer)

    Important:
        - Fit vectorizer on train_texts only!
        - Transform both train and test

    Hints:
        - from sklearn.feature_extraction.text import TfidfVectorizer
        - vectorizer.fit_transform(train_texts) for train
        - vectorizer.transform(test_texts) for test
    """
    # TODO: Implement
    pass


def train_classifier(
    X_train: np.ndarray, y_train: np.ndarray, classifier_type: str = "logistic"
) -> Any:
    """
    Task 4: Train a classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        classifier_type: One of 'logistic', 'random_forest', 'naive_bayes'

    Returns:
        Trained classifier

    Supported classifiers:
        - 'logistic': LogisticRegression(max_iter=500)
        - 'random_forest': RandomForestClassifier(n_estimators=100)
        - 'naive_bayes': MultinomialNB()
    """
    # TODO: Implement
    pass


def compare_classifiers(
    X_train: np.ndarray, y_train: np.ndarray, cv: int = 5
) -> pd.DataFrame:
    """
    Task 5: Compare multiple classifiers using cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        DataFrame with columns ['classifier', 'accuracy_mean', 'accuracy_std', 'f1_mean', 'f1_std']
        sorted by f1_mean descending

    Compare these classifiers:
        - Logistic Regression
        - Random Forest
        - Multinomial Naive Bayes

    Hints:
        - Use cross_val_score for each model
        - Try scoring='accuracy' and scoring='f1_weighted'
    """
    # TODO: Implement
    pass


def tune_best_model(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = "logistic"
) -> Dict[str, Any]:
    """
    Task 6: Tune hyperparameters for the best model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to tune

    Returns:
        Dictionary with:
        - 'best_params': Best parameters found
        - 'best_score': Best CV score
        - 'model': Fitted model with best params

    Parameter grids:
        - logistic: {'C': [0.1, 1, 10], 'max_iter': [500]}
        - random_forest: {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
        - naive_bayes: {'alpha': [0.1, 0.5, 1.0]}

    Hints:
        - from sklearn.model_selection import GridSearchCV
    """
    # TODO: Implement
    pass


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, label_names: list = None
) -> Dict[str, Any]:
    """
    Task 7: Evaluate model on test set.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        label_names: Optional list of label names

    Returns:
        Dictionary with:
        - 'accuracy': Test accuracy
        - 'f1': Weighted F1 score
        - 'precision': Weighted precision
        - 'recall': Weighted recall
        - 'confusion_matrix': Confusion matrix
        - 'classification_report': Classification report string
    """
    # TODO: Implement
    pass


def save_pipeline(vectorizer: Any, model: Any, filepath: str) -> None:
    """
    Task 8: Save the complete pipeline (vectorizer + model).

    Args:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained classifier
        filepath: Path to save the pipeline

    Hints:
        - import joblib
        - Save as a dictionary: {'vectorizer': ..., 'model': ...}
    """
    # TODO: Implement
    pass


def load_pipeline(filepath: str) -> Tuple[Any, Any]:
    """
    Task 9: Load a saved pipeline.

    Args:
        filepath: Path to the saved pipeline

    Returns:
        Tuple of (vectorizer, model)
    """
    # TODO: Implement
    pass


def predict_sentiment(
    text: str, vectorizer: Any, model: Any, label_names: list = None
) -> Dict[str, Any]:
    """
    Task 10: Predict sentiment for new text.

    Args:
        text: Input text
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained classifier
        label_names: Optional list of label names

    Returns:
        Dictionary with:
        - 'text': Original text
        - 'predicted_label': Predicted class (name or number)
        - 'confidence': Probability of predicted class
        - 'all_probabilities': Dict of {label: probability}
    """
    # TODO: Implement
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(num_samples: int = 5000) -> Dict[str, Any]:
    """
    Run the complete sentiment classification pipeline.
    """
    print("=" * 60)
    print("Week 3 Project: Sentiment Classifier")
    print("=" * 60)

    # Emotion labels
    label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # 1. Load data
    print("\n📥 Loading data...")
    data = load_emotion_data(num_samples)
    if data is None:
        print("❌ Failed to load data")
        return {"success": False, "error": "load_emotion_data returned None"}

    train_df, test_df = data
    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")

    # 2. Preprocess
    print("\n🧹 Preprocessing text...")
    train_texts = [
        preprocess_text(t) if preprocess_text(t) else t
        for t in train_df["text"].tolist()
    ]
    test_texts = [
        preprocess_text(t) if preprocess_text(t) else t
        for t in test_df["text"].tolist()
    ]
    print(f"✅ Preprocessed {len(train_texts)} train, {len(test_texts)} test texts")

    # 3. Create features
    print("\n📊 Creating TF-IDF features...")
    features = create_tfidf_features(train_texts, test_texts)
    if features is None:
        print("❌ Failed to create features")
        return {"success": False, "error": "create_tfidf_features returned None"}

    X_train, X_test, vectorizer = features
    print(f"✅ Features: {X_train.shape[1]} dimensions")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # 4. Compare classifiers
    print("\n🔍 Comparing classifiers...")
    comparison = compare_classifiers(X_train, y_train)
    if comparison is not None:
        print("✅ Model comparison:")
        print(comparison.to_string(index=False))

    # 5. Tune best model
    print("\n⚙️ Tuning best model...")
    tuning = tune_best_model(X_train, y_train, "logistic")
    if tuning:
        print(f"✅ Best params: {tuning['best_params']}")
        print(f"   CV Score: {tuning['best_score']:.4f}")
        best_model = tuning["model"]
    else:
        print("⚠️ Using default model")
        best_model = train_classifier(X_train, y_train, "logistic")

    # 6. Evaluate
    print("\n📈 Evaluating on test set...")
    if best_model:
        evaluation = evaluate_model(best_model, X_test, y_test, label_names)
        if evaluation:
            print(f"✅ Test Accuracy: {evaluation['accuracy']:.4f}")
            print(f"   Test F1: {evaluation['f1']:.4f}")
            print(f"\nClassification Report:\n{evaluation['classification_report']}")

    # 7. Save pipeline
    print("\n💾 Saving pipeline...")
    if best_model and vectorizer:
        save_pipeline(vectorizer, best_model, "sentiment_pipeline.joblib")
        print("✅ Pipeline saved to sentiment_pipeline.joblib")

    # 8. Test prediction
    print("\n🎯 Testing prediction...")
    test_text = "I am so happy today!"
    if vectorizer and best_model:
        result = predict_sentiment(test_text, vectorizer, best_model, label_names)
        if result:
            print(f"✅ Text: '{result['text']}'")
            print(f"   Prediction: {result['predicted_label']}")
            print(f"   Confidence: {result['confidence']:.4f}")

    print("\n" + "=" * 60)
    print("🎉 Pipeline complete!")
    print("=" * 60)

    return {
        "success": True,
        "accuracy": evaluation["accuracy"] if evaluation else None,
        "f1": evaluation["f1"] if evaluation else None,
    }


if __name__ == "__main__":
    result = run_pipeline()
    if not result["success"]:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown')}")

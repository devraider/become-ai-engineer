"""
Week 3 Project: Sentiment Classifier Pipeline - SOLUTION
========================================================

Complete sentiment classification pipeline using the Emotion dataset.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import joblib


# =============================================================================
# STEP 1: Load the Emotion Dataset
# =============================================================================


def load_emotion_data(num_samples=None):
    """
    Load the Emotion dataset from HuggingFace.

    Args:
        num_samples: Optional limit on samples to load

    Returns:
        tuple: (train_df, test_df) with columns ['text', 'label']
    """
    from datasets import load_dataset

    dataset = load_dataset("dair-ai/emotion")

    train_data = dataset["train"]
    test_data = dataset["test"]

    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
        test_data = test_data.select(range(min(num_samples // 5, len(test_data))))

    train_df = pd.DataFrame({"text": train_data["text"], "label": train_data["label"]})

    test_df = pd.DataFrame({"text": test_data["text"], "label": test_data["label"]})

    return train_df, test_df


# =============================================================================
# STEP 2: Preprocess Text Data
# =============================================================================


def preprocess_text(text):
    """
    Clean and preprocess a single text string.

    Args:
        text: Raw text string

    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


# =============================================================================
# STEP 3: Create TF-IDF Features
# =============================================================================


def create_tfidf_features(train_texts, test_texts, max_features=5000):
    """
    Create TF-IDF feature vectors from text.

    Args:
        train_texts: List of training texts
        test_texts: List of test texts
        max_features: Maximum number of features

    Returns:
        tuple: (X_train, X_test, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), stop_words="english", min_df=2
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


# =============================================================================
# STEP 4: Train Different Classifiers
# =============================================================================


def train_classifier(X_train, y_train, classifier_type="logistic"):
    """
    Train a classifier on the features.

    Args:
        X_train: Training features
        y_train: Training labels
        classifier_type: 'logistic', 'random_forest', or 'naive_bayes'

    Returns:
        Trained classifier
    """
    if classifier_type == "logistic":
        model = LogisticRegression(max_iter=500, random_state=42)
    elif classifier_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == "naive_bayes":
        model = MultinomialNB()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    model.fit(X_train, y_train)
    return model


# =============================================================================
# STEP 5: Compare Classifiers
# =============================================================================


def compare_classifiers(X_train, y_train, cv=5):
    """
    Compare multiple classifiers using cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        pd.DataFrame: Comparison results
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB(),
    }

    results = []
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted")
        results.append(
            {"classifier": name, "mean_f1": scores.mean(), "std_f1": scores.std()}
        )

    df = pd.DataFrame(results)
    df = df.sort_values("mean_f1", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# STEP 6: Tune Best Model
# =============================================================================


def tune_best_model(X_train, y_train, cv=3):
    """
    Perform hyperparameter tuning on the best model.

    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds

    Returns:
        dict: {'best_model': model, 'best_params': dict, 'best_score': float}
    """
    param_grid = {"C": [0.1, 1, 10], "max_iter": [500]}

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    return {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


# =============================================================================
# STEP 7: Evaluate Model
# =============================================================================


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "predictions": y_pred,
    }


# =============================================================================
# STEP 8: Save Pipeline
# =============================================================================


def save_pipeline(vectorizer, model, filepath):
    """
    Save the complete pipeline (vectorizer + model) to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer
        model: Trained classifier
        filepath: Path to save the pipeline
    """
    pipeline = {"vectorizer": vectorizer, "model": model}
    joblib.dump(pipeline, filepath)


# =============================================================================
# STEP 9: Load Pipeline
# =============================================================================


def load_pipeline(filepath):
    """
    Load a saved pipeline from disk.

    Args:
        filepath: Path to the saved pipeline

    Returns:
        tuple: (vectorizer, model)
    """
    pipeline = joblib.load(filepath)
    return pipeline["vectorizer"], pipeline["model"]


# =============================================================================
# STEP 10: Predict on New Text
# =============================================================================


def predict_sentiment(text, vectorizer, model, label_names=None):
    """
    Predict sentiment for a new text.

    Args:
        text: Raw text string
        vectorizer: Fitted TfidfVectorizer
        model: Trained classifier
        label_names: Optional list of label names

    Returns:
        dict: {'predicted_label': int, 'confidence': float, 'label_name': str}
    """
    # Preprocess
    cleaned_text = preprocess_text(text)

    # Vectorize
    features = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(features)[0]

    # Get confidence if available
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features)[0]
        confidence = probas[prediction]
    else:
        confidence = None

    result = {"predicted_label": int(prediction), "confidence": confidence}

    if label_names:
        result["label_name"] = label_names[prediction]

    return result


# =============================================================================
# MAIN PIPELINE DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Project: Sentiment Classifier Pipeline - SOLUTION")
    print("=" * 60)

    # Emotion labels
    EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    print("\n1. Loading Emotion dataset...")
    try:
        train_df, test_df = load_emotion_data(num_samples=2000)
        print(f"   Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    except Exception as e:
        print(f"   Could not load dataset: {e}")
        print("   Creating sample data for demonstration...")
        train_df = pd.DataFrame(
            {
                "text": [
                    "I am so happy today",
                    "This is terrible",
                    "I love you so much",
                    "I'm really angry",
                    "This is scary",
                ]
                * 100,
                "label": [1, 0, 2, 3, 4] * 100,
            }
        )
        test_df = train_df.sample(100, random_state=42)

    print("\n2. Preprocessing texts...")
    train_df["cleaned"] = train_df["text"].apply(preprocess_text)
    test_df["cleaned"] = test_df["text"].apply(preprocess_text)
    print(f"   Sample: '{train_df['text'].iloc[0]}' -> '{train_df['cleaned'].iloc[0]}'")

    print("\n3. Creating TF-IDF features...")
    X_train, X_test, vectorizer = create_tfidf_features(
        train_df["cleaned"].tolist(), test_df["cleaned"].tolist(), max_features=3000
    )
    print(f"   Features: {X_train.shape[1]}")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print("\n4. Training classifiers...")
    for clf_type in ["logistic", "random_forest", "naive_bayes"]:
        model = train_classifier(X_train, y_train, clf_type)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"   {clf_type}: accuracy={acc:.4f}")

    print("\n5. Comparing classifiers with cross-validation...")
    comparison = compare_classifiers(X_train, y_train, cv=3)
    print(comparison.to_string(index=False))

    print("\n6. Tuning best model...")
    tune_results = tune_best_model(X_train, y_train, cv=3)
    print(f"   Best params: {tune_results['best_params']}")
    print(f"   Best CV score: {tune_results['best_score']:.4f}")

    best_model = tune_results["best_model"]

    print("\n7. Evaluating on test set...")
    eval_results = evaluate_model(best_model, X_test, y_test)
    print(f"   Accuracy: {eval_results['accuracy']:.4f}")
    print(f"   F1 Score: {eval_results['f1']:.4f}")

    print("\n8-9. Saving and loading pipeline...")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        save_pipeline(vectorizer, best_model, f.name)
        loaded_vec, loaded_model = load_pipeline(f.name)
        print("   Pipeline saved and loaded successfully!")

    print("\n10. Predicting on new text...")
    test_texts = [
        "I am so happy and excited!",
        "This makes me really sad",
        "I love spending time with you",
    ]

    for text in test_texts:
        result = predict_sentiment(text, vectorizer, best_model, EMOTION_LABELS)
        label = result.get("label_name", result["predicted_label"])
        conf = result["confidence"]
        conf_str = f"{conf:.2f}" if conf else "N/A"
        print(f"   '{text}' -> {label} (confidence: {conf_str})")

    print("\n" + "=" * 60)
    print("Pipeline complete!")

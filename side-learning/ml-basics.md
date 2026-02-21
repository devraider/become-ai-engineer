# Machine Learning Basics

A refresher on fundamental ML concepts needed for Week 3 and beyond.

## Table of Contents

- [What is Machine Learning?](#what-is-machine-learning)
- [Types of Machine Learning](#types-of-machine-learning)
- [Train/Test Split](#traintest-split)
- [Overfitting vs Underfitting](#overfitting-vs-underfitting)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Cross-Validation](#cross-validation)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation Metrics](#model-evaluation-metrics)

---

## What is Machine Learning?

Machine Learning is the science of getting computers to learn from data without being explicitly programmed.

**Traditional Programming:**

```
Input + Rules → Output
```

**Machine Learning:**

```
Input + Output → Rules (Model)
```

The model learns patterns from data and can then make predictions on new, unseen data.

---

## Types of Machine Learning

### 1. Supervised Learning

Learn from labeled examples (input-output pairs).

| Type               | What it predicts   | Examples                                  |
| ------------------ | ------------------ | ----------------------------------------- |
| **Classification** | Categories/classes | Spam detection, sentiment analysis        |
| **Regression**     | Continuous values  | Price prediction, temperature forecasting |

```python
# Classification example
X = [[height, weight], ...]  # Features
y = ['cat', 'dog', ...]      # Labels (what we want to predict)
model.fit(X, y)
```

### 2. Unsupervised Learning

Find patterns in unlabeled data.

- **Clustering**: Group similar items (customer segmentation)
- **Dimensionality Reduction**: Compress data (PCA, t-SNE)
- **Association**: Find relationships (market basket analysis)

### 3. Reinforcement Learning

Learn by interacting with an environment and receiving rewards/penalties.

- Game playing (AlphaGo, chess)
- Robotics
- Recommendation systems

---

## Train/Test Split

**Why split data?**

If you evaluate on the same data you trained on, the model appears better than it actually is. It has "memorized" the answers rather than learned patterns.

```python
from sklearn.model_selection import train_test_split

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  # For reproducibility
)
```

**Typical splits:**

- **Train**: 60-80% - Model learns from this
- **Validation**: 10-20% - Tune hyperparameters
- **Test**: 10-20% - Final evaluation (only use once!)

**Stratification:**

```python
# Maintain class proportions in splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

---

## Overfitting vs Underfitting

### Underfitting (High Bias)

Model is too simple to capture the patterns.

**Symptoms:**

- Poor performance on training data
- Poor performance on test data
- High training error

**Causes:**

- Model too simple
- Not enough features
- Too much regularization

### Overfitting (High Variance)

Model memorizes training data instead of learning patterns.

**Symptoms:**

- Great performance on training data
- Poor performance on test data
- Large gap between train and test error

**Causes:**

- Model too complex
- Not enough training data
- Training too long
- Too many features

### How to Fix

| Problem          | Solutions                                                         |
| ---------------- | ----------------------------------------------------------------- |
| **Underfitting** | More complex model, more features, less regularization            |
| **Overfitting**  | More data, simpler model, regularization, dropout, early stopping |

```
Error
  │
  │   Underfitting │ Just Right │ Overfitting
  │       ↓        │     ↓      │     ↓
  │   ┌──────────────────────────────────┐
  │   │                                  │
  │   │    Training Error                │
  │   │ ────────────────────────         │
  │   │                    ┌───          │
  │   │    Validation Error│             │
  │   │ ───────────────────┘             │
  │   └──────────────────────────────────┘
  └─────────────────────────────────────────► Model Complexity
```

---

## Bias-Variance Tradeoff

**Bias**: Error from overly simplistic assumptions

- High bias = underfitting
- Model misses relevant patterns

**Variance**: Error from sensitivity to small fluctuations

- High variance = overfitting
- Model captures noise as if it were signal

**Total Error = Bias² + Variance + Noise**

The goal is to find the sweet spot that minimizes total error.

```
                    Total Error
       │               /
Error  │              /
       │   Bias²     /
       │    ↘      /
       │      ↘  /
       │       ↘⁻
       │      /  ↖
       │    /      Variance
       │  /
       └──────────────────────► Model Complexity
```

---

## Cross-Validation

A more robust way to evaluate models than a single train/test split.

### K-Fold Cross-Validation

1. Split data into K parts (folds)
2. Train on K-1 folds, test on 1 fold
3. Repeat K times, each fold serves as test once
4. Average the results

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```

**Visual:**

```
Fold 1: [TEST][Train][Train][Train][Train]
Fold 2: [Train][TEST][Train][Train][Train]
Fold 3: [Train][Train][TEST][Train][Train]
Fold 4: [Train][Train][Train][TEST][Train]
Fold 5: [Train][Train][Train][Train][TEST]
```

**Benefits:**

- Uses all data for training and testing
- More reliable estimate of model performance
- Helps detect overfitting

---

## Feature Engineering

The process of creating new features or transforming existing ones to improve model performance.

### Why It Matters

> "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." — Andrew Ng

### Common Techniques

**1. Scaling/Normalization**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-score normalization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**2. Encoding Categorical Variables**

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Label encoding (for ordinal: small < medium < large)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding (for nominal: red, green, blue)
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X_categorical)
```

**3. Text Features**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(texts)
```

**4. Feature Selection**

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Model Evaluation Metrics

### For Classification

| Metric        | Formula               | When to Use                                |
| ------------- | --------------------- | ------------------------------------------ |
| **Accuracy**  | (TP + TN) / Total     | Balanced classes                           |
| **Precision** | TP / (TP + FP)        | False positives costly (spam filter)       |
| **Recall**    | TP / (TP + FN)        | False negatives costly (disease detection) |
| **F1 Score**  | 2 × (P × R) / (P + R) | Imbalanced classes                         |
| **AUC-ROC**   | Area under ROC curve  | Binary classification ranking              |

**Confusion Matrix:**

```
                 Predicted
              |  Pos  |  Neg  |
    ----------|-------|-------|
    Actual Pos|  TP   |  FN   |
    Actual Neg|  FP   |  TN   |
```

- **True Positive (TP)**: Correctly predicted positive
- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

### For Regression

| Metric   | Description                                           |
| -------- | ----------------------------------------------------- |
| **MSE**  | Mean Squared Error - average of squared differences   |
| **RMSE** | Root MSE - in original units                          |
| **MAE**  | Mean Absolute Error - average of absolute differences |
| **R²**   | Coefficient of determination - % variance explained   |

---

## Summary Table

| Concept                 | Key Point                                        |
| ----------------------- | ------------------------------------------------ |
| **Machine Learning**    | Learn patterns from data to make predictions     |
| **Train/Test Split**    | Never evaluate on training data                  |
| **Overfitting**         | Model memorizes training data, fails on new data |
| **Underfitting**        | Model too simple to capture patterns             |
| **Bias-Variance**       | Balance between simplicity and complexity        |
| **Cross-Validation**    | More robust evaluation using all data            |
| **Feature Engineering** | Creating/transforming features to help the model |

---

## Further Reading

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer) - Excellent visual explanations

# Week 3 - Machine Learning Fundamentals

Master the foundations of machine learning with scikit-learn. Learn classification, regression, model evaluation, and build your first ML pipeline.

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [ML Glossary](https://ml-cheatsheet.readthedocs.io/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)

### Useful Video Tutorials

- [scikit-learn Tutorial](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)
- [Machine Learning for Beginners](https://www.youtube.com/watch?v=7eh4d6sabA0)

## Installation

```bash
cd week-3
uv init
uv venv
source .venv/bin/activate
uv add scikit-learn pandas numpy matplotlib seaborn datasets pytest
```

## Concepts

### 1. Machine Learning Overview

Machine Learning is teaching computers to learn patterns from data instead of explicit programming.

**Types of ML:**

| Type              | Goal                            | Examples                             |
| ----------------- | ------------------------------- | ------------------------------------ |
| **Supervised**    | Learn from labeled data         | Classification, Regression           |
| **Unsupervised**  | Find patterns in unlabeled data | Clustering, Dimensionality reduction |
| **Reinforcement** | Learn from rewards/penalties    | Game AI, Robotics                    |

**The ML workflow:**

```
Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

---

### 2. scikit-learn Basics

scikit-learn provides a consistent API for all ML algorithms:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Preprocess (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Predict
y_pred = model.predict(X_test_scaled)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

**Key pattern:** Every scikit-learn estimator has:

- `.fit(X, y)` - Learn from data
- `.predict(X)` - Make predictions
- `.score(X, y)` - Evaluate performance

**Common preprocessing:**

```python
# Scaling (for algorithms sensitive to feature magnitude)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Handling missing values
from sklearn.impute import SimpleImputer
```

📝 **Exercise 1 (Basic)** - Practice scikit-learn basics in [exercises/exercise_basic_1_sklearn_basics.py](exercises/exercise_basic_1_sklearn_basics.py)

> 📖 Need ML fundamentals? See [side-learning/ml-basics.md](../side-learning/ml-basics.md)

---

### 3. Classification Algorithms

Classification predicts discrete categories (spam/not spam, positive/negative).

**Popular algorithms:**

| Algorithm               | When to Use                 | Pros                         | Cons                                |
| ----------------------- | --------------------------- | ---------------------------- | ----------------------------------- |
| **Logistic Regression** | Binary/multiclass, baseline | Fast, interpretable          | Linear boundaries                   |
| **Random Forest**       | General purpose             | Handles non-linear, robust   | Slower, less interpretable          |
| **SVM**                 | High-dimensional data       | Effective in high dimensions | Slow on large datasets              |
| **Naive Bayes**         | Text classification         | Fast, works with small data  | Assumes feature independence        |
| **KNN**                 | Small datasets              | Simple, no training          | Slow prediction, sensitive to scale |

**Example - Text Classification:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create a pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train
text_clf.fit(train_texts, train_labels)

# Predict
predictions = text_clf.predict(test_texts)
```

---

### 4. Model Evaluation

Never evaluate on training data! Always use held-out test data or cross-validation.

**Train/Test Split:**

```python
from sklearn.model_selection import train_test_split

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,      # Reproducibility
    stratify=y            # Maintain class balance
)
```

**Cross-Validation:**

```python
from sklearn.model_selection import cross_val_score

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

**Classification Metrics:**

| Metric        | Formula               | When to Use                     |
| ------------- | --------------------- | ------------------------------- |
| **Accuracy**  | Correct / Total       | Balanced classes                |
| **Precision** | TP / (TP + FP)        | Cost of false positives is high |
| **Recall**    | TP / (TP + FN)        | Cost of false negatives is high |
| **F1 Score**  | 2 _ (P _ R) / (P + R) | Imbalanced classes              |
| **AUC-ROC**   | Area under ROC curve  | Binary classification ranking   |

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# All metrics at once
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

**Confusion Matrix:**

```
                 Predicted
              |  Pos  |  Neg  |
    ----------|-------|-------|
    Actual Pos|  TP   |  FN   |
    Actual Neg|  FP   |  TN   |
```

📝 **Exercise 2 (Intermediate)** - Practice model evaluation in [exercises/exercise_intermediate_2_model_evaluation.py](exercises/exercise_intermediate_2_model_evaluation.py)

---

### 5. Feature Engineering

Feature engineering often matters more than algorithm choice!

**Common techniques:**

```python
# 1. Text features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(texts)

# 2. Numerical scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 3. Categorical encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)

# 4. Feature selection
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=100)
X_selected = selector.fit_transform(X, y)
```

**For text/NLP:**

- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)
- N-grams
- Text length, word count, etc.

---

### 6. Pipelines

Pipelines chain preprocessing and modeling into one object:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])

# Use like any estimator
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
score = pipe.score(X_test, y_test)
```

**Benefits:**

- Prevents data leakage (preprocessing fitted only on train)
- Cleaner code
- Easy to save/load models
- Works with cross-validation

---

### 7. Hyperparameter Tuning

Find the best model parameters with GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

**RandomizedSearchCV** for large parameter spaces:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': uniform(0.01, 0.1)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=50,  # Number of random combinations to try
    cv=5,
    random_state=42
)
```

---

## Weekly Project

Build a **Sentiment Classifier** - A complete ML pipeline that:

1. Loads the emotion dataset from HuggingFace
2. Preprocesses text data (cleaning, TF-IDF)
3. Trains multiple classifiers (Logistic Regression, Random Forest, Naive Bayes)
4. Compares models using cross-validation
5. Tunes hyperparameters for the best model
6. Evaluates on test set with full metrics
7. Saves the final model

📝 **Project file:** [exercises/project_pipeline.py](exercises/project_pipeline.py)
📝 **Tests:** [exercises/tests/test_project_pipeline.py](exercises/tests/test_project_pipeline.py)

📝 **Exercise 3 (Advanced)** - Master pipelines & feature engineering in [exercises/exercise_advanced_3_pipelines.py](exercises/exercise_advanced_3_pipelines.py)

---

## Interview Questions

### Basic Level

1. **What is the difference between supervised and unsupervised learning?**
   - Supervised: Learn from labeled data (X, y pairs) - classification, regression
   - Unsupervised: Find patterns in unlabeled data (X only) - clustering, dimensionality reduction

2. **Why do we split data into train and test sets?**
   - To evaluate how well the model generalizes to unseen data
   - Training metrics are overly optimistic (model memorizes training data)
   - Test set simulates real-world performance

3. **What is overfitting and how do you prevent it?**
   - Model learns noise in training data, fails on new data
   - Prevention: More data, regularization, cross-validation, simpler models, early stopping

### Intermediate Level

4. **When would you use precision vs recall?**
   - Precision: When false positives are costly (spam filter - don't block important emails)
   - Recall: When false negatives are costly (disease detection - don't miss sick patients)
   - F1: When you need balance between both

5. **Explain cross-validation and why it's better than a single train/test split.**
   - Split data into k folds, train on k-1, test on 1, rotate
   - More reliable estimate - tests on all data points
   - Helps detect variance in model performance
   - Single split can be lucky/unlucky

6. **What is the purpose of feature scaling?**
   - Many algorithms (SVM, KNN, neural networks) are sensitive to feature magnitude
   - Features with larger values dominate distance calculations
   - Gradient descent converges faster with scaled features
   - Not needed for tree-based methods

### Advanced Level

7. **How do you handle imbalanced datasets?**
   - Resampling: Oversample minority (SMOTE), undersample majority
   - Class weights: Penalize errors on minority class more
   - Different metrics: Use F1, AUC-ROC instead of accuracy
   - Ensemble methods: Balanced random forest

8. **Explain the bias-variance tradeoff.**
   - Bias: Error from model being too simple (underfitting)
   - Variance: Error from model being too complex (overfitting)
   - Goal: Find sweet spot where total error is minimized
   - Complex models = low bias, high variance
   - Simple models = high bias, low variance

9. **How would you approach a text classification problem from scratch?**
   - EDA: Class distribution, text lengths, common words
   - Preprocessing: Clean text, handle missing values
   - Feature engineering: TF-IDF, n-grams, text statistics
   - Baseline model: Logistic regression with TF-IDF
   - Try multiple models, compare with cross-validation
   - Tune best model, evaluate on held-out test set
   - Consider embeddings/transformers if baseline insufficient

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Explain the difference between supervised and unsupervised learning
- [ ] Use scikit-learn's fit/predict/score API pattern
- [ ] Split data properly with train_test_split (including stratification)
- [ ] Apply common preprocessing (scaling, encoding, TF-IDF)
- [ ] Train classification models (Logistic Regression, Random Forest, Naive Bayes)
- [ ] Evaluate models with accuracy, precision, recall, F1 score
- [ ] Interpret a confusion matrix and classification report
- [ ] Use cross-validation for reliable model comparison
- [ ] Build sklearn Pipelines for clean, leak-free workflows
- [ ] Tune hyperparameters with GridSearchCV
- [ ] Save and load trained models with joblib
- [ ] Explain the bias-variance tradeoff in interviews

**[→ View Full Roadmap](ROADMAP.md)** | **[→ Begin Week 4](../week-4/README.md)**

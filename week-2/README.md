# Week 2 - Data Foundations for AI

> **Goal**: Master data manipulation skills needed for AI/ML work. After this week, you'll prepare datasets, understand embeddings, and work with real AI data formats.

---

## 📚 References

| Topic        | Link                                        |
| ------------ | ------------------------------------------- |
| NumPy Docs   | https://numpy.org/doc/                      |
| Pandas Docs  | https://pandas.pydata.org/docs/             |
| NumPy Video  | https://www.youtube.com/watch?v=QUT1VHiLmmI |
| Pandas Video | https://www.youtube.com/watch?v=vmEHCJofslg |

---

## 🛠 Installation

```bash
# Create a fresh project for Week 2
cd week-2
uv init week2-data
cd week2-data
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv add numpy pandas matplotlib seaborn jupyter datasets scikit-learn
```

---

## Part 1: NumPy Essentials for AI

NumPy is the backbone of all AI/ML libraries. Every neural network, embedding, and model uses NumPy arrays under the hood.

### Why NumPy for AI?

- **Embeddings** are NumPy arrays (e.g., word2vec, BERT outputs)
- **Model weights** are stored as arrays
- **Batch processing** requires array operations
- **GPU libraries** (PyTorch, TensorFlow) convert to/from NumPy

### 1.1 Arrays - The Building Block

```python
import numpy as np

# 1D array - like a sentence embedding
embedding = np.array([0.1, 0.5, -0.3, 0.8, 0.2])
print(f"Shape: {embedding.shape}")  # (5,)

# 2D array - batch of embeddings (like multiple sentences)
batch_embeddings = np.array([
    [0.1, 0.5, -0.3],  # sentence 1
    [0.2, 0.1, 0.9],   # sentence 2
    [0.8, -0.2, 0.4]   # sentence 3
])
print(f"Shape: {batch_embeddings.shape}")  # (3, 3) = 3 sentences, 3 dimensions

# 3D array - batch of sequences (like transformer input)
# Shape: (batch_size, sequence_length, embedding_dim)
transformer_input = np.random.randn(2, 4, 8)  # 2 samples, 4 tokens, 8-dim embeddings
print(f"Transformer input shape: {transformer_input.shape}")
```

### 1.2 Common Operations You'll Use Daily

```python
# Create special arrays
zeros = np.zeros((10, 768))      # Initialize embedding matrix
ones = np.ones((100,))           # Mask array
random = np.random.randn(5, 3)   # Random normal (for weight initialization)

# Reshape - critical for model inputs
flat = np.array([1, 2, 3, 4, 5, 6])
matrix = flat.reshape(2, 3)      # Convert to 2x3 matrix
print(matrix)
# [[1 2 3]
#  [4 5 6]]

# Transpose - swap dimensions
print(matrix.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### 1.3 Math Operations (Used in Similarity & Attention)

```python
# Dot product - used in attention mechanism and similarity
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
similarity = np.dot(vec1, vec2)  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication - core of neural networks
weights = np.random.randn(768, 256)  # Linear layer weights
input_data = np.random.randn(32, 768)  # 32 samples, 768 features
output = input_data @ weights  # Shape: (32, 256)

# Cosine similarity - comparing embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

emb1 = np.array([1, 0, 0])
emb2 = np.array([1, 1, 0])
print(f"Similarity: {cosine_similarity(emb1, emb2):.3f}")  # ~0.707
```

### 1.4 Softmax - The AI Essential Function

```python
def softmax(x):
    """Convert scores to probabilities - used in every classifier & attention"""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probabilities: {probs}")  # [0.659, 0.242, 0.099]
print(f"Sum: {probs.sum()}")      # 1.0
```

### ✅ Exercise 1 (Basic): Embedding Operations

> 📁 **Complete the exercise**: [exercises/exercise_basic_1_numpy.py](exercises/exercise_basic_1_numpy.py)
>
> Run with: `python exercises/exercise_basic_1_numpy.py`
>
> Run tests: `python -m pytest exercises/tests/test_exercise_basic_1_numpy.py -v`
>
> 📖 Need a Python refresher? See [side-learning/python-basics.md](../side-learning/python-basics.md)

```python
# TODO: Complete these exercises

# 1. Create a random embedding matrix for 1000 words, each with 384 dimensions
# vocab_embeddings = ???

# 2. Get the embedding for word at index 42
# word_42_embedding = ???

# 3. Calculate cosine similarity between word 10 and word 20
# similarity = ???

# 4. Find the 5 most similar words to word 100 (hint: use argpartition or argsort)
# top_5_indices = ???
```

> 💡 **Solution**: See [exercises/solutions/solution_basic_1_numpy.py](exercises/solutions/solution_basic_1_numpy.py)

---

## Part 2: Pandas for AI Data Preparation

Pandas is how you load, clean, and prepare data before feeding it to AI models.

### 2.1 Loading Data (Common AI Formats)

```python
import pandas as pd

# From CSV (most common)
df = pd.read_csv("dataset.csv")

# From JSON (API responses, HuggingFace datasets)
df = pd.read_json("data.json")

# From HuggingFace datasets (you'll use this a lot!)
# pip install datasets
from datasets import load_dataset
dataset = load_dataset("imdb", split="train")
df = pd.DataFrame(dataset)
```

### 2.2 Quick Data Exploration

```python
# Create sample AI training data
data = {
    "text": [
        "I love this product!",
        "Terrible experience",
        "It's okay, nothing special",
        "Best purchase ever!",
        "Would not recommend"
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative"],
    "confidence": [0.95, 0.87, 0.62, 0.91, 0.78]
}
df = pd.DataFrame(data)

# Essential exploration commands
print(df.head())           # First 5 rows
print(df.info())           # Data types and memory
print(df.describe())       # Statistics
print(df["label"].value_counts())  # Class distribution - critical for ML!
```

### 2.3 Data Cleaning for ML

```python
# Handle missing values
df = df.dropna()                    # Remove rows with missing values
df = df.fillna("")                  # Or fill with empty string
df["confidence"] = df["confidence"].fillna(df["confidence"].mean())  # Fill with mean

# Remove duplicates (common in scraped data)
df = df.drop_duplicates(subset=["text"])

# Text cleaning for NLP
df["text_clean"] = df["text"].str.lower()           # Lowercase
df["text_clean"] = df["text_clean"].str.strip()     # Remove whitespace
df["word_count"] = df["text"].str.split().str.len() # Count words

# Filter by conditions
df_positive = df[df["label"] == "positive"]
df_confident = df[df["confidence"] > 0.8]
```

### 2.4 Preparing Data for Training

```python
# Label encoding - convert text labels to numbers
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label_id"] = df["label"].map(label_map)

# One-hot encoding (for multi-class)
one_hot = pd.get_dummies(df["label"], prefix="label")
df = pd.concat([df, one_hot], axis=1)

# Train/Test split preparation
from sklearn.model_selection import train_test_split

texts = df["text"].tolist()
labels = df["label_id"].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
```

### 2.5 Real-World Example: Preparing a Sentiment Dataset

```python
import pandas as pd
from datasets import load_dataset

# Load real dataset from HuggingFace
dataset = load_dataset("amazon_polarity", split="train[:1000]")  # First 1000 samples
df = pd.DataFrame(dataset)

print("Original columns:", df.columns.tolist())
print(df.head())

# Rename for clarity
df = df.rename(columns={"content": "text", "label": "sentiment"})

# Check class balance
print("\nClass distribution:")
print(df["sentiment"].value_counts())

# Clean text
df["text"] = df["text"].str.strip()
df["text_length"] = df["text"].str.len()

# Filter very short/long texts
df = df[(df["text_length"] > 10) & (df["text_length"] < 1000)]

# Export for training
df.to_csv("prepared_sentiment_data.csv", index=False)
print(f"\nSaved {len(df)} samples")
```

### ✅ Exercise 2 (Intermediate): Prepare a Dataset for Classification

> 📁 **Complete the exercise**: [exercises/exercise_intermediate_2_pandas.py](exercises/exercise_intermediate_2_pandas.py)
>
> Run with: `python exercises/exercise_intermediate_2_pandas.py`
>
> Run tests: `python -m pytest exercises/tests/test_exercise_intermediate_2_pandas.py -v`

```python
# TODO: Complete these tasks

# 1. Load the "emotion" dataset from HuggingFace
# dataset = load_dataset("emotion", split="train")
# df = ???

# 2. Check the class distribution - are classes balanced?
# ???

# 3. Add a column "text_length" with the character count of each text
# ???

# 4. Find the average text length per emotion
# ???

# 5. Create train/test DataFrames (80/20 split)
# ???
```

> 💡 **Solution**: See [exercises/solutions/solution_intermediate_2_pandas.py](exercises/solutions/solution_intermediate_2_pandas.py)

---

## Part 3: Visualization for AI

Quick charts help you understand data and debug models.

### 3.1 Essential Plots for ML

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Class distribution - always check this first!
labels = ["positive", "negative", "neutral"]
counts = [450, 380, 170]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['green', 'red', 'gray'])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.savefig("class_distribution.png")
plt.show()

# Training loss curve - you'll see this constantly
epochs = range(1, 11)
train_loss = [2.3, 1.8, 1.2, 0.8, 0.5, 0.35, 0.25, 0.2, 0.15, 0.12]
val_loss = [2.4, 1.9, 1.4, 1.0, 0.7, 0.55, 0.5, 0.48, 0.47, 0.47]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, 'b-', label='Train Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.savefig("training_curve.png")
plt.show()
```

### 3.2 Confusion Matrix - Model Evaluation

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Simulated predictions
y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 0, 2, 2, 1]

cm = confusion_matrix(y_true, y_pred)
labels = ["Negative", "Neutral", "Positive"]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()
```

---

## 🚀 Week 2 Project: Build a Data Pipeline for Sentiment Analysis

Create a complete data preparation pipeline that you'll use in later weeks.

### Project Requirements

1. **Load Data**: Use the `amazon_polarity` or `yelp_review_full` dataset from HuggingFace
2. **Explore**: Show class distribution and basic statistics
3. **Clean**: Handle missing values, clean text, filter by length
4. **Prepare**: Create train/validation/test splits with stratification
5. **Analyze**: Create visualizations (class distribution, text length distribution)
6. **Export**: Save processed data to CSV files

### Project Files

> 📁 **Complete the project**: [exercises/project_pipeline.py](exercises/project_pipeline.py)
>
> Run with: `python exercises/project_pipeline.py`
>
> Run tests: `python -m pytest exercises/tests/test_project.py -v`
>
> 💡 **Solution**: See [exercises/solutions/solution_project.py](exercises/solutions/solution_project.py)

### ✅ Exercise 3 (Advanced): Data Analysis for AI

> 📁 **Complete the exercise**: [exercises/exercise_advanced_3_data_analysis.py](exercises/exercise_advanced_3_data_analysis.py)
>
> Run tests: `python -m pytest exercises/tests/test_exercise_advanced_3_data_analysis.py -v`
>
> Topics: Embedding statistics, batch similarity, clustering, pairwise distances
>
> 💡 **Solution**: See [exercises/solutions/solution_advanced_3_data_analysis.py](exercises/solutions/solution_advanced_3_data_analysis.py)

### Expected Output

```
📊 Dataset loaded: 5000 samples
📋 Columns: ['text', 'label']

Class Distribution:
  Positive: 2500 (50%)
  Negative: 2500 (50%)

Text Length Stats:
  Mean: 234 characters
  Min: 15 characters
  Max: 892 characters

✅ Cleaned: 4823 samples (removed 177 outliers)
✅ Created visualizations: class_dist.png, length_dist.png
✅ Saved splits: train(3858), val(482), test(483)
✅ Pipeline complete!
```

---

## 📝 Interview Questions - Week 2

Practice answering these:

1. **What is the difference between a NumPy array and a Python list?**
   - Arrays are homogeneous (same type), fixed size, contiguous memory
   - Much faster for numerical operations (vectorized)
   - Support broadcasting and advanced indexing

2. **How do you handle imbalanced classes in a dataset?**
   - Oversampling minority class (SMOTE)
   - Undersampling majority class
   - Class weights in loss function
   - Stratified splits to maintain proportions

3. **What is stratification in train/test split?**
   - Ensures each split has the same class distribution as the original
   - Critical for imbalanced datasets

4. **How do you normalize numerical features?**
   - Min-Max: `(x - min) / (max - min)` → [0, 1]
   - Z-score: `(x - mean) / std` → mean=0, std=1

5. **What's the shape of input to a transformer model?**
   - `(batch_size, sequence_length, hidden_dim)`
   - Example: `(32, 512, 768)` for BERT

---

## ✅ Week 2 Takeaways

After this week, you should be able to:

- [ ] Create and manipulate NumPy arrays (shapes, reshaping, transpose)
- [ ] Calculate cosine similarity between embeddings
- [ ] Implement softmax from scratch
- [ ] Load data from CSV, JSON, and HuggingFace datasets
- [ ] Clean and preprocess text data with Pandas
- [ ] Check and handle class imbalance
- [ ] Create train/val/test splits with stratification
- [ ] Visualize class distribution and training metrics
- [ ] Explain these concepts in an interview

---

## 🔜 Next Week Preview

**Week 3: Machine Learning Fundamentals**

- scikit-learn basics
- Classification algorithms
- Model evaluation and cross-validation
- Building ML pipelines
- Build: A chatbot with memory

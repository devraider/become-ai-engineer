"""
Week 2 - Exercise 2: SOLUTIONS
==============================

This file contains the complete solutions for Exercise 2.
Try to solve the exercises yourself first before looking at these!
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_emotion_dataset() -> pd.DataFrame:
    """
    Task 1: Load the 'emotion' dataset from HuggingFace.

    Solution: Use the datasets library to load and convert to pandas.
    """
    from datasets import load_dataset

    # Load the training split of the emotion dataset
    dataset = load_dataset("emotion", split="train")

    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)

    return df


def check_class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """
    Task 2: Get the distribution of classes in the dataset.

    Solution: Use value_counts() which returns counts sorted by default.
    """
    return df[label_col].value_counts()


def add_text_length(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Task 3: Add a 'text_length' column with character count.

    Solution:
    - Use .copy() to avoid modifying original
    - Use .str.len() for character count
    """
    # Create a copy to avoid modifying the original
    result = df.copy()

    # Add text_length column
    result["text_length"] = result[text_col].str.len()

    return result


def average_length_per_class(df: pd.DataFrame) -> pd.Series:
    """
    Task 4: Calculate average text length for each label/class.

    Solution: Use groupby + mean
    """
    return df.groupby("label")["text_length"].mean()


def create_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task 5: Split DataFrame into train and test sets with stratification.

    Solution: Use sklearn's train_test_split with stratify parameter.
    Stratification ensures each split has the same class distribution.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],  # This is the key for stratification!
    )

    return train_df, test_df


def clean_text_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Task 6 (Bonus): Clean text data for NLP tasks.

    Solution: Chain string operations and remove duplicates.
    """
    # Create a copy
    result = df.copy()

    # Apply cleaning steps
    result[text_col] = (
        result[text_col]
        .str.lower()  # 1. Convert to lowercase
        .str.strip()  # 2. Remove leading/trailing whitespace
    )

    # 3. Remove duplicates based on text
    result = result.drop_duplicates(subset=[text_col])

    return result


def filter_by_length(
    df: pd.DataFrame, min_length: int = 10, max_length: int = 500
) -> pd.DataFrame:
    """
    Task 7 (Bonus): Filter rows by text length.

    Solution: Use boolean indexing with conditions.
    """
    return df[(df["text_length"] >= min_length) & (df["text_length"] <= max_length)]


# =============================================================================
# ADDITIONAL USEFUL PATTERNS
# =============================================================================


def add_word_count(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Additional: Add word count column.

    Useful for filtering out very short or very long texts.
    """
    result = df.copy()
    result["word_count"] = result[text_col].str.split().str.len()
    return result


def add_emotion_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional: Add human-readable emotion names.

    The emotion dataset uses numeric labels, this adds readable names.
    """
    result = df.copy()

    label_names = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    result["emotion"] = result["label"].map(label_names)
    return result


def create_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Additional: Create three-way split (train/val/test).

    Common pattern: 80% train, 10% validation, 10% test
    """
    from sklearn.model_selection import train_test_split

    # First split: separate test set
    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Second split: separate validation from training
    # Adjust val_size to account for smaller dataset
    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val["label"],
    )

    return train_df, val_df, test_df


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("Exercise 2 Solutions Demo")
    print("=" * 50)

    # Task 1
    print("\n1. Loading emotion dataset...")
    try:
        df = load_emotion_dataset()
        print(f"   Loaded {len(df)} samples")
        print(f"   Columns: {df.columns.tolist()}")
    except ImportError:
        print("   HuggingFace datasets not installed. Using sample data.")
        df = pd.DataFrame(
            {
                "text": ["I am happy", "This is sad", "I love this", "So angry!"] * 100,
                "label": [1, 0, 2, 3] * 100,
            }
        )

    # Task 2
    print("\n2. Class distribution:")
    dist = check_class_distribution(df)
    print(dist)

    # Task 3
    print("\n3. Adding text_length column...")
    df = add_text_length(df)
    print(f"   Sample lengths: {df['text_length'].head().tolist()}")

    # Task 4
    print("\n4. Average length per class:")
    avg = average_length_per_class(df)
    print(avg)

    # Task 5
    print("\n5. Train/test split:")
    train_df, test_df = create_train_test_split(df)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

    # Check stratification worked
    print("\n   Verifying stratification:")
    train_dist = train_df["label"].value_counts(normalize=True)
    test_dist = test_df["label"].value_counts(normalize=True)
    print(f"   Train class 0 ratio: {train_dist.get(0, 0):.3f}")
    print(f"   Test class 0 ratio: {test_dist.get(0, 0):.3f}")

    # Task 6 (Bonus)
    print("\n6. Cleaning text data...")
    dirty_df = pd.DataFrame(
        {"text": ["  HELLO  ", "hello", "WORLD", "  world  "], "label": [0, 1, 2, 3]}
    )
    cleaned = clean_text_data(dirty_df)
    print(f"   Before: {len(dirty_df)} rows")
    print(f"   After: {len(cleaned)} rows")

    # Task 7 (Bonus)
    print("\n7. Filtering by length...")
    filtered = filter_by_length(df, min_length=50, max_length=200)
    print(f"   Before: {len(df)} rows")
    print(f"   After: {len(filtered)} rows")

    # Additional demos
    print("\n--- Additional Patterns ---")

    df_with_names = add_emotion_names(df)
    print(f"\nEmotion names added: {df_with_names['emotion'].unique()}")

    train, val, test = create_train_val_test_split(df)
    print(f"\n3-way split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

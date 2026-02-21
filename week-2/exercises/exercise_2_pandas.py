"""
Week 2 - Exercise 2: Pandas for AI Data Preparation
====================================================

Complete the TODO sections below and run the tests to verify your solution.

Run this file to check your progress:
    python exercise_2_pandas.py

Run tests:
    python -m pytest tests/test_exercise_2.py -v

Note: This exercise uses HuggingFace datasets. Install with:
    uv add datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def load_emotion_dataset() -> pd.DataFrame:
    """
    Task 1: Load the 'emotion' dataset from HuggingFace.

    The emotion dataset contains text samples labeled with emotions:
    0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise

    Returns:
        DataFrame with columns: ['text', 'label']

    Example:
        >>> df = load_emotion_dataset()
        >>> df.columns.tolist()
        ['text', 'label']
    """
    # TODO: Load the dataset and convert to DataFrame
    # Hint: from datasets import load_dataset
    #       dataset = load_dataset("emotion", split="train")
    #       Then convert to DataFrame
    pass


def check_class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """
    Task 2: Get the distribution of classes in the dataset.

    Args:
        df: Input DataFrame
        label_col: Name of the label column

    Returns:
        Series with counts for each class, sorted by count (descending)

    Example:
        >>> dist = check_class_distribution(df)
        >>> dist.index[0]  # Most common class
        1
    """
    # TODO: Return value counts of the label column
    pass


def add_text_length(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Task 3: Add a 'text_length' column with character count.

    Args:
        df: Input DataFrame
        text_col: Name of the text column

    Returns:
        DataFrame with new 'text_length' column added

    Example:
        >>> df = add_text_length(df)
        >>> 'text_length' in df.columns
        True
    """
    # TODO: Add text_length column
    # Hint: Use .str.len()
    # Important: Return a copy, don't modify the original
    pass


def average_length_per_class(df: pd.DataFrame) -> pd.Series:
    """
    Task 4: Calculate average text length for each label/class.

    Args:
        df: DataFrame with 'label' and 'text_length' columns

    Returns:
        Series with average length per label

    Example:
        >>> avg = average_length_per_class(df)
        >>> len(avg)  # Number of classes
        6
    """
    # TODO: Group by label and calculate mean of text_length
    # Hint: Use .groupby() and .mean()
    pass


def create_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Task 5: Split DataFrame into train and test sets with stratification.

    Stratification ensures each split has the same class distribution
    as the original dataset.

    Args:
        df: Input DataFrame with 'label' column
        test_size: Fraction of data for test set (0.2 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)

    Example:
        >>> train_df, test_df = create_train_test_split(df)
        >>> len(train_df) > len(test_df)
        True
    """
    # TODO: Split the data with stratification
    # Hint: from sklearn.model_selection import train_test_split
    #       Use stratify=df['label'] parameter
    pass


def clean_text_data(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Task 6 (Bonus): Clean text data for NLP tasks.

    Apply these cleaning steps:
    1. Convert to lowercase
    2. Strip whitespace
    3. Remove duplicate rows based on text

    Args:
        df: Input DataFrame
        text_col: Name of the text column

    Returns:
        Cleaned DataFrame
    """
    # TODO: Implement text cleaning
    # Important: Return a copy, don't modify the original
    pass


def filter_by_length(
    df: pd.DataFrame, min_length: int = 10, max_length: int = 500
) -> pd.DataFrame:
    """
    Task 7 (Bonus): Filter rows by text length.

    Args:
        df: DataFrame with 'text_length' column
        min_length: Minimum text length (inclusive)
        max_length: Maximum text length (inclusive)

    Returns:
        Filtered DataFrame
    """
    # TODO: Filter rows where text_length is within range
    pass


# =============================================================================
# QUICK CHECK - Run this file directly to test your solutions
# =============================================================================

# Check if HuggingFace datasets is available
try:
    from datasets import load_dataset as _hf_load

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


if __name__ == "__main__":
    print("=" * 60)
    print("Week 2 - Exercise 2: Pandas for AI Data Preparation")
    print("=" * 60)

    all_passed = True

    # Create sample data for testing
    sample_df = pd.DataFrame(
        {
            "text": [
                "I am so happy today!",
                "This is terrible news",
                "I love you so much",
                "Why are you angry?",
                "I'm scared of the dark",
                "What a surprise!",
            ],
            "label": [1, 0, 2, 3, 4, 5],
        }
    )

    # Task 1: Load dataset
    print("\n📝 Task 1: Load emotion dataset...")
    try:
        if HF_AVAILABLE:
            df = load_emotion_dataset()
            if df is not None and len(df) > 1000:
                print(f"   ✅ PASSED - Loaded {len(df)} samples")
            else:
                print(f"   ❌ FAILED - Got: {df}")
                all_passed = False
        else:
            print("   ⏭️  SKIPPED - HuggingFace datasets not installed")
            print("      Install with: uv add datasets")
            df = sample_df  # Use sample for other tests
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False
        df = sample_df

    # Task 2: Class distribution
    print("\n📝 Task 2: Check class distribution...")
    try:
        dist = check_class_distribution(df)
        if dist is not None and isinstance(dist, pd.Series):
            print(f"   ✅ PASSED - Found {len(dist)} classes")
        else:
            print(f"   ❌ FAILED - Expected Series, got: {type(dist)}")
            all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 3: Add text length
    print("\n📝 Task 3: Add text_length column...")
    try:
        df_with_length = add_text_length(df)
        if df_with_length is not None and "text_length" in df_with_length.columns:
            print(f"   ✅ PASSED - Added text_length column")
        else:
            print(f"   ❌ FAILED - text_length column not found")
            all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 4: Average length per class
    print("\n📝 Task 4: Average length per class...")
    try:
        if df_with_length is not None:
            avg = average_length_per_class(df_with_length)
            if avg is not None and isinstance(avg, pd.Series):
                print(f"   ✅ PASSED - Calculated averages for {len(avg)} classes")
            else:
                print(f"   ❌ FAILED - Got None or wrong type")
                all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 5: Train/test split
    print("\n📝 Task 5: Create train/test split...")
    try:
        # Use larger sample for split
        large_df = pd.concat([sample_df] * 10, ignore_index=True)
        train_df, test_df = create_train_test_split(large_df, test_size=0.2)
        if train_df is not None and test_df is not None:
            print(f"   ✅ PASSED - Train: {len(train_df)}, Test: {len(test_df)}")
        else:
            print(f"   ❌ FAILED - Got None")
            all_passed = False
    except Exception as e:
        print(f"   ❌ ERROR - {e}")
        all_passed = False

    # Task 6: Clean text (Bonus)
    print("\n📝 Task 6 (Bonus): Clean text data...")
    try:
        dirty_df = pd.DataFrame(
            {"text": ["  HELLO  ", "hello", "WORLD"], "label": [0, 1, 2]}
        )
        cleaned = clean_text_data(dirty_df)
        if cleaned is not None:
            print(f"   ✅ PASSED - Cleaned {len(dirty_df)} → {len(cleaned)} rows")
        else:
            print(f"   ⏭️  SKIPPED (bonus task)")
    except Exception as e:
        print(f"   ⏭️  SKIPPED - {e}")

    # Task 7: Filter by length (Bonus)
    print("\n📝 Task 7 (Bonus): Filter by length...")
    try:
        if df_with_length is not None:
            filtered = filter_by_length(df_with_length, min_length=20, max_length=100)
            if filtered is not None:
                print(f"   ✅ PASSED - Filtered to {len(filtered)} rows")
            else:
                print(f"   ⏭️  SKIPPED (bonus task)")
    except Exception as e:
        print(f"   ⏭️  SKIPPED - {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TASKS PASSED! Great job!")
        print("\nRun full tests: python -m pytest tests/test_exercise_2.py -v")
    else:
        print("❌ Some tasks need work. Keep trying!")
    print("=" * 60)

"""
Week 2 Project: Sentiment Data Pipeline
========================================

Build a complete data preparation pipeline for sentiment analysis.

Requirements:
1. Load data from HuggingFace
2. Explore and show statistics
3. Clean and preprocess
4. Create train/val/test splits with stratification
5. Visualize class distribution
6. Export to CSV files

Run this file:
    python project_pipeline.py

Run tests:
    python -m pytest tests/test_project.py -v
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def load_data(
    dataset_name: str = "amazon_polarity", num_samples: int = 5000
) -> pd.DataFrame:
    """
    Task 1: Load dataset from HuggingFace.

    Args:
        dataset_name: Name of the HuggingFace dataset
        num_samples: Number of samples to load (use slicing for speed)

    Returns:
        DataFrame with 'text' and 'label' columns

    Hints:
        - from datasets import load_dataset
        - Use split=f"train[:{num_samples}]" to limit samples
        - The amazon_polarity dataset has 'content' (rename to 'text') and 'label'
    """
    # TODO: Implement
    pass


def explore_data(df: pd.DataFrame) -> dict:
    """
    Task 2: Explore the dataset and return statistics.

    Args:
        df: Input DataFrame with 'text' and 'label' columns

    Returns:
        Dictionary with keys:
        - 'num_samples': Total number of samples
        - 'num_classes': Number of unique classes
        - 'class_distribution': Series with counts per class
        - 'avg_text_length': Average character length of text

    Should also print the statistics to console.
    """
    # TODO: Implement
    pass


def clean_data(
    df: pd.DataFrame, min_length: int = 10, max_length: int = 1000
) -> pd.DataFrame:
    """
    Task 3: Clean and preprocess the data.

    Steps to implement:
    1. Remove rows with missing text
    2. Strip whitespace from text
    3. Add 'text_length' column
    4. Filter rows outside min/max length range
    5. Remove duplicate texts

    Args:
        df: Input DataFrame
        min_length: Minimum text length to keep
        max_length: Maximum text length to keep

    Returns:
        Cleaned DataFrame
    """
    # TODO: Implement
    pass


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Task 4: Create train/validation/test splits with stratification.

    Args:
        df: Input DataFrame with 'label' column
        train_size: Fraction for training (default 0.8)
        val_size: Fraction for validation (default 0.1)
        test_size: Fraction for testing (default 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    Hints:
        - Use sklearn.model_selection.train_test_split
        - Split twice: first separate test, then split train into train+val
        - Use stratify parameter for balanced splits
    """
    # TODO: Implement
    pass


def visualize(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Task 5: Create and save visualizations.

    Create two plots:
    1. Class distribution bar chart -> 'class_distribution.png'
    2. Text length histogram -> 'text_length_distribution.png'

    Args:
        df: DataFrame with 'label' and 'text_length' columns
        output_dir: Directory to save plots

    Hints:
        - Use matplotlib.pyplot
        - Create output_dir if it doesn't exist (os.makedirs)
        - Use plt.savefig() to save
    """
    # TODO: Implement
    pass


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs/data",
) -> None:
    """
    Task 6: Save the splits to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save CSV files

    Creates:
        - {output_dir}/train.csv
        - {output_dir}/val.csv
        - {output_dir}/test.csv
    """
    # TODO: Implement
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(
    dataset_name: str = "amazon_polarity",
    num_samples: int = 5000,
    output_dir: str = "outputs",
) -> dict:
    """
    Run the complete data pipeline.

    Returns:
        Dictionary with pipeline results
    """
    print("=" * 60)
    print("Week 2 Project: Sentiment Data Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n📥 Loading data...")
    df = load_data(dataset_name, num_samples)
    if df is None:
        print("❌ Failed to load data")
        return {"success": False, "error": "load_data returned None"}
    print(f"✅ Loaded {len(df)} samples")

    # 2. Explore data
    print("\n🔍 Exploring data...")
    stats = explore_data(df)
    if stats is None:
        print("❌ Failed to explore data")
        return {"success": False, "error": "explore_data returned None"}

    # 3. Clean data
    print("\n🧹 Cleaning data...")
    original_count = len(df)
    df = clean_data(df)
    if df is None:
        print("❌ Failed to clean data")
        return {"success": False, "error": "clean_data returned None"}
    print(f"✅ Cleaned: {original_count} → {len(df)} samples")

    # 4. Visualize
    print("\n📊 Creating visualizations...")
    visualize(df, output_dir)
    print(f"✅ Saved plots to {output_dir}/")

    # 5. Create splits
    print("\n✂️ Creating train/val/test splits...")
    splits = create_splits(df)
    if splits is None or len(splits) != 3:
        print("❌ Failed to create splits")
        return {"success": False, "error": "create_splits failed"}
    train_df, val_df, test_df = splits
    print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 6. Save splits
    print("\n💾 Saving splits to CSV...")
    data_dir = os.path.join(output_dir, "data")
    save_splits(train_df, val_df, test_df, data_dir)
    print(f"✅ Saved to {data_dir}/")

    print("\n" + "=" * 60)
    print("🎉 Pipeline complete!")
    print("=" * 60)

    return {
        "success": True,
        "stats": stats,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }


if __name__ == "__main__":
    result = run_pipeline()
    if not result["success"]:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")

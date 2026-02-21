"""
Week 2 Project: SOLUTION
========================

Complete solution for the Sentiment Data Pipeline project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_data(
    dataset_name: str = "amazon_polarity", num_samples: int = 5000
) -> pd.DataFrame:
    """Load dataset from HuggingFace."""

    # Load with sample limit
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    df = pd.DataFrame(dataset)

    # Rename columns for consistency (amazon_polarity uses 'content')
    if "content" in df.columns:
        df = df.rename(columns={"content": "text"})

    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Explore the dataset and return statistics."""
    stats = {
        "num_samples": len(df),
        "num_classes": df["label"].nunique(),
        "class_distribution": df["label"].value_counts(),
        "avg_text_length": df["text"].str.len().mean(),
    }

    # Print statistics
    print(f"📊 Dataset Statistics:")
    print(f"   Samples: {stats['num_samples']}")
    print(f"   Classes: {stats['num_classes']}")
    print(f"   Avg text length: {stats['avg_text_length']:.1f} chars")
    print(f"\n   Class Distribution:")
    for label, count in stats["class_distribution"].items():
        pct = count / stats["num_samples"] * 100
        print(f"   - Class {label}: {count} ({pct:.1f}%)")

    return stats


def clean_data(
    df: pd.DataFrame, min_length: int = 10, max_length: int = 1000
) -> pd.DataFrame:
    """Clean and preprocess the data."""
    # Work on a copy
    df = df.copy()

    # 1. Remove missing text
    df = df.dropna(subset=["text"])

    # 2. Strip whitespace
    df["text"] = df["text"].str.strip()

    # 3. Add text_length column
    df["text_length"] = df["text"].str.len()

    # 4. Filter by length
    df = df[(df["text_length"] >= min_length) & (df["text_length"] <= max_length)]

    # 5. Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Reset index
    df = df.reset_index(drop=True)

    return df


def create_splits(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with stratification."""

    # First split: separate test set
    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Second split: separate validation from training
    # Adjust val_size since we're splitting from train_val
    adjusted_val_size = val_size / (train_size + val_size)

    train_df, val_df = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val["label"],
    )

    return train_df, val_df, test_df


def visualize(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Create and save visualizations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Class distribution bar chart
    plt.figure(figsize=(8, 5))
    class_counts = df["label"].value_counts().sort_index()
    colors = ["#e74c3c", "#2ecc71"] if len(class_counts) == 2 else None
    class_counts.plot(kind="bar", color=colors)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=100)
    plt.close()

    # 2. Text length histogram
    plt.figure(figsize=(8, 5))
    df["text_length"].hist(bins=50, edgecolor="black", alpha=0.7)
    plt.title("Text Length Distribution")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.axvline(
        df["text_length"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['text_length'].mean():.0f}",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"), dpi=100)
    plt.close()


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs/data",
) -> None:
    """Save the splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


def run_pipeline(
    dataset_name: str = "amazon_polarity",
    num_samples: int = 5000,
    output_dir: str = "outputs",
) -> dict:
    """Run the complete data pipeline."""
    print("=" * 60)
    print("Week 2 Project: Sentiment Data Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n📥 Loading data...")
    df = load_data(dataset_name, num_samples)
    print(f"✅ Loaded {len(df)} samples")

    # 2. Explore data
    print("\n🔍 Exploring data...")
    stats = explore_data(df)

    # 3. Clean data
    print("\n🧹 Cleaning data...")
    original_count = len(df)
    df = clean_data(df)
    print(f"✅ Cleaned: {original_count} → {len(df)} samples")

    # 4. Visualize
    print("\n📊 Creating visualizations...")
    visualize(df, output_dir)
    print(f"✅ Saved plots to {output_dir}/")

    # 5. Create splits
    print("\n✂️ Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(df)
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
    run_pipeline()

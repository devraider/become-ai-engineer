"""
Week 2 Project: Tests for Sentiment Data Pipeline
==================================================

Run tests:
    python -m pytest tests/test_project.py -v
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from project_pipeline import (
    load_data,
    explore_data,
    clean_data,
    create_splits,
    visualize,
    save_splits,
)


# =============================================================================
# SAMPLE DATA FOR TESTS
# =============================================================================


def sample_df() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": [
                "This is a great product, I love it!",
                "Terrible experience, would not recommend",
                "It's okay, nothing special about it",
                "Best purchase I ever made, amazing quality",
                "Worst thing I've bought, complete waste",
                "Decent product for the price point",
                "Absolutely fantastic, exceeded expectations",
                "Very disappointed with this purchase",
                "Good value, works as expected",
                "Not worth the money at all",
            ]
            * 10,  # 100 samples
            "label": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0] * 10,
        }
    )


def sample_df_with_issues() -> pd.DataFrame:
    """Create DataFrame with data quality issues for cleaning tests."""
    return pd.DataFrame(
        {
            "text": [
                "Good product",
                "  Needs trimming  ",
                "Short",  # Too short
                "A" * 2000,  # Too long
                None,  # Missing
                "Good product",  # Duplicate
                "Another valid review here",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1],
        }
    )


# =============================================================================
# TESTS FOR load_data
# =============================================================================

# Check if HuggingFace datasets is available
try:
    from datasets import load_dataset as _hf_load

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TestLoadData:
    """Tests for load_data function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        if not HF_AVAILABLE:
            pytest.skip("HuggingFace datasets not installed")

        df = load_data("amazon_polarity", num_samples=100)
        assert df is not None, "Function returned None"
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    def test_has_required_columns(self):
        """Test that DataFrame has text and label columns."""
        if not HF_AVAILABLE:
            pytest.skip("HuggingFace datasets not installed")

        df = load_data("amazon_polarity", num_samples=100)
        if df is not None:
            assert "text" in df.columns, "Missing 'text' column"
            assert "label" in df.columns, "Missing 'label' column"

    def test_respects_num_samples(self):
        """Test that num_samples parameter works."""
        if not HF_AVAILABLE:
            pytest.skip("HuggingFace datasets not installed")

        df = load_data("amazon_polarity", num_samples=50)
        if df is not None:
            assert len(df) == 50, f"Expected 50 samples, got {len(df)}"


# =============================================================================
# TESTS FOR explore_data
# =============================================================================


class TestExploreData:
    """Tests for explore_data function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        df = sample_df()
        result = explore_data(df)
        assert result is not None, "Function returned None"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_has_required_keys(self):
        """Test that result has all required keys."""
        df = sample_df()
        result = explore_data(df)
        if result is not None:
            required_keys = [
                "num_samples",
                "num_classes",
                "class_distribution",
                "avg_text_length",
            ]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_num_samples_correct(self):
        """Test that num_samples is correct."""
        df = sample_df()
        result = explore_data(df)
        if result is not None:
            assert result["num_samples"] == len(df), "num_samples incorrect"

    def test_num_classes_correct(self):
        """Test that num_classes is correct."""
        df = sample_df()
        result = explore_data(df)
        if result is not None:
            assert result["num_classes"] == 2, "num_classes should be 2"


# =============================================================================
# TESTS FOR clean_data
# =============================================================================


class TestCleanData:
    """Tests for clean_data function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = sample_df()
        result = clean_data(df)
        assert result is not None, "Function returned None"
        assert isinstance(
            result, pd.DataFrame
        ), f"Expected DataFrame, got {type(result)}"

    def test_adds_text_length_column(self):
        """Test that text_length column is added."""
        df = sample_df()
        result = clean_data(df)
        if result is not None:
            assert "text_length" in result.columns, "Missing 'text_length' column"

    def test_removes_short_texts(self):
        """Test that short texts are removed."""
        df = sample_df_with_issues()
        result = clean_data(df, min_length=10)
        if result is not None:
            # "Short" has 5 chars, should be removed
            short_texts = result[result["text_length"] < 10]
            assert len(short_texts) == 0, "Short texts should be removed"

    def test_removes_long_texts(self):
        """Test that long texts are removed."""
        df = sample_df_with_issues()
        result = clean_data(df, max_length=1000)
        if result is not None:
            long_texts = result[result["text_length"] > 1000]
            assert len(long_texts) == 0, "Long texts should be removed"

    def test_removes_duplicates(self):
        """Test that duplicate texts are removed."""
        df = sample_df_with_issues()
        result = clean_data(df)
        if result is not None:
            # Check no duplicate texts
            assert result["text"].nunique() == len(
                result
            ), "Duplicates should be removed"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        df = pd.DataFrame({"text": ["  hello world  ", "  test  "], "label": [0, 1]})
        result = clean_data(df, min_length=1)
        if result is not None:
            for text in result["text"]:
                assert text == text.strip(), "Whitespace should be stripped"


# =============================================================================
# TESTS FOR create_splits
# =============================================================================


class TestCreateSplits:
    """Tests for create_splits function."""

    def test_returns_three_dataframes(self):
        """Test that function returns three DataFrames."""
        df = sample_df()
        result = create_splits(df)
        assert result is not None, "Function returned None"
        assert len(result) == 3, f"Expected 3 DataFrames, got {len(result)}"
        train_df, val_df, test_df = result
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_preserves_all_samples(self):
        """Test that all samples are preserved across splits."""
        df = sample_df()
        result = create_splits(df)
        if result is not None:
            train_df, val_df, test_df = result
            total = len(train_df) + len(val_df) + len(test_df)
            assert total == len(df), f"Expected {len(df)} total, got {total}"

    def test_split_ratios_approximate(self):
        """Test that split ratios are approximately correct."""
        df = sample_df()
        result = create_splits(df, train_size=0.8, val_size=0.1, test_size=0.1)
        if result is not None:
            train_df, val_df, test_df = result
            total = len(df)

            train_ratio = len(train_df) / total
            val_ratio = len(val_df) / total
            test_ratio = len(test_df) / total

            assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio} not ~0.8"
            assert 0.05 <= val_ratio <= 0.15, f"Val ratio {val_ratio} not ~0.1"
            assert 0.05 <= test_ratio <= 0.15, f"Test ratio {test_ratio} not ~0.1"

    def test_stratification(self):
        """Test that splits are stratified."""
        df = sample_df()
        result = create_splits(df)
        if result is not None:
            train_df, val_df, test_df = result

            # Check class ratios are similar
            orig_ratio = (df["label"] == 1).mean()
            train_ratio = (train_df["label"] == 1).mean()
            test_ratio = (test_df["label"] == 1).mean()

            assert abs(orig_ratio - train_ratio) < 0.1, "Train not stratified"
            assert abs(orig_ratio - test_ratio) < 0.1, "Test not stratified"


# =============================================================================
# TESTS FOR visualize
# =============================================================================


class TestVisualize:
    """Tests for visualize function."""

    def test_creates_output_directory(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = sample_df()
            df["text_length"] = df["text"].str.len()

            output_dir = os.path.join(tmpdir, "new_dir")
            visualize(df, output_dir)

            assert os.path.exists(output_dir), "Output directory should be created"

    def test_creates_class_distribution_plot(self):
        """Test that class distribution plot is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = sample_df()
            df["text_length"] = df["text"].str.len()

            visualize(df, tmpdir)

            plot_path = os.path.join(tmpdir, "class_distribution.png")
            assert os.path.exists(plot_path), "class_distribution.png should be created"

    def test_creates_length_distribution_plot(self):
        """Test that text length distribution plot is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = sample_df()
            df["text_length"] = df["text"].str.len()

            visualize(df, tmpdir)

            plot_path = os.path.join(tmpdir, "text_length_distribution.png")
            assert os.path.exists(
                plot_path
            ), "text_length_distribution.png should be created"


# =============================================================================
# TESTS FOR save_splits
# =============================================================================


class TestSaveSplits:
    """Tests for save_splits function."""

    def test_creates_csv_files(self):
        """Test that CSV files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = sample_df()
            train_df = df.iloc[:60]
            val_df = df.iloc[60:80]
            test_df = df.iloc[80:]

            save_splits(train_df, val_df, test_df, tmpdir)

            assert os.path.exists(
                os.path.join(tmpdir, "train.csv")
            ), "train.csv missing"
            assert os.path.exists(os.path.join(tmpdir, "val.csv")), "val.csv missing"
            assert os.path.exists(os.path.join(tmpdir, "test.csv")), "test.csv missing"

    def test_csv_contents_correct(self):
        """Test that CSV contents are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = sample_df()
            train_df = df.iloc[:60]
            val_df = df.iloc[60:80]
            test_df = df.iloc[80:]

            save_splits(train_df, val_df, test_df, tmpdir)

            # Read back and verify
            loaded_train = pd.read_csv(os.path.join(tmpdir, "train.csv"))
            assert len(loaded_train) == len(
                train_df
            ), "train.csv has wrong number of rows"

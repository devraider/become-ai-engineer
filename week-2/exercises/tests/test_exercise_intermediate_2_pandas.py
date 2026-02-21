"""
Week 2 - Tests for Exercise 2: Pandas Data Preparation
=======================================================

Run all tests:
    python -m pytest exercises/tests/ -v

Run only exercise 2 tests:
    python -m pytest exercises/tests/test_exercise_2.py -v

Note: Some tests require HuggingFace datasets library.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import exercises
sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_intermediate_2_pandas import (
    load_emotion_dataset,
    check_class_distribution,
    add_text_length,
    average_length_per_class,
    create_train_test_split,
    clean_text_data,
    filter_by_length,
)


# =============================================================================
# SAMPLE DATA FOR TESTS
# =============================================================================


def sample_df() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
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


# =============================================================================
# TESTS FOR check_class_distribution
# =============================================================================


class TestCheckClassDistribution:
    """Tests for Task 2: check_class_distribution"""

    def test_returns_series(self):
        """Test that class distribution returns a Series."""
        df = sample_df()
        dist = check_class_distribution(df)
        assert dist is not None, "Function returned None"
        assert isinstance(dist, pd.Series), f"Expected Series, got {type(dist)}"

    def test_counts_correct(self):
        """Test that counts are correct."""
        df = sample_df()
        dist = check_class_distribution(df)
        assert dist.sum() == len(df), "Total count should equal DataFrame length"

    def test_with_imbalanced_data(self):
        """Test with imbalanced class distribution."""
        df = pd.DataFrame(
            {
                "text": ["a"] * 10 + ["b"] * 3 + ["c"] * 1,
                "label": [0] * 10 + [1] * 3 + [2] * 1,
            }
        )
        dist = check_class_distribution(df)
        assert dist[0] == 10
        assert dist[1] == 3
        assert dist[2] == 1


# =============================================================================
# TESTS FOR add_text_length
# =============================================================================


class TestAddTextLength:
    """Tests for Task 3: add_text_length"""

    def test_creates_column(self):
        """Test that text_length column is created."""
        df = sample_df()
        result = add_text_length(df)
        assert result is not None, "Function returned None"
        assert "text_length" in result.columns, "text_length column not found"

    def test_values_correct(self):
        """Test that text length values are correct."""
        df = pd.DataFrame({"text": ["hello", "hi", "goodbye"]})
        result = add_text_length(df)
        expected = [5, 2, 7]
        assert list(result["text_length"]) == expected

    def test_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = sample_df()
        original_cols = df.columns.tolist()
        _ = add_text_length(df)
        assert (
            df.columns.tolist() == original_cols
        ), "Original DataFrame should not be modified"

    def test_empty_strings(self):
        """Test handling of empty strings."""
        df = pd.DataFrame({"text": ["", "hello", ""]})
        result = add_text_length(df)
        assert list(result["text_length"]) == [0, 5, 0]


# =============================================================================
# TESTS FOR average_length_per_class
# =============================================================================


class TestAverageLengthPerClass:
    """Tests for Task 4: average_length_per_class"""

    def test_calculates_average(self):
        """Test average length calculation."""
        df = pd.DataFrame({"label": [0, 0, 1, 1], "text_length": [10, 20, 30, 40]})
        avg = average_length_per_class(df)
        assert avg is not None, "Function returned None"
        assert avg[0] == 15.0, f"Expected 15.0 for label 0, got {avg[0]}"
        assert avg[1] == 35.0, f"Expected 35.0 for label 1, got {avg[1]}"

    def test_returns_series(self):
        """Test that result is a Series."""
        df = pd.DataFrame({"label": [0, 1, 2], "text_length": [10, 20, 30]})
        avg = average_length_per_class(df)
        assert isinstance(avg, pd.Series), f"Expected Series, got {type(avg)}"

    def test_single_sample_per_class(self):
        """Test with single sample per class."""
        df = pd.DataFrame({"label": [0, 1, 2], "text_length": [100, 200, 300]})
        avg = average_length_per_class(df)
        assert avg[0] == 100
        assert avg[1] == 200
        assert avg[2] == 300


# =============================================================================
# TESTS FOR create_train_test_split
# =============================================================================


class TestCreateTrainTestSplit:
    """Tests for Task 5: create_train_test_split"""

    def test_split_sizes(self):
        """Test that split sizes are approximately correct."""
        df = sample_df()
        # Add more samples for meaningful split
        df = pd.concat([df] * 10, ignore_index=True)
        train_df, test_df = create_train_test_split(df, test_size=0.2)

        assert train_df is not None and test_df is not None, "Function returned None"
        total = len(train_df) + len(test_df)
        assert total == len(df), "Split should preserve all rows"

        test_ratio = len(test_df) / total
        assert (
            0.15 <= test_ratio <= 0.25
        ), f"Test ratio should be ~0.2, got {test_ratio}"

    def test_stratification(self):
        """Test that stratification preserves class distribution."""
        df = pd.DataFrame(
            {"text": ["a"] * 80 + ["b"] * 20, "label": [0] * 80 + [1] * 20}
        )
        train_df, test_df = create_train_test_split(df, test_size=0.2)

        # Check that class ratios are similar
        train_ratio = (train_df["label"] == 0).mean()
        test_ratio = (test_df["label"] == 0).mean()
        assert (
            abs(train_ratio - test_ratio) < 0.1
        ), "Stratification should preserve class ratios"

    def test_reproducibility(self):
        """Test that same random_state gives same split."""
        df = pd.concat([sample_df()] * 10, ignore_index=True)
        train1, test1 = create_train_test_split(df, test_size=0.2, random_state=42)
        train2, test2 = create_train_test_split(df, test_size=0.2, random_state=42)

        assert train1.equals(train2), "Same random_state should give same train split"
        assert test1.equals(test2), "Same random_state should give same test split"

    def test_returns_dataframes(self):
        """Test that function returns DataFrames."""
        df = pd.concat([sample_df()] * 10, ignore_index=True)
        train_df, test_df = create_train_test_split(df)

        assert isinstance(
            train_df, pd.DataFrame
        ), f"Expected DataFrame, got {type(train_df)}"
        assert isinstance(
            test_df, pd.DataFrame
        ), f"Expected DataFrame, got {type(test_df)}"


# =============================================================================
# TESTS FOR clean_text_data (Bonus)
# =============================================================================


class TestCleanTextData:
    """Tests for Task 6 (Bonus): clean_text_data"""

    def test_lowercase(self):
        """Test that text is converted to lowercase."""
        df = pd.DataFrame({"text": ["HELLO", "World", "TEST"]})
        result = clean_text_data(df)
        if result is not None:
            assert all(
                result["text"] == result["text"].str.lower()
            ), "Text should be lowercase"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        df = pd.DataFrame({"text": ["  hello  ", "world  ", "  test"]})
        result = clean_text_data(df)
        if result is not None:
            expected = ["hello", "world", "test"]
            assert list(result["text"]) == expected

    def test_removes_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame({"text": ["hello", "hello", "world"], "label": [0, 1, 2]})
        result = clean_text_data(df)
        if result is not None:
            # Should have 2 unique texts
            assert len(result) == 2 or result["text"].nunique() == 2

    def test_preserves_original(self):
        """Test original DataFrame is not modified."""
        df = pd.DataFrame({"text": ["  HELLO  "], "label": [0]})
        original_text = df["text"].iloc[0]
        _ = clean_text_data(df)
        assert df["text"].iloc[0] == original_text, "Original should not be modified"


# =============================================================================
# TESTS FOR filter_by_length (Bonus)
# =============================================================================


class TestFilterByLength:
    """Tests for Task 7 (Bonus): filter_by_length"""

    def test_filters_correctly(self):
        """Test filtering by length."""
        df = pd.DataFrame(
            {
                "text": [
                    "short",
                    "medium length text",
                    "this is a very long text indeed",
                ],
                "text_length": [5, 18, 32],
            }
        )
        result = filter_by_length(df, min_length=10, max_length=20)
        if result is not None:
            assert len(result) == 1, f"Expected 1 row, got {len(result)}"
            assert result.iloc[0]["text"] == "medium length text"

    def test_inclusive_bounds(self):
        """Test that bounds are inclusive."""
        df = pd.DataFrame({"text": ["a", "b", "c"], "text_length": [10, 15, 20]})
        result = filter_by_length(df, min_length=10, max_length=20)
        if result is not None:
            assert len(result) == 3, "All rows should match inclusive bounds"

    def test_no_matches(self):
        """Test when no rows match the filter."""
        df = pd.DataFrame({"text": ["short", "also short"], "text_length": [5, 10]})
        result = filter_by_length(df, min_length=100, max_length=200)
        if result is not None:
            assert len(result) == 0, "Should return empty DataFrame"


# =============================================================================
# TESTS REQUIRING HUGGINGFACE (Optional)
# =============================================================================

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TestLoadEmotionDataset:
    """Tests for Task 1: load_emotion_dataset (requires HuggingFace)"""

    def test_loads_dataset(self):
        """Test loading the emotion dataset."""
        if not HF_AVAILABLE:
            import pytest

            pytest.skip("HuggingFace datasets not installed")

        df = load_emotion_dataset()
        assert df is not None, "Function returned None"
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    def test_has_required_columns(self):
        """Test that DataFrame has required columns."""
        if not HF_AVAILABLE:
            import pytest

            pytest.skip("HuggingFace datasets not installed")

        df = load_emotion_dataset()
        assert "text" in df.columns, "DataFrame should have 'text' column"
        assert "label" in df.columns, "DataFrame should have 'label' column"

    def test_has_many_samples(self):
        """Test that dataset has many samples."""
        if not HF_AVAILABLE:
            import pytest

            pytest.skip("HuggingFace datasets not installed")

        df = load_emotion_dataset()
        assert len(df) > 1000, "Dataset should have many samples"

    def test_has_6_classes(self):
        """Test that emotion dataset has 6 emotion classes."""
        if not HF_AVAILABLE:
            import pytest

            pytest.skip("HuggingFace datasets not installed")

        df = load_emotion_dataset()
        unique_labels = df["label"].nunique()
        assert unique_labels == 6, f"Expected 6 emotion classes, got {unique_labels}"

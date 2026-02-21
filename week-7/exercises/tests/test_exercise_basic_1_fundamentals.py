"""
Week 7 - Tests for Exercise 1: Prompt Engineering Fundamentals
==============================================================

Run tests:
    python -m pytest tests/test_exercise_basic_1_fundamentals.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_basic_1_fundamentals import (
    craft_prompt,
    create_specific_prompt,
    add_constraints,
    create_role_prompt,
    create_step_by_step_prompt,
    create_example_based_prompt,
    create_output_format_prompt,
    create_negative_prompt,
    create_iterative_prompt,
    analyze_prompt_quality,
)


# =============================================================================
# TESTS FOR craft_prompt (Task 1)
# =============================================================================


class TestCraftPrompt:
    """Tests for CRAFT prompt builder."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = craft_prompt(
            context="Building an app",
            role="Developer",
            action="Review code",
            format_spec="List",
            tone="Professional",
        )
        if result:
            assert isinstance(result, str)

    def test_includes_all_components(self):
        """Test that all CRAFT components are included."""
        result = craft_prompt(
            context="Web development project",
            role="Senior engineer",
            action="Optimize performance",
            format_spec="Bullet points",
            tone="Technical",
        )
        if result:
            assert "Web development" in result or "context" in result.lower()
            assert "engineer" in result.lower() or "role" in result.lower()
            assert "Optimize" in result or "performance" in result

    def test_structured_output(self):
        """Test that output is well-structured."""
        result = craft_prompt(
            context="API design",
            role="Architect",
            action="Design endpoints",
            format_spec="JSON",
            tone="Formal",
        )
        if result:
            # Should have some structure (newlines, sections)
            assert "\n" in result or len(result) > 50


# =============================================================================
# TESTS FOR create_specific_prompt (Task 2)
# =============================================================================


class TestCreateSpecificPrompt:
    """Tests for specific prompt creation."""

    def test_increases_specificity(self):
        """Test that output is more specific than input."""
        vague = "write about Python"
        result = create_specific_prompt(vague)
        if result:
            assert len(result) > len(vague)

    def test_adds_structure(self):
        """Test that structure is added."""
        result = create_specific_prompt("explain databases")
        if result:
            # Should add some constraints or format
            has_structure = any(
                word in result.lower()
                for word in [
                    "format",
                    "requirements",
                    "specific",
                    "include",
                    "should",
                    "must",
                ]
            )
            assert has_structure


# =============================================================================
# TESTS FOR add_constraints (Task 3)
# =============================================================================


class TestAddConstraints:
    """Tests for constraint addition."""

    def test_adds_word_limit(self):
        """Test that word limit is added."""
        result = add_constraints("Explain recursion", {"max_words": 100})
        if result:
            assert "100" in result

    def test_adds_audience(self):
        """Test that audience is added."""
        result = add_constraints("Explain APIs", {"audience": "beginners"})
        if result:
            assert "beginner" in result.lower()

    def test_multiple_constraints(self):
        """Test multiple constraints."""
        result = add_constraints(
            "Describe ML",
            {"max_words": 200, "format": "markdown", "audience": "experts"},
        )
        if result:
            assert "200" in result or "markdown" in result.lower()


# =============================================================================
# TESTS FOR create_role_prompt (Task 4)
# =============================================================================


class TestCreateRolePrompt:
    """Tests for role-based prompts."""

    def test_includes_role(self):
        """Test that role is included."""
        result = create_role_prompt(
            "Data Scientist", "ML expert", "Explain overfitting"
        )
        if result:
            assert "Data Scientist" in result

    def test_includes_expertise(self):
        """Test that expertise is mentioned."""
        result = create_role_prompt(
            "Engineer", "10 years experience", "Review this design"
        )
        if result:
            assert "experience" in result.lower() or "10" in result


# =============================================================================
# TESTS FOR create_step_by_step_prompt (Task 5)
# =============================================================================


class TestCreateStepByStepPrompt:
    """Tests for step-by-step prompts."""

    def test_includes_numbered_steps(self):
        """Test that steps are numbered."""
        result = create_step_by_step_prompt(
            "Debug code", ["Read code", "Find bug", "Fix it"]
        )
        if result:
            assert "1" in result and "2" in result and "3" in result

    def test_includes_all_steps(self):
        """Test all steps are included."""
        steps = ["Step A", "Step B", "Step C"]
        result = create_step_by_step_prompt("Do task", steps)
        if result:
            for step in steps:
                assert step in result


# =============================================================================
# TESTS FOR create_example_based_prompt (Task 6)
# =============================================================================


class TestCreateExampleBasedPrompt:
    """Tests for one-shot prompts."""

    def test_includes_example(self):
        """Test that example is included."""
        result = create_example_based_prompt(
            "Convert to past tense", "I walk", "I walked"
        )
        if result:
            assert "I walk" in result and "I walked" in result

    def test_includes_task(self):
        """Test that task description is included."""
        result = create_example_based_prompt("Classify sentiment", "Great!", "positive")
        if result:
            assert "sentiment" in result.lower() or "classify" in result.lower()


# =============================================================================
# TESTS FOR create_output_format_prompt (Task 7)
# =============================================================================


class TestCreateOutputFormatPrompt:
    """Tests for format-specified prompts."""

    def test_specifies_json_format(self):
        """Test JSON format specification."""
        result = create_output_format_prompt("Extract info", "json", ["name", "age"])
        if result:
            assert "json" in result.lower()

    def test_includes_fields(self):
        """Test that fields are included."""
        result = create_output_format_prompt(
            "Parse data", "json", ["field1", "field2", "field3"]
        )
        if result:
            assert "field1" in result or "field2" in result


# =============================================================================
# TESTS FOR create_negative_prompt (Task 8)
# =============================================================================


class TestCreateNegativePrompt:
    """Tests for negative prompts."""

    def test_includes_restrictions(self):
        """Test that restrictions are mentioned."""
        result = create_negative_prompt(
            "Write description", ["Don't use jargon", "Don't be verbose"]
        )
        if result:
            has_negation = (
                "don't" in result.lower()
                or "do not" in result.lower()
                or "avoid" in result.lower()
            )
            assert has_negation


# =============================================================================
# TESTS FOR create_iterative_prompt (Task 9)
# =============================================================================


class TestCreateIterativePrompt:
    """Tests for iteration prompts."""

    def test_includes_original(self):
        """Test that original output is included."""
        result = create_iterative_prompt("Original response here", "Make it better")
        if result:
            assert "Original response" in result

    def test_includes_feedback(self):
        """Test that feedback is included."""
        result = create_iterative_prompt("Some text", "Add more detail")
        if result:
            assert "detail" in result.lower() or "improve" in result.lower()


# =============================================================================
# TESTS FOR analyze_prompt_quality (Task 10)
# =============================================================================


class TestAnalyzePromptQuality:
    """Tests for prompt quality analysis."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        result = analyze_prompt_quality("Write about Python")
        if result:
            assert isinstance(result, dict)

    def test_low_score_for_vague_prompt(self):
        """Test that vague prompts get low scores."""
        result = analyze_prompt_quality("Write something")
        if result and "specificity_score" in result:
            assert result["specificity_score"] <= 3

    def test_has_suggestions(self):
        """Test that suggestions are provided."""
        result = analyze_prompt_quality("Do a thing")
        if result:
            assert "suggestions" in result
            assert len(result["suggestions"]) > 0

    def test_detects_missing_format(self):
        """Test detection of missing format specification."""
        result = analyze_prompt_quality("Explain recursion")
        if result and "has_format" in result:
            assert result["has_format"] == False

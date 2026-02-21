"""
Tests for Week 5 Exercise 1: Prompt Engineering
"""

import pytest
from exercise_basic_1_prompts import (
    basic_prompt,
    role_based_prompt,
    few_shot_prompt,
    structured_output_prompt,
    chain_of_thought_prompt,
    constrained_prompt,
    code_generation_prompt,
    comparison_prompt,
    refinement_prompt,
    safety_prompt,
)


class TestBasicPrompt:
    def test_returns_string(self):
        result = basic_prompt("machine learning")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_topic(self):
        result = basic_prompt("neural networks")
        assert "neural networks" in result.lower() or "neural" in result.lower()

    def test_has_constraints(self):
        result = basic_prompt("APIs")
        # Should mention word limit or be beginner-friendly
        assert any(
            word in result.lower()
            for word in ["word", "beginner", "simple", "explain", "example"]
        )


class TestRoleBasedPrompt:
    def test_returns_string(self):
        result = role_based_prompt("Python expert", "explain decorators")
        assert isinstance(result, str)

    def test_contains_role(self):
        result = role_based_prompt("data scientist", "analyze data")
        assert "data scientist" in result.lower()

    def test_contains_task(self):
        result = role_based_prompt("engineer", "write code")
        assert "write code" in result.lower() or "code" in result.lower()


class TestFewShotPrompt:
    def test_includes_examples(self):
        examples = [
            {"input": "happy", "output": "POSITIVE"},
            {"input": "sad", "output": "NEGATIVE"},
        ]
        result = few_shot_prompt(examples, "excited")

        assert "happy" in result
        assert "POSITIVE" in result
        assert "sad" in result
        assert "NEGATIVE" in result

    def test_includes_new_input(self):
        examples = [{"input": "test", "output": "TEST"}]
        result = few_shot_prompt(examples, "new_input")
        assert "new_input" in result

    def test_format_structure(self):
        examples = [{"input": "a", "output": "A"}]
        result = few_shot_prompt(examples, "b")
        # Should have some structured format
        assert "input" in result.lower() or ":" in result


class TestStructuredOutputPrompt:
    def test_requests_json(self):
        result = structured_output_prompt("Some data", ["field1", "field2"])
        assert "json" in result.lower()

    def test_includes_fields(self):
        result = structured_output_prompt("Data", ["name", "date", "amount"])
        assert "name" in result
        assert "date" in result
        assert "amount" in result

    def test_includes_data(self):
        result = structured_output_prompt("Meeting on Monday", ["date"])
        assert "Meeting" in result or "Monday" in result


class TestChainOfThoughtPrompt:
    def test_encourages_reasoning(self):
        result = chain_of_thought_prompt("What is 2+2?")
        # Should mention step-by-step or reasoning
        assert any(
            phrase in result.lower()
            for phrase in ["step", "think", "reason", "explain"]
        )

    def test_includes_problem(self):
        problem = "Calculate the area of a circle"
        result = chain_of_thought_prompt(problem)
        assert "circle" in result.lower() or "area" in result.lower()


class TestConstrainedPrompt:
    def test_includes_constraints(self):
        constraints = {"max_words": "50", "tone": "professional"}
        result = constrained_prompt("Write a summary", constraints)
        assert "50" in result or "word" in result.lower()
        assert "professional" in result.lower()

    def test_includes_task(self):
        result = constrained_prompt("Translate this", {"language": "French"})
        assert "translate" in result.lower() or "French" in result


class TestCodeGenerationPrompt:
    def test_includes_language(self):
        result = code_generation_prompt("Python", "sort a list", ["be efficient"])
        assert "python" in result.lower()

    def test_includes_requirements(self):
        reqs = ["handle errors", "be fast"]
        result = code_generation_prompt("JavaScript", "fetch data", reqs)
        assert "error" in result.lower() or "handle" in result.lower()

    def test_includes_task(self):
        result = code_generation_prompt("Python", "read CSV file", [])
        assert "csv" in result.lower() or "read" in result.lower()


class TestComparisonPrompt:
    def test_includes_items(self):
        result = comparison_prompt(["Python", "JavaScript"], ["speed"])
        assert "python" in result.lower()
        assert "javascript" in result.lower()

    def test_includes_criteria(self):
        result = comparison_prompt(["A", "B"], ["cost", "quality"])
        assert "cost" in result.lower()
        assert "quality" in result.lower()


class TestRefinementPrompt:
    def test_includes_original(self):
        original = "This is original text"
        result = refinement_prompt(original, "make it shorter")
        assert "original" in result.lower() or "This is" in result

    def test_includes_feedback(self):
        result = refinement_prompt("Some text", "improve clarity")
        assert "clarity" in result.lower() or "improve" in result.lower()


class TestSafetyPrompt:
    def test_includes_user_input(self):
        result = safety_prompt("What is Python?")
        assert "python" in result.lower()

    def test_has_safety_instructions(self):
        result = safety_prompt("Tell me about hacking")
        # Should have some safety boundaries
        assert len(result) > len("Tell me about hacking")  # Added context

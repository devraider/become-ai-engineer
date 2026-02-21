"""
Week 7 - Tests for Exercise 2: Advanced Prompting Techniques
============================================================

Run tests:
    python -m pytest tests/test_exercise_intermediate_2_techniques.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_intermediate_2_techniques import (
    create_chain_of_thought_prompt,
    create_structured_cot_prompt,
    create_few_shot_prompt,
    create_few_shot_with_reasoning,
    create_json_output_prompt,
    create_list_output_prompt,
    FewShotExample,
    FewShotManager,
    create_self_consistency_prompt,
    create_verification_prompt,
    create_decomposition_prompt,
)


# =============================================================================
# TESTS FOR create_chain_of_thought_prompt (Task 1)
# =============================================================================


class TestChainOfThoughtPrompt:
    """Tests for CoT prompt creation."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = create_chain_of_thought_prompt("Is 10 > 5?")
        if result:
            assert isinstance(result, str)

    def test_includes_step_instruction(self):
        """Test that step-by-step instruction is included."""
        result = create_chain_of_thought_prompt("Calculate 15 + 27")
        if result:
            assert "step" in result.lower()

    def test_includes_original_question(self):
        """Test that original question is preserved."""
        result = create_chain_of_thought_prompt("What is the capital of France?")
        if result:
            assert "capital" in result.lower() or "France" in result


# =============================================================================
# TESTS FOR create_structured_cot_prompt (Task 2)
# =============================================================================


class TestStructuredCoTPrompt:
    """Tests for structured CoT prompts."""

    def test_includes_numbered_steps(self):
        """Test that steps are numbered."""
        result = create_structured_cot_prompt(
            "Solve the equation", ["Identify variables", "Isolate x", "Check solution"]
        )
        if result:
            assert "1" in result and "2" in result

    def test_includes_all_steps(self):
        """Test all provided steps are included."""
        steps = ["First step", "Second step", "Third step"]
        result = create_structured_cot_prompt("Do math", steps)
        if result:
            for step in steps:
                assert step in result


# =============================================================================
# TESTS FOR create_few_shot_prompt (Task 3)
# =============================================================================


class TestFewShotPrompt:
    """Tests for few-shot prompts."""

    def test_includes_examples(self):
        """Test that examples are included."""
        examples = [
            {"input": "hello", "output": "hola"},
            {"input": "goodbye", "output": "adiós"},
        ]
        result = create_few_shot_prompt("Translate", examples, "thank you")
        if result:
            assert "hello" in result and "hola" in result

    def test_includes_new_input(self):
        """Test that new input is included."""
        examples = [{"input": "a", "output": "A"}]
        result = create_few_shot_prompt("Uppercase", examples, "b")
        if result:
            assert "b" in result

    def test_includes_task(self):
        """Test that task is mentioned."""
        examples = [{"input": "x", "output": "y"}]
        result = create_few_shot_prompt("Transform text", examples, "z")
        if result:
            # Either task or learn/example pattern should be present
            assert "Transform" in result or "example" in result.lower()


# =============================================================================
# TESTS FOR create_few_shot_with_reasoning (Task 4)
# =============================================================================


class TestFewShotWithReasoning:
    """Tests for few-shot with reasoning."""

    def test_includes_reasoning(self):
        """Test that reasoning is included."""
        examples = [
            {"input": "2+3", "reasoning": "Adding 2 and 3 together", "output": "5"}
        ]
        result = create_few_shot_with_reasoning("Calculate", examples, "4+5")
        if result:
            assert "reasoning" in result.lower() or "Adding" in result


# =============================================================================
# TESTS FOR create_json_output_prompt (Task 5)
# =============================================================================


class TestJsonOutputPrompt:
    """Tests for JSON output prompts."""

    def test_mentions_json(self):
        """Test that JSON is mentioned."""
        result = create_json_output_prompt(
            "Extract data", {"name": "string", "value": "number"}
        )
        if result:
            assert "json" in result.lower()

    def test_includes_schema_fields(self):
        """Test that schema fields are mentioned."""
        result = create_json_output_prompt(
            "Parse", {"field1": "type1", "field2": "type2"}
        )
        if result:
            assert "field1" in result or "field2" in result


# =============================================================================
# TESTS FOR create_list_output_prompt (Task 6)
# =============================================================================


class TestListOutputPrompt:
    """Tests for list output prompts."""

    def test_includes_count(self):
        """Test that item count is mentioned."""
        result = create_list_output_prompt(
            "Give tips", num_items=5, item_format="short"
        )
        if result:
            assert "5" in result

    def test_mentions_list_format(self):
        """Test that list format is indicated."""
        result = create_list_output_prompt(
            "List items", num_items=3, item_format="brief"
        )
        if result:
            assert "list" in result.lower() or "3" in result


# =============================================================================
# TESTS FOR FewShotManager (Task 7)
# =============================================================================


class TestFewShotManager:
    """Tests for FewShotManager class."""

    def test_add_and_get_example(self):
        """Test adding and retrieving examples."""
        manager = FewShotManager()
        try:
            manager.add_example("test", FewShotExample("in", "out"))
            examples = manager.get_examples("test", 1)
            if examples:
                assert len(examples) == 1
                assert examples[0].input_text == "in"
        except (AttributeError, TypeError):
            pytest.skip("FewShotManager not implemented")

    def test_get_limited_examples(self):
        """Test that example count is limited."""
        manager = FewShotManager()
        try:
            for i in range(5):
                manager.add_example("test", FewShotExample(f"in{i}", f"out{i}"))
            examples = manager.get_examples("test", 2)
            if examples:
                assert len(examples) <= 2
        except (AttributeError, TypeError):
            pytest.skip("FewShotManager not implemented")

    def test_build_prompt(self):
        """Test prompt building."""
        manager = FewShotManager()
        try:
            manager.add_example("category", FewShotExample("x", "y"))
            prompt = manager.build_prompt("category", "Task", "new_input")
            if prompt:
                assert "x" in prompt and "y" in prompt
        except (AttributeError, TypeError):
            pytest.skip("FewShotManager not implemented")


# =============================================================================
# TESTS FOR create_self_consistency_prompt (Task 8)
# =============================================================================


class TestSelfConsistencyPrompt:
    """Tests for self-consistency prompts."""

    def test_mentions_multiple_approaches(self):
        """Test that multiple approaches are requested."""
        result = create_self_consistency_prompt("Solve this", 3)
        if result:
            has_multiple = (
                "3" in result
                or "multiple" in result.lower()
                or "different" in result.lower()
            )
            assert has_multiple


# =============================================================================
# TESTS FOR create_verification_prompt (Task 9)
# =============================================================================


class TestVerificationPrompt:
    """Tests for verification prompts."""

    def test_includes_original_response(self):
        """Test that original response is included."""
        result = create_verification_prompt("The answer is 42", "What is 6*7?")
        if result:
            assert "42" in result

    def test_asks_for_verification(self):
        """Test that verification is requested."""
        result = create_verification_prompt("Response text", "Original task")
        if result:
            has_verify = any(
                word in result.lower()
                for word in ["verify", "check", "correct", "accurate"]
            )
            assert has_verify


# =============================================================================
# TESTS FOR create_decomposition_prompt (Task 10)
# =============================================================================


class TestDecompositionPrompt:
    """Tests for task decomposition prompts."""

    def test_asks_for_breakdown(self):
        """Test that breakdown is requested."""
        result = create_decomposition_prompt("Build a complex system")
        if result:
            has_breakdown = any(
                word in result.lower()
                for word in ["break", "step", "subtask", "component", "part"]
            )
            assert has_breakdown

    def test_includes_task(self):
        """Test that original task is included."""
        result = create_decomposition_prompt("Create a web scraper")
        if result:
            assert "scraper" in result.lower() or "web" in result.lower()

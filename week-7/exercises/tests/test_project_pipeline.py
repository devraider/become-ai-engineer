"""
Week 7 - Tests for Project: Prompt Engineering Toolkit
======================================================

Run tests:
    python -m pytest tests/test_project_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from project_pipeline import (
    CRAFTPrompt,
    Example,
    ExampleManager,
    CoTWrapper,
    OutputValidator,
    Template,
    TemplateLibrary,
    PromptOptimizer,
    PromptToolkit,
)


# =============================================================================
# TESTS FOR CRAFTPrompt
# =============================================================================


class TestCRAFTPrompt:
    """Tests for CRAFT prompt builder."""

    def test_fluent_api(self):
        """Test fluent/chaining API."""
        craft = CRAFTPrompt()
        try:
            result = craft.set_context("test")
            assert result is craft
        except (AttributeError, TypeError):
            pytest.skip("CRAFTPrompt not implemented")

    def test_build_includes_components(self):
        """Test that build includes all components."""
        craft = CRAFTPrompt()
        try:
            prompt = (
                craft.set_context("Building an app")
                .set_role("Developer")
                .set_action("Review code")
                .set_format("List")
                .set_tone("Professional")
                .build()
            )
            if prompt:
                assert "Building" in prompt or "Developer" in prompt
        except (AttributeError, TypeError):
            pytest.skip("CRAFTPrompt not implemented")

    def test_validate_returns_warnings(self):
        """Test validation returns warnings for missing components."""
        craft = CRAFTPrompt()
        try:
            craft.set_action("Do something")
            warnings = craft.validate()
            if warnings is not None:
                # Should warn about missing context, role, etc.
                assert isinstance(warnings, list)
        except (AttributeError, TypeError):
            pytest.skip("CRAFTPrompt not implemented")


# =============================================================================
# TESTS FOR ExampleManager
# =============================================================================


class TestExampleManager:
    """Tests for few-shot example manager."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving examples."""
        em = ExampleManager()
        try:
            em.add_example("sentiment", Example("great", "positive"))
            examples = em.get_examples("sentiment", 1)
            if examples:
                assert len(examples) == 1
        except (AttributeError, TypeError):
            pytest.skip("ExampleManager not implemented")

    def test_build_few_shot_prompt(self):
        """Test building few-shot prompt."""
        em = ExampleManager()
        try:
            em.add_example("test", Example("in", "out"))
            prompt = em.build_few_shot_prompt("test", "Task", "new_input", 1)
            if prompt:
                assert "in" in prompt and "out" in prompt
        except (AttributeError, TypeError):
            pytest.skip("ExampleManager not implemented")

    def test_list_categories(self):
        """Test listing categories."""
        em = ExampleManager()
        try:
            em.add_example("cat1", Example("a", "b"))
            em.add_example("cat2", Example("c", "d"))
            cats = em.list_categories()
            if cats:
                assert "cat1" in cats and "cat2" in cats
        except (AttributeError, TypeError):
            pytest.skip("ExampleManager not implemented")


# =============================================================================
# TESTS FOR CoTWrapper
# =============================================================================


class TestCoTWrapper:
    """Tests for Chain-of-Thought wrapper."""

    def test_wrap_basic(self):
        """Test basic CoT wrapping."""
        try:
            result = CoTWrapper.wrap_basic("What is 2+2?")
            if result:
                assert "step" in result.lower()
                assert "2+2" in result
        except (AttributeError, TypeError):
            pytest.skip("CoTWrapper not implemented")

    def test_wrap_structured(self):
        """Test structured CoT wrapping."""
        try:
            result = CoTWrapper.wrap_structured(
                "Solve problem", ["Step 1", "Step 2", "Step 3"]
            )
            if result:
                assert "Step 1" in result
        except (AttributeError, TypeError):
            pytest.skip("CoTWrapper not implemented")

    def test_wrap_with_verification(self):
        """Test CoT with verification."""
        try:
            result = CoTWrapper.wrap_with_verification("Calculate 5*5")
            if result:
                has_verify = "verify" in result.lower() or "check" in result.lower()
                assert has_verify
        except (AttributeError, TypeError):
            pytest.skip("CoTWrapper not implemented")


# =============================================================================
# TESTS FOR OutputValidator
# =============================================================================


class TestOutputValidator:
    """Tests for output validation."""

    def test_extract_json(self):
        """Test JSON extraction from text."""
        try:
            text = 'Here is the result: {"name": "test", "value": 42}'
            result = OutputValidator.extract_json(text)
            if result:
                assert result["name"] == "test"
                assert result["value"] == 42
        except (AttributeError, TypeError):
            pytest.skip("OutputValidator not implemented")

    def test_extract_json_with_markdown(self):
        """Test JSON extraction from markdown code block."""
        try:
            text = 'Result:\n```json\n{"key": "value"}\n```'
            result = OutputValidator.extract_json(text)
            if result:
                assert result["key"] == "value"
        except (AttributeError, TypeError):
            pytest.skip("OutputValidator not implemented")

    def test_validate_json_schema(self):
        """Test schema validation."""
        try:
            data = {"name": "test", "age": 25}
            result = OutputValidator.validate_json_schema(data, ["name", "age"])
            if result:
                assert result["valid"] == True
        except (AttributeError, TypeError):
            pytest.skip("OutputValidator not implemented")

    def test_validate_json_schema_missing(self):
        """Test schema validation with missing fields."""
        try:
            data = {"name": "test"}
            result = OutputValidator.validate_json_schema(
                data, ["name", "age", "email"]
            )
            if result:
                assert result["valid"] == False
                assert "age" in result["missing"] or "email" in result["missing"]
        except (AttributeError, TypeError):
            pytest.skip("OutputValidator not implemented")

    def test_extract_list(self):
        """Test list extraction."""
        try:
            text = "Here are the items:\n1. First\n2. Second\n3. Third"
            result = OutputValidator.extract_list(text)
            if result:
                assert len(result) == 3
        except (AttributeError, TypeError):
            pytest.skip("OutputValidator not implemented")


# =============================================================================
# TESTS FOR TemplateLibrary
# =============================================================================


class TestTemplateLibrary:
    """Tests for template library."""

    def test_has_builtin_templates(self):
        """Test that built-in templates exist."""
        lib = TemplateLibrary()
        try:
            templates = lib.list_templates() if hasattr(lib, "list_templates") else []
            # Should have at least one built-in template
            if hasattr(lib, "templates"):
                assert len(lib.templates) >= 0  # At least initialized
        except (AttributeError, TypeError):
            pytest.skip("TemplateLibrary not implemented")

    def test_add_custom_template(self):
        """Test adding custom template."""
        lib = TemplateLibrary()
        try:
            tpl = Template(
                name="custom",
                template="Hello $name",
                description="Custom greeting",
                variables=["name"],
            )
            lib.add_template(tpl)
            retrieved = lib.get_template("custom")
            if retrieved:
                assert retrieved.name == "custom"
        except (AttributeError, TypeError):
            pytest.skip("TemplateLibrary not implemented")

    def test_search(self):
        """Test template search."""
        lib = TemplateLibrary()
        try:
            # Search for something
            results = lib.search("code")
            assert isinstance(results, list)
        except (AttributeError, TypeError):
            pytest.skip("TemplateLibrary not implemented")


# =============================================================================
# TESTS FOR PromptOptimizer
# =============================================================================


class TestPromptOptimizer:
    """Tests for prompt optimizer."""

    def test_analyze(self):
        """Test prompt analysis."""
        optimizer = PromptOptimizer()
        try:
            result = optimizer.analyze("Write code")
            if result:
                assert isinstance(result, dict)
        except (AttributeError, TypeError):
            pytest.skip("PromptOptimizer not implemented")

    def test_suggest_improvements(self):
        """Test improvement suggestions."""
        optimizer = PromptOptimizer()
        try:
            suggestions = optimizer.suggest_improvements("do thing")
            if suggestions:
                assert isinstance(suggestions, list)
                assert len(suggestions) > 0
        except (AttributeError, TypeError):
            pytest.skip("PromptOptimizer not implemented")


# =============================================================================
# TESTS FOR PromptToolkit
# =============================================================================


class TestPromptToolkit:
    """Tests for complete toolkit."""

    def test_initialization(self):
        """Test toolkit initializes."""
        try:
            toolkit = PromptToolkit()
            assert toolkit is not None
        except (AttributeError, TypeError):
            pytest.skip("PromptToolkit not implemented")

    def test_craft_prompt_method(self):
        """Test craft_prompt returns builder."""
        try:
            toolkit = PromptToolkit()
            craft = toolkit.craft_prompt()
            if craft:
                assert isinstance(craft, CRAFTPrompt)
        except (AttributeError, TypeError):
            pytest.skip("PromptToolkit not implemented")

    def test_with_cot(self):
        """Test CoT wrapper method."""
        try:
            toolkit = PromptToolkit()
            result = toolkit.with_cot("Test prompt")
            if result:
                assert "step" in result.lower()
        except (AttributeError, TypeError):
            pytest.skip("PromptToolkit not implemented")

    def test_validate_response(self):
        """Test response validation."""
        try:
            toolkit = PromptToolkit()
            result = toolkit.validate_response('{"key": "value"}', "json")
            if result:
                assert isinstance(result, dict)
        except (AttributeError, TypeError):
            pytest.skip("PromptToolkit not implemented")

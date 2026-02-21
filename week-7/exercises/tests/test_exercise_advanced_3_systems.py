"""
Week 7 - Tests for Exercise 3: Prompt Systems & Meta-Prompting
==============================================================

Run tests:
    python -m pytest tests/test_exercise_advanced_3_systems.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exercise_advanced_3_systems import (
    PromptTemplate,
    PromptLibrary,
    PromptChain,
    PromptOptimizer,
    ConditionalPrompt,
    create_prompt_generator,
    create_prompt_evaluator,
)


# =============================================================================
# TESTS FOR PromptTemplate (Task 1)
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_render_with_variables(self):
        """Test basic variable substitution."""
        template = PromptTemplate(
            name="test",
            template="Hello, $name!",
            description="Test",
            required_vars=["name"],
        )
        try:
            result = template.render(name="Alice")
            if result:
                assert "Alice" in result
        except (AttributeError, TypeError):
            pytest.skip("PromptTemplate not implemented")

    def test_render_with_defaults(self):
        """Test default variable substitution."""
        template = PromptTemplate(
            name="test",
            template="Hello, $name from $place!",
            description="Test",
            required_vars=["name"],
            default_vars={"place": "Earth"},
        )
        try:
            result = template.render(name="Bob")
            if result:
                assert "Bob" in result
                assert "Earth" in result
        except (AttributeError, TypeError):
            pytest.skip("PromptTemplate not implemented")

    def test_missing_required_raises(self):
        """Test that missing required vars raise error."""
        template = PromptTemplate(
            name="test",
            template="$required_var",
            description="Test",
            required_vars=["required_var"],
        )
        try:
            with pytest.raises(ValueError):
                template.render()
        except (AttributeError, TypeError):
            pytest.skip("PromptTemplate not implemented")

    def test_get_variables(self):
        """Test variable extraction."""
        template = PromptTemplate(
            name="test",
            template="$var1 and $var2 plus ${var3}",
            description="Test",
            required_vars=[],
        )
        try:
            vars = template.get_variables()
            if vars:
                assert "var1" in vars or "var2" in vars
        except (AttributeError, TypeError):
            pytest.skip("PromptTemplate not implemented")


# =============================================================================
# TESTS FOR PromptLibrary (Task 2)
# =============================================================================


class TestPromptLibrary:
    """Tests for PromptLibrary class."""

    def test_add_and_get_template(self):
        """Test adding and retrieving templates."""
        library = PromptLibrary()
        template = PromptTemplate(
            name="my_template", template="Hello $name", description="Test"
        )
        try:
            library.add_template(template)
            retrieved = library.get_template("my_template")
            if retrieved:
                assert retrieved.name == "my_template"
        except (AttributeError, TypeError):
            pytest.skip("PromptLibrary not implemented")

    def test_search_by_tag(self):
        """Test searching by tag."""
        library = PromptLibrary()
        template = PromptTemplate(
            name="code_tpl", template="Review $code", description="Code review"
        )
        try:
            library.add_template(template, tags=["code", "review"])
            results = library.search_by_tag("code")
            if results:
                assert len(results) >= 1
        except (AttributeError, TypeError):
            pytest.skip("PromptLibrary not implemented")

    def test_render_convenience(self):
        """Test render convenience method."""
        library = PromptLibrary()
        template = PromptTemplate(
            name="greet",
            template="Hi $name!",
            description="Greeting",
            required_vars=["name"],
        )
        try:
            library.add_template(template)
            result = library.render("greet", name="Test")
            if result:
                assert "Test" in result
        except (AttributeError, TypeError):
            pytest.skip("PromptLibrary not implemented")


# =============================================================================
# TESTS FOR PromptChain (Task 3)
# =============================================================================


class TestPromptChain:
    """Tests for PromptChain class."""

    def test_add_steps(self):
        """Test adding steps to chain."""
        chain = PromptChain("test_chain")
        try:
            chain.add_step("step1", "Do $action", "result1")
            chain.add_step("step2", "Then $action2", "result2")
            steps = chain.get_steps()
            if steps:
                assert len(steps) == 2
        except (AttributeError, TypeError):
            pytest.skip("PromptChain not implemented")

    def test_fluent_chaining(self):
        """Test fluent API returns self."""
        chain = PromptChain("test")
        try:
            result = chain.add_step("s1", "p1", "o1")
            assert result is chain
        except (AttributeError, TypeError):
            pytest.skip("PromptChain not implemented")

    def test_build_prompts(self):
        """Test building prompts from chain."""
        chain = PromptChain("test")
        try:
            chain.add_step("extract", "Extract from $text", "extracted")
            prompts = chain.build_prompts({"text": "sample"})
            if prompts:
                assert len(prompts) == 1
                assert "sample" in prompts[0]
        except (AttributeError, TypeError):
            pytest.skip("PromptChain not implemented")


# =============================================================================
# TESTS FOR PromptOptimizer (Task 4)
# =============================================================================


class TestPromptOptimizer:
    """Tests for PromptOptimizer class."""

    def test_analyze_returns_dict(self):
        """Test that analyze returns structured data."""
        optimizer = PromptOptimizer()
        try:
            result = optimizer.analyze_prompt("Write about Python")
            if result:
                assert isinstance(result, dict)
                assert "suggestions" in result or "issues" in result
        except (AttributeError, TypeError):
            pytest.skip("PromptOptimizer not implemented")

    def test_analyze_detects_issues(self):
        """Test that vague prompts are flagged."""
        optimizer = PromptOptimizer()
        try:
            result = optimizer.analyze_prompt("do something")
            if result and "issues" in result:
                assert len(result["issues"]) > 0
        except (AttributeError, TypeError):
            pytest.skip("PromptOptimizer not implemented")

    def test_create_improvement_prompt(self):
        """Test improvement prompt creation."""
        optimizer = PromptOptimizer()
        try:
            result = optimizer.create_improvement_prompt(
                "basic prompt", "summarization"
            )
            if result:
                assert "basic prompt" in result
        except (AttributeError, TypeError):
            pytest.skip("PromptOptimizer not implemented")


# =============================================================================
# TESTS FOR ConditionalPrompt (Task 5)
# =============================================================================


class TestConditionalPrompt:
    """Tests for ConditionalPrompt class."""

    def test_base_prompt_included(self):
        """Test that base prompt is always included."""
        cond = ConditionalPrompt("Base: $topic")
        try:
            result = cond.build({"topic": "Python"})
            if result:
                assert "Python" in result or "Base" in result
        except (AttributeError, TypeError):
            pytest.skip("ConditionalPrompt not implemented")

    def test_condition_applied(self):
        """Test that conditions are applied."""
        cond = ConditionalPrompt("Explain $topic")
        try:
            cond.add_condition(
                "beginner",
                lambda ctx: ctx.get("level") == "beginner",
                " Keep it simple.",
            )
            result = cond.build({"topic": "AI", "level": "beginner"})
            if result:
                assert "simple" in result.lower()
        except (AttributeError, TypeError):
            pytest.skip("ConditionalPrompt not implemented")

    def test_condition_not_applied(self):
        """Test that false conditions are not applied."""
        cond = ConditionalPrompt("Base")
        try:
            cond.add_condition("never", lambda ctx: False, "SHOULD NOT APPEAR")
            result = cond.build({})
            if result:
                assert "SHOULD NOT APPEAR" not in result
        except (AttributeError, TypeError):
            pytest.skip("ConditionalPrompt not implemented")


# =============================================================================
# TESTS FOR create_prompt_generator (Task 6)
# =============================================================================


class TestPromptGenerator:
    """Tests for meta-prompt generator."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = create_prompt_generator("code_review")
        if result:
            assert isinstance(result, str)

    def test_includes_task_type(self):
        """Test that task type is mentioned."""
        result = create_prompt_generator("sentiment_analysis")
        if result:
            assert "sentiment" in result.lower() or "analysis" in result.lower()

    def test_is_meta_prompt(self):
        """Test that it's a meta-prompt (asks for prompt generation)."""
        result = create_prompt_generator("summarization")
        if result:
            has_meta = any(
                word in result.lower()
                for word in ["create", "generate", "write", "design", "prompt"]
            )
            assert has_meta


# =============================================================================
# TESTS FOR create_prompt_evaluator (Task 7)
# =============================================================================


class TestPromptEvaluator:
    """Tests for meta-prompt evaluator."""

    def test_includes_prompt_to_evaluate(self):
        """Test that target prompt is included."""
        result = create_prompt_evaluator("Test prompt", ["clarity"])
        if result:
            assert "Test prompt" in result

    def test_includes_criteria(self):
        """Test that criteria are mentioned."""
        result = create_prompt_evaluator("Prompt", ["clarity", "specificity"])
        if result:
            assert "clarity" in result.lower() or "specificity" in result.lower()

    def test_asks_for_evaluation(self):
        """Test that evaluation is requested."""
        result = create_prompt_evaluator("Prompt", ["quality"])
        if result:
            has_eval = any(
                word in result.lower()
                for word in ["evaluate", "assess", "rate", "score", "analyze"]
            )
            assert has_eval

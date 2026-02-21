"""
Week 7 Exercise 3 (Advanced): Prompt Systems & Meta-Prompting
=============================================================

Build reusable prompt systems and meta-prompting capabilities.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv pydantic

Run this file:
    python exercise_advanced_3_systems.py

Run tests:
    python -m pytest tests/test_exercise_advanced_3_systems.py -v
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from string import Template
from abc import ABC, abstractmethod

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# =============================================================================
# SETUP
# =============================================================================


def setup_gemini():
    """Set up Gemini API client."""
    if not GENAI_AVAILABLE:
        return None

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# =============================================================================
# YOUR TASKS - Complete the classes and functions below
# =============================================================================


@dataclass
class PromptTemplate:
    """
    Task 1: Implement a PromptTemplate class.

    A template that can be rendered with variables.

    Attributes:
        name: Template identifier
        template: The template string with $variable placeholders
        description: What this template is for
        required_vars: List of required variable names
        default_vars: Dict of default values for optional variables
    """

    name: str
    template: str
    description: str
    required_vars: List[str] = field(default_factory=list)
    default_vars: Dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            The rendered prompt string

        Raises:
            ValueError: If required variables are missing

        Example:
            >>> tpl = PromptTemplate(
            ...     name="greeting",
            ...     template="Hello, $name! Welcome to $place.",
            ...     description="A greeting",
            ...     required_vars=["name"],
            ...     default_vars={"place": "our app"}
            ... )
            >>> tpl.render(name="Alice")
            'Hello, Alice! Welcome to our app.'
        """
        # TODO: Implement template rendering
        # 1. Check for missing required variables
        # 2. Merge defaults with provided kwargs
        # 3. Use string.Template for substitution
        pass

    def get_variables(self) -> List[str]:
        """
        Extract all variable names from the template.

        Returns:
            List of variable names found in template
        """
        # TODO: Extract variables from template string
        # Hint: Look for $variable or ${variable} patterns
        pass


class PromptLibrary:
    """
    Task 2: Implement a library for storing and retrieving prompt templates.

    Should support:
    - Adding templates
    - Getting templates by name
    - Searching templates by tag/category
    - Quick rendering by name
    """

    def __init__(self):
        """Initialize the prompt library."""
        # TODO: Initialize storage
        pass

    def add_template(self, template: PromptTemplate, tags: List[str] = None) -> None:
        """
        Add a template to the library.

        Args:
            template: The PromptTemplate to add
            tags: Optional list of tags for categorization
        """
        # TODO: Store template with tags
        pass

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.

        Args:
            name: The template name

        Returns:
            The template or None if not found
        """
        # TODO: Retrieve template
        pass

    def search_by_tag(self, tag: str) -> List[PromptTemplate]:
        """
        Find all templates with a given tag.

        Args:
            tag: The tag to search for

        Returns:
            List of matching templates
        """
        # TODO: Search templates by tag
        pass

    def render(self, name: str, **kwargs) -> str:
        """
        Convenience method to get and render a template.

        Args:
            name: Template name
            **kwargs: Variables for rendering

        Returns:
            Rendered prompt string
        """
        # TODO: Get template and render it
        pass

    def list_templates(self) -> List[str]:
        """Return list of all template names."""
        # TODO: Return template names
        pass


class PromptChain:
    """
    Task 3: Implement a prompt chain for multi-step prompting.

    A chain executes multiple prompts in sequence, where each
    prompt can use the output of previous prompts.
    """

    def __init__(self, name: str):
        """
        Initialize the chain.

        Args:
            name: Name for this chain
        """
        self.name = name
        # TODO: Initialize chain storage
        pass

    def add_step(
        self, step_name: str, prompt_template: str, output_key: str
    ) -> "PromptChain":
        """
        Add a step to the chain.

        Args:
            step_name: Name for this step
            prompt_template: Template with $variable placeholders
            output_key: Key to store the output under

        Returns:
            Self for fluent chaining

        Example:
            >>> chain = PromptChain("analysis")
            >>> chain.add_step("extract", "Extract keywords from: $text", "keywords")
            >>> chain.add_step("analyze", "Analyze these keywords: $keywords", "analysis")
        """
        # TODO: Add step to chain
        pass

    def get_steps(self) -> List[Dict[str, str]]:
        """Return list of steps with their names and templates."""
        # TODO: Return step information
        pass

    def build_prompts(self, initial_inputs: Dict[str, str]) -> List[str]:
        """
        Build all prompts in the chain (without executing).

        This is useful for previewing what will be executed.
        Note: Later prompts may have unresolved variables if they
        depend on outputs from previous steps.

        Args:
            initial_inputs: Initial input values

        Returns:
            List of prompt strings
        """
        # TODO: Build prompts for preview
        pass


class PromptOptimizer:
    """
    Task 4: Implement a prompt optimizer using meta-prompting.

    Uses an LLM to improve prompts based on specific criteria.
    """

    def __init__(self, model=None):
        """
        Initialize optimizer with optional model.

        Args:
            model: Gemini model instance (optional)
        """
        self.model = model or setup_gemini()

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt and identify potential improvements.

        Should return analysis without calling the LLM - just analyze
        the prompt structure locally.

        Args:
            prompt: The prompt to analyze

        Returns:
            Dict with:
            - word_count: Number of words
            - has_role: Whether it defines a role
            - has_format: Whether it specifies output format
            - has_examples: Whether it includes examples
            - has_constraints: Whether it has constraints
            - issues: List of potential issues
            - suggestions: List of improvement suggestions
        """
        # TODO: Analyze prompt locally
        pass

    def create_improvement_prompt(self, original: str, goal: str) -> str:
        """
        Create a meta-prompt to improve the original prompt.

        Args:
            original: The prompt to improve
            goal: What the prompt is trying to achieve

        Returns:
            A meta-prompt asking for improvements
        """
        # TODO: Create meta-prompt for improvement
        pass

    def improve_prompt(self, original: str, goal: str) -> str:
        """
        Use the LLM to improve a prompt (requires API).

        Args:
            original: The prompt to improve
            goal: What the prompt is trying to achieve

        Returns:
            The improved prompt
        """
        # TODO: Generate improved prompt using LLM
        pass


class ConditionalPrompt:
    """
    Task 5: Implement conditional prompting logic.

    A prompt that changes based on conditions.
    """

    def __init__(self, base_prompt: str):
        """
        Initialize with a base prompt.

        Args:
            base_prompt: The base prompt template
        """
        self.base_prompt = base_prompt
        # TODO: Initialize conditions storage
        pass

    def add_condition(
        self, condition_name: str, check_fn: Callable[[Dict], bool], addition: str
    ) -> "ConditionalPrompt":
        """
        Add a conditional addition to the prompt.

        Args:
            condition_name: Name for this condition
            check_fn: Function that takes context dict and returns bool
            addition: Text to add if condition is true

        Returns:
            Self for chaining
        """
        # TODO: Store condition
        pass

    def build(self, context: Dict[str, Any]) -> str:
        """
        Build the final prompt based on context.

        Args:
            context: Dict of context values for condition checking

        Returns:
            The complete prompt with conditional additions
        """
        # TODO: Build prompt with conditions applied
        pass


def create_prompt_generator(task_type: str) -> str:
    """
    Task 6: Create a meta-prompt that generates prompts.

    This prompt asks the LLM to generate a good prompt for a given task type.

    Args:
        task_type: Type of task (e.g., "code_review", "summarization", "translation")

    Returns:
        A meta-prompt for generating task-specific prompts

    Example:
        >>> meta = create_prompt_generator("code_review")
        >>> "code review" in meta.lower()
        True
    """
    # TODO: Create meta-prompt for prompt generation
    pass


def create_prompt_evaluator(prompt_to_evaluate: str, criteria: List[str]) -> str:
    """
    Task 7: Create a prompt to evaluate another prompt.

    Args:
        prompt_to_evaluate: The prompt to evaluate
        criteria: List of criteria to evaluate against

    Returns:
        A meta-prompt for evaluation

    Example:
        >>> eval_prompt = create_prompt_evaluator(
        ...     "Write code",
        ...     ["clarity", "specificity", "completeness"]
        ... )
        >>> "evaluate" in eval_prompt.lower() or "rate" in eval_prompt.lower()
        True
    """
    # TODO: Create evaluation meta-prompt
    pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 7 - Exercise 3: Prompt Systems & Meta-Prompting")
    print("=" * 60)

    # Test Task 1: PromptTemplate
    print("\n1. Testing PromptTemplate...")
    try:
        template = PromptTemplate(
            name="code_review",
            template="Review this $language code:\n$code\nFocus on: $focus",
            description="Code review template",
            required_vars=["language", "code"],
            default_vars={"focus": "bugs and best practices"},
        )
        rendered = template.render(language="Python", code="print('hello')")
        if rendered and "Python" in rendered:
            print("   ✓ PromptTemplate working")
            print(f"   Variables: {template.get_variables()}")
        else:
            print("   ✗ render() not working correctly")
    except Exception as e:
        print(f"   ✗ PromptTemplate error: {e}")

    # Test Task 2: PromptLibrary
    print("\n2. Testing PromptLibrary...")
    try:
        library = PromptLibrary()
        library.add_template(template, tags=["code", "review"])
        retrieved = library.get_template("code_review")
        if retrieved and retrieved.name == "code_review":
            print("   ✓ PromptLibrary working")
        else:
            print("   ✗ template retrieval not working")
    except Exception as e:
        print(f"   ✗ PromptLibrary error: {e}")

    # Test Task 3: PromptChain
    print("\n3. Testing PromptChain...")
    try:
        chain = PromptChain("summarize_and_translate")
        chain.add_step("summarize", "Summarize: $text", "summary")
        chain.add_step(
            "translate", "Translate to $target_language: $summary", "translation"
        )
        steps = chain.get_steps()
        if len(steps) == 2:
            print("   ✓ PromptChain working")
            print(f"   Steps: {[s['step_name'] for s in steps]}")
        else:
            print("   ✗ add_step() not working correctly")
    except Exception as e:
        print(f"   ✗ PromptChain error: {e}")

    # Test Task 4: PromptOptimizer
    print("\n4. Testing PromptOptimizer...")
    try:
        optimizer = PromptOptimizer()
        analysis = optimizer.analyze_prompt("Write about Python")
        if analysis and "suggestions" in analysis:
            print("   ✓ PromptOptimizer analyze working")
            print(f"   Issues found: {len(analysis.get('issues', []))}")
        else:
            print("   ✗ analyze_prompt not returning correct structure")
    except Exception as e:
        print(f"   ✗ PromptOptimizer error: {e}")

    # Test Task 5: ConditionalPrompt
    print("\n5. Testing ConditionalPrompt...")
    try:
        conditional = ConditionalPrompt("Explain $topic")
        conditional.add_condition(
            "beginner",
            lambda ctx: ctx.get("level") == "beginner",
            "\nUse simple language and analogies.",
        )
        conditional.add_condition(
            "expert",
            lambda ctx: ctx.get("level") == "expert",
            "\nInclude technical details and edge cases.",
        )
        result = conditional.build({"topic": "recursion", "level": "beginner"})
        if result and "simple language" in result:
            print("   ✓ ConditionalPrompt working")
        else:
            print("   ✗ conditional logic not working")
    except Exception as e:
        print(f"   ✗ ConditionalPrompt error: {e}")

    # Test Task 6: Prompt generator
    print("\n6. Testing prompt generator...")
    meta_prompt = create_prompt_generator("sentiment_analysis")
    if meta_prompt:
        print("   ✓ Meta-prompt generator created")
    else:
        print("   ✗ create_prompt_generator not implemented")

    # Test Task 7: Prompt evaluator
    print("\n7. Testing prompt evaluator...")
    eval_prompt = create_prompt_evaluator(
        "Analyze the sentiment of this text: $text",
        ["clarity", "specificity", "format"],
    )
    if eval_prompt:
        print("   ✓ Prompt evaluator created")
    else:
        print("   ✗ create_prompt_evaluator not implemented")

    print("\n" + "=" * 60)
    print("Run tests: python -m pytest tests/test_exercise_advanced_3_systems.py -v")
    print("=" * 60)

"""
Week 7 Project: Prompt Engineering Toolkit
==========================================

Build a comprehensive toolkit for prompt engineering that you can
use in real projects.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv pydantic

Run this file:
    python project_pipeline.py

Run tests:
    python -m pytest tests/test_project_pipeline.py -v
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from string import Template
from datetime import datetime

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    Field = None
    ValidationError = None
    PYDANTIC_AVAILABLE = False


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
# PART 1: CRAFT Prompt Builder
# =============================================================================


@dataclass
class CRAFTPrompt:
    """
    Task 1: Implement a CRAFT prompt builder.

    CRAFT = Context, Role, Action, Format, Tone

    This class should help users build well-structured prompts
    using the CRAFT framework.
    """

    context: str = ""
    role: str = ""
    action: str = ""
    format_spec: str = ""
    tone: str = ""

    def set_context(self, context: str) -> "CRAFTPrompt":
        """Set the context and return self for chaining."""
        # TODO: Implement
        pass

    def set_role(self, role: str) -> "CRAFTPrompt":
        """Set the role and return self for chaining."""
        # TODO: Implement
        pass

    def set_action(self, action: str) -> "CRAFTPrompt":
        """Set the action and return self for chaining."""
        # TODO: Implement
        pass

    def set_format(self, format_spec: str) -> "CRAFTPrompt":
        """Set the format and return self for chaining."""
        # TODO: Implement
        pass

    def set_tone(self, tone: str) -> "CRAFTPrompt":
        """Set the tone and return self for chaining."""
        # TODO: Implement
        pass

    def build(self) -> str:
        """
        Build the complete prompt from CRAFT components.

        Returns:
            The formatted prompt string
        """
        # TODO: Combine all components into a well-structured prompt
        pass

    def validate(self) -> List[str]:
        """
        Check if the prompt has all recommended components.

        Returns:
            List of warnings for missing components
        """
        # TODO: Return list of missing components
        pass


# =============================================================================
# PART 2: Few-Shot Example Manager
# =============================================================================


@dataclass
class Example:
    """A single input-output example with optional reasoning."""

    input_text: str
    output_text: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExampleManager:
    """
    Task 2: Implement a manager for few-shot examples.

    Should support:
    - Adding examples to categories
    - Selecting best examples for a task
    - Building few-shot prompts
    """

    def __init__(self):
        """Initialize the example manager."""
        # TODO: Initialize storage
        # self.categories: Dict[str, List[Example]] = {}
        pass

    def add_example(self, category: str, example: Example) -> None:
        """Add an example to a category."""
        # TODO: Implement
        pass

    def get_examples(
        self, category: str, n: int = 3, include_reasoning: bool = False
    ) -> List[Example]:
        """
        Get examples from a category.

        Args:
            category: The category name
            n: Number of examples to return
            include_reasoning: Whether to include reasoning examples

        Returns:
            List of examples
        """
        # TODO: Implement
        pass

    def build_few_shot_prompt(
        self, category: str, task_description: str, new_input: str, n_examples: int = 3
    ) -> str:
        """
        Build a complete few-shot prompt.

        Args:
            category: Category to get examples from
            task_description: Description of the task
            new_input: The new input to process
            n_examples: How many examples to include

        Returns:
            Complete few-shot prompt
        """
        # TODO: Implement
        pass

    def list_categories(self) -> List[str]:
        """Return list of all categories."""
        # TODO: Implement
        pass


# =============================================================================
# PART 3: Chain-of-Thought Wrapper
# =============================================================================


class CoTWrapper:
    """
    Task 3: Implement Chain-of-Thought wrapping for prompts.

    Takes any prompt and adds CoT reasoning instructions.
    """

    @staticmethod
    def wrap_basic(prompt: str) -> str:
        """
        Add basic "think step by step" instruction.

        Args:
            prompt: The original prompt

        Returns:
            Prompt with CoT instruction
        """
        # TODO: Implement
        pass

    @staticmethod
    def wrap_structured(prompt: str, steps: List[str]) -> str:
        """
        Add structured reasoning steps.

        Args:
            prompt: The original prompt
            steps: Specific reasoning steps to follow

        Returns:
            Prompt with structured CoT
        """
        # TODO: Implement
        pass

    @staticmethod
    def wrap_with_verification(prompt: str) -> str:
        """
        Add CoT with self-verification at the end.

        Args:
            prompt: The original prompt

        Returns:
            Prompt with reasoning and verification step
        """
        # TODO: Implement
        pass


# =============================================================================
# PART 4: Output Validator
# =============================================================================


class OutputValidator:
    """
    Task 4: Implement output validation for LLM responses.

    Validates that responses match expected formats.
    """

    @staticmethod
    def extract_json(response: str) -> Optional[Dict]:
        """
        Extract JSON from an LLM response.

        Handles cases where JSON is embedded in other text.

        Args:
            response: The LLM response text

        Returns:
            Parsed JSON dict or None if not found
        """
        # TODO: Implement JSON extraction
        # Hint: Look for {...} patterns
        pass

    @staticmethod
    def validate_json_schema(data: Dict, required_fields: List[str]) -> Dict[str, Any]:
        """
        Validate JSON data against required fields.

        Args:
            data: The JSON data to validate
            required_fields: List of required field names

        Returns:
            Dict with "valid": bool and "missing": list of missing fields
        """
        # TODO: Implement schema validation
        pass

    @staticmethod
    def extract_list(response: str) -> List[str]:
        """
        Extract a numbered or bulleted list from response.

        Args:
            response: The LLM response text

        Returns:
            List of extracted items
        """
        # TODO: Implement list extraction
        pass

    @staticmethod
    def validate_list_length(items: List, min_items: int, max_items: int) -> bool:
        """Check if list has expected number of items."""
        # TODO: Implement
        pass


# =============================================================================
# PART 5: Prompt Template Library
# =============================================================================


@dataclass
class Template:
    """A prompt template with metadata."""

    name: str
    template: str
    description: str
    variables: List[str]
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        # TODO: Implement
        pass


class TemplateLibrary:
    """
    Task 5: Implement a template library with built-in templates.
    """

    def __init__(self):
        """Initialize with some built-in templates."""
        self.templates: Dict[str, Template] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in templates."""
        # TODO: Add built-in templates for common tasks:
        # - code_review
        # - summarize
        # - explain_concept
        # - debug_error
        # - translate
        pass

    def add_template(self, template: Template) -> None:
        """Add a custom template."""
        # TODO: Implement
        pass

    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name."""
        # TODO: Implement
        pass

    def search(self, query: str) -> List[Template]:
        """Search templates by name, description, or tags."""
        # TODO: Implement
        pass

    def render(self, name: str, **kwargs) -> str:
        """Get and render a template."""
        # TODO: Implement
        pass


# =============================================================================
# PART 6: Meta-Prompt Optimizer
# =============================================================================


class PromptOptimizer:
    """
    Task 6: Implement prompt optimization using meta-prompting.
    """

    def __init__(self, model=None):
        """Initialize with optional model."""
        self.model = model or setup_gemini()

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt and return metrics.

        Returns:
            Dict with analysis results
        """
        # TODO: Implement local analysis
        # Check for: role, format, examples, constraints, specificity
        pass

    def create_improvement_meta_prompt(self, original: str, issues: List[str]) -> str:
        """
        Create a meta-prompt to fix specific issues.

        Args:
            original: The original prompt
            issues: List of issues to address

        Returns:
            Meta-prompt for improvement
        """
        # TODO: Implement
        pass

    def suggest_improvements(self, prompt: str) -> List[str]:
        """
        Get improvement suggestions (without LLM call).

        Args:
            prompt: The prompt to analyze

        Returns:
            List of suggestions
        """
        # TODO: Implement
        pass


# =============================================================================
# PART 7: Complete Toolkit
# =============================================================================


class PromptToolkit:
    """
    Task 7: Combine all components into a unified toolkit.

    This is the main interface for the complete prompt engineering toolkit.
    """

    def __init__(self, model=None):
        """Initialize all toolkit components."""
        # TODO: Initialize all components
        # self.craft = CRAFTPrompt()
        # self.examples = ExampleManager()
        # self.templates = TemplateLibrary()
        # self.optimizer = PromptOptimizer(model)
        pass

    def craft_prompt(self) -> CRAFTPrompt:
        """Get a new CRAFT prompt builder."""
        # TODO: Return new CRAFTPrompt instance
        pass

    def from_template(self, template_name: str, **kwargs) -> str:
        """Create prompt from template."""
        # TODO: Implement
        pass

    def with_examples(self, category: str, task: str, input_text: str) -> str:
        """Create few-shot prompt with examples."""
        # TODO: Implement
        pass

    def with_cot(self, prompt: str, steps: Optional[List[str]] = None) -> str:
        """Add Chain-of-Thought to any prompt."""
        # TODO: Implement
        pass

    def analyze_and_improve(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze a prompt and suggest improvements.

        Returns:
            Dict with analysis and suggestions
        """
        # TODO: Implement
        pass

    def validate_response(
        self, response: str, expected_format: str = "text"
    ) -> Dict[str, Any]:
        """
        Validate an LLM response.

        Args:
            response: The response to validate
            expected_format: "json", "list", or "text"

        Returns:
            Dict with validation results
        """
        # TODO: Implement
        pass


# =============================================================================
# MAIN - Test the toolkit
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 7 Project: Prompt Engineering Toolkit")
    print("=" * 60)

    # Test CRAFT Builder
    print("\n1. Testing CRAFT Builder...")
    try:
        craft = CRAFTPrompt()
        prompt = (
            craft.set_context("Building a Python web application")
            .set_role("Senior software engineer")
            .set_action("Review this code for security issues")
            .set_format("Numbered list of issues with severity")
            .set_tone("Professional and constructive")
            .build()
        )
        if prompt:
            print("   ✓ CRAFT prompt built")
            warnings = craft.validate()
            print(f"   Warnings: {warnings}")
        else:
            print("   ✗ CRAFT builder not implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test Example Manager
    print("\n2. Testing Example Manager...")
    try:
        em = ExampleManager()
        em.add_example("sentiment", Example("Great!", "positive"))
        em.add_example("sentiment", Example("Terrible", "negative"))
        em.add_example("sentiment", Example("It's okay", "neutral"))
        prompt = em.build_few_shot_prompt("sentiment", "Classify sentiment", "Amazing!")
        if prompt:
            print("   ✓ Few-shot prompt built")
        else:
            print("   ✗ Example manager not fully implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test CoT Wrapper
    print("\n3. Testing CoT Wrapper...")
    try:
        basic_cot = CoTWrapper.wrap_basic("What is 17 * 23?")
        if basic_cot and "step" in basic_cot.lower():
            print("   ✓ Basic CoT working")
        else:
            print("   ✗ CoT wrapper not implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test Output Validator
    print("\n4. Testing Output Validator...")
    try:
        test_response = 'Here is the analysis: {"sentiment": "positive", "score": 0.9}'
        extracted = OutputValidator.extract_json(test_response)
        if extracted and "sentiment" in extracted:
            print("   ✓ JSON extraction working")
        else:
            print("   ✗ Output validator not implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test Template Library
    print("\n5. Testing Template Library...")
    try:
        lib = TemplateLibrary()
        templates = lib.search("code")
        print(f"   Found {len(templates)} code-related templates")
        if templates:
            print("   ✓ Template library working")
        else:
            print("   ✗ No built-in templates found")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test Optimizer
    print("\n6. Testing Prompt Optimizer...")
    try:
        optimizer = PromptOptimizer()
        analysis = optimizer.analyze("Write about Python")
        if analysis:
            print("   ✓ Prompt analyzer working")
            suggestions = optimizer.suggest_improvements("Write about Python")
            print(f"   Suggestions: {suggestions[:2]}")
        else:
            print("   ✗ Optimizer not implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test Complete Toolkit
    print("\n7. Testing Complete Toolkit...")
    try:
        toolkit = PromptToolkit()
        craft = toolkit.craft_prompt()
        if craft:
            print("   ✓ Toolkit initialized")
        else:
            print("   ✗ Toolkit not fully implemented")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Run tests: python -m pytest tests/test_project_pipeline.py -v")
    print("=" * 60)

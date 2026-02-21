"""
Week 7 Exercise 3 (Advanced): Prompt Systems & Meta-Prompting - SOLUTIONS
========================================================================
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from string import Template

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


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


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution."""

    name: str
    template: str
    description: str
    required_vars: List[str] = field(default_factory=list)
    default_vars: Dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        # Check for missing required variables
        missing = set(self.required_vars) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Merge defaults with provided kwargs
        all_vars = {**self.default_vars, **kwargs}

        # Use string.Template for substitution
        return Template(self.template).safe_substitute(**all_vars)

    def get_variables(self) -> List[str]:
        """Extract all variable names from the template."""
        # Match $var or ${var} patterns
        pattern = r"\$\{?(\w+)\}?"
        matches = re.findall(pattern, self.template)
        return list(set(matches))


class PromptLibrary:
    """Library for storing and retrieving prompt templates."""

    def __init__(self):
        """Initialize the prompt library."""
        self.templates: Dict[str, PromptTemplate] = {}
        self.tags: Dict[str, List[str]] = {}  # tag -> list of template names

    def add_template(self, template: PromptTemplate, tags: List[str] = None) -> None:
        """Add a template to the library."""
        self.templates[template.name] = template

        if tags:
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                self.tags[tag].append(template.name)

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def search_by_tag(self, tag: str) -> List[PromptTemplate]:
        """Find all templates with a given tag."""
        template_names = self.tags.get(tag, [])
        return [
            self.templates[name] for name in template_names if name in self.templates
        ]

    def render(self, name: str, **kwargs) -> str:
        """Convenience method to get and render a template."""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        return template.render(**kwargs)

    def list_templates(self) -> List[str]:
        """Return list of all template names."""
        return list(self.templates.keys())


class PromptChain:
    """A chain of prompts for multi-step prompting."""

    def __init__(self, name: str):
        """Initialize the chain."""
        self.name = name
        self.steps: List[Dict[str, str]] = []

    def add_step(
        self, step_name: str, prompt_template: str, output_key: str
    ) -> "PromptChain":
        """Add a step to the chain."""
        self.steps.append(
            {
                "step_name": step_name,
                "template": prompt_template,
                "output_key": output_key,
            }
        )
        return self

    def get_steps(self) -> List[Dict[str, str]]:
        """Return list of steps with their names and templates."""
        return self.steps

    def build_prompts(self, initial_inputs: Dict[str, str]) -> List[str]:
        """Build all prompts in the chain (without executing)."""
        prompts = []
        context = dict(initial_inputs)

        for step in self.steps:
            template = Template(step["template"])
            prompt = template.safe_substitute(**context)
            prompts.append(prompt)
            # Note: In preview mode, we can't fill in outputs from previous steps
            context[step["output_key"]] = f"[output from {step['step_name']}]"

        return prompts


class PromptOptimizer:
    """Prompt optimizer using meta-prompting."""

    def __init__(self, model=None):
        """Initialize optimizer with optional model."""
        self.model = model or setup_gemini()

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and identify potential improvements."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        # Check for components
        has_role = any(x in prompt_lower for x in ["you are", "act as", "role"])
        has_format = any(
            x in prompt_lower for x in ["format", "json", "markdown", "list"]
        )
        has_examples = any(
            x in prompt_lower for x in ["example", "e.g.", "for instance"]
        )
        has_constraints = any(
            x in prompt_lower for x in ["must", "should", "limit", "maximum"]
        )

        # Identify issues
        issues = []
        suggestions = []

        if word_count < 10:
            issues.append("Prompt is very short")
            suggestions.append("Add more detail and context")

        if not has_role:
            issues.append("No role/persona specified")
            suggestions.append("Add a role like 'You are a senior developer...'")

        if not has_format:
            issues.append("No output format specified")
            suggestions.append("Specify expected format (JSON, markdown, list, etc.)")

        if not has_examples:
            issues.append("No examples provided")
            suggestions.append("Consider adding examples for clarity")

        if not has_constraints:
            issues.append("No constraints specified")
            suggestions.append(
                "Add constraints like length limits or style requirements"
            )

        return {
            "word_count": word_count,
            "has_role": has_role,
            "has_format": has_format,
            "has_examples": has_examples,
            "has_constraints": has_constraints,
            "issues": issues,
            "suggestions": suggestions,
        }

    def create_improvement_prompt(self, original: str, goal: str) -> str:
        """Create a meta-prompt to improve the original prompt."""
        return f"""You are a prompt engineering expert.

I have this prompt that I want to improve:
---
{original}
---

The goal of this prompt is: {goal}

Please improve this prompt by:
1. Adding a clear role/persona
2. Making the task more specific
3. Specifying the output format
4. Adding helpful constraints
5. Including examples if appropriate

Provide ONLY the improved prompt, ready to use."""

    def improve_prompt(self, original: str, goal: str) -> str:
        """Use the LLM to improve a prompt (requires API)."""
        if not self.model:
            return original

        meta_prompt = self.create_improvement_prompt(original, goal)

        try:
            response = self.model.generate_content(meta_prompt)
            return response.text
        except Exception:
            return original


class ConditionalPrompt:
    """A prompt that changes based on conditions."""

    def __init__(self, base_prompt: str):
        """Initialize with a base prompt."""
        self.base_prompt = base_prompt
        self.conditions: List[Dict[str, Any]] = []

    def add_condition(
        self, condition_name: str, check_fn: Callable[[Dict], bool], addition: str
    ) -> "ConditionalPrompt":
        """Add a conditional addition to the prompt."""
        self.conditions.append(
            {"name": condition_name, "check": check_fn, "addition": addition}
        )
        return self

    def build(self, context: Dict[str, Any]) -> str:
        """Build the final prompt based on context."""
        # Start with base prompt, substituting variables
        prompt = Template(self.base_prompt).safe_substitute(**context)

        # Apply conditions
        for condition in self.conditions:
            if condition["check"](context):
                prompt += condition["addition"]

        return prompt


def create_prompt_generator(task_type: str) -> str:
    """Create a meta-prompt that generates prompts."""
    return f"""You are a prompt engineering expert.

Create an effective prompt for the following task type: {task_type}

Your prompt should include:
1. A clear role/persona for the AI
2. Specific, unambiguous instructions
3. The expected output format
4. Any relevant constraints (length, style, etc.)
5. One or two examples if helpful

The prompt you create should be ready to use immediately.
Return ONLY the prompt, no explanations."""


def create_prompt_evaluator(prompt_to_evaluate: str, criteria: List[str]) -> str:
    """Create a prompt to evaluate another prompt."""
    criteria_list = "\n".join(f"- {c}" for c in criteria)

    return f"""Evaluate the following prompt against these criteria:

Criteria to assess:
{criteria_list}

Prompt to evaluate:
---
{prompt_to_evaluate}
---

For each criterion, rate from 1-5 and explain your rating.

Then provide:
1. Overall score (1-10)
2. Top 3 strengths
3. Top 3 areas for improvement
4. Suggested improved version of the prompt"""

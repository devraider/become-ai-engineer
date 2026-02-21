"""
Week 7 Project: Prompt Engineering Toolkit - SOLUTIONS
=====================================================
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
    """CRAFT prompt builder."""

    context: str = ""
    role: str = ""
    action: str = ""
    format_spec: str = ""
    tone: str = ""

    def set_context(self, context: str) -> "CRAFTPrompt":
        """Set the context and return self for chaining."""
        self.context = context
        return self

    def set_role(self, role: str) -> "CRAFTPrompt":
        """Set the role and return self for chaining."""
        self.role = role
        return self

    def set_action(self, action: str) -> "CRAFTPrompt":
        """Set the action and return self for chaining."""
        self.action = action
        return self

    def set_format(self, format_spec: str) -> "CRAFTPrompt":
        """Set the format and return self for chaining."""
        self.format_spec = format_spec
        return self

    def set_tone(self, tone: str) -> "CRAFTPrompt":
        """Set the tone and return self for chaining."""
        self.tone = tone
        return self

    def build(self) -> str:
        """Build the complete prompt from CRAFT components."""
        parts = []

        if self.context:
            parts.append(f"Context: {self.context}")

        if self.role:
            parts.append(f"Role: You are {self.role}.")

        if self.action:
            parts.append(f"Task: {self.action}")

        if self.format_spec:
            parts.append(f"Output Format: {self.format_spec}")

        if self.tone:
            parts.append(f"Tone: {self.tone}")

        if parts:
            parts.append("\nPlease proceed with the task.")

        return "\n\n".join(parts)

    def validate(self) -> List[str]:
        """Check if the prompt has all recommended components."""
        warnings = []

        if not self.context:
            warnings.append("Missing context")
        if not self.role:
            warnings.append("Missing role")
        if not self.action:
            warnings.append("Missing action/task")
        if not self.format_spec:
            warnings.append("Missing format specification")
        if not self.tone:
            warnings.append("Missing tone")

        return warnings


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
    """Manager for few-shot examples."""

    def __init__(self):
        """Initialize the example manager."""
        self.categories: Dict[str, List[Example]] = {}

    def add_example(self, category: str, example: Example) -> None:
        """Add an example to a category."""
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(example)

    def get_examples(
        self, category: str, n: int = 3, include_reasoning: bool = False
    ) -> List[Example]:
        """Get examples from a category."""
        if category not in self.categories:
            return []

        examples = self.categories[category][:n]

        if not include_reasoning:
            # Return without reasoning
            return [
                Example(e.input_text, e.output_text, None, e.metadata) for e in examples
            ]

        return examples

    def build_few_shot_prompt(
        self, category: str, task_description: str, new_input: str, n_examples: int = 3
    ) -> str:
        """Build a complete few-shot prompt."""
        examples = self.get_examples(category, n_examples, include_reasoning=True)

        parts = [f"Task: {task_description}\n", "Examples:\n"]

        for i, ex in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Input: {ex.input_text}")
            if ex.reasoning:
                parts.append(f"Reasoning: {ex.reasoning}")
            parts.append(f"Output: {ex.output_text}")
            parts.append("")

        parts.append("Now process this input:")
        parts.append(f"Input: {new_input}")
        parts.append("Output:")

        return "\n".join(parts)

    def list_categories(self) -> List[str]:
        """Return list of all categories."""
        return list(self.categories.keys())


# =============================================================================
# PART 3: Chain-of-Thought Wrapper
# =============================================================================


class CoTWrapper:
    """Chain-of-Thought wrapping for prompts."""

    @staticmethod
    def wrap_basic(prompt: str) -> str:
        """Add basic 'think step by step' instruction."""
        return f"""{prompt}

Think through this step by step before giving your final answer."""

    @staticmethod
    def wrap_structured(prompt: str, steps: List[str]) -> str:
        """Add structured reasoning steps."""
        steps_text = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

        return f"""{prompt}

Work through this following these steps:
{steps_text}

Show your work for each step, then provide your final answer."""

    @staticmethod
    def wrap_with_verification(prompt: str) -> str:
        """Add CoT with self-verification at the end."""
        return f"""{prompt}

Think through this step by step:
1. Work through the problem systematically
2. Arrive at your answer
3. Verify your answer by checking your work
4. If you find any errors, correct them

Show your reasoning, verification, and final answer."""


# =============================================================================
# PART 4: Output Validator
# =============================================================================


class OutputValidator:
    """Output validation for LLM responses."""

    @staticmethod
    def extract_json(response: str) -> Optional[Dict]:
        """Extract JSON from an LLM response."""
        # Try to find JSON in the response
        # First, try direct parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Look for JSON in markdown code blocks
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(code_block_pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Look for {...} patterns
        brace_pattern = r"\{[^{}]*\}"
        matches = re.findall(brace_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Try nested braces
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def validate_json_schema(data: Dict, required_fields: List[str]) -> Dict[str, Any]:
        """Validate JSON data against required fields."""
        missing = [f for f in required_fields if f not in data]

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "present": [f for f in required_fields if f in data],
        }

    @staticmethod
    def extract_list(response: str) -> List[str]:
        """Extract a numbered or bulleted list from response."""
        items = []

        # Match numbered lists (1. item, 2. item, etc.)
        numbered_pattern = r"^\s*\d+[\.\)]\s*(.+)$"

        # Match bulleted lists (- item, * item, • item)
        bulleted_pattern = r"^\s*[-*•]\s*(.+)$"

        for line in response.split("\n"):
            line = line.strip()

            # Try numbered pattern
            match = re.match(numbered_pattern, line)
            if match:
                items.append(match.group(1).strip())
                continue

            # Try bulleted pattern
            match = re.match(bulleted_pattern, line)
            if match:
                items.append(match.group(1).strip())

        return items

    @staticmethod
    def validate_list_length(items: List, min_items: int, max_items: int) -> bool:
        """Check if list has expected number of items."""
        return min_items <= len(items) <= max_items


# =============================================================================
# PART 5: Prompt Template Library
# =============================================================================


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""

    name: str
    template: str
    description: str
    variables: List[str]
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        return Template(self.template).safe_substitute(**kwargs)


class TemplateLibrary:
    """Template library with built-in templates."""

    def __init__(self):
        """Initialize with built-in templates."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in templates."""
        builtins = [
            PromptTemplate(
                name="code_review",
                template="""You are a senior $language developer.

Review this code:
```$language
$code
```

Provide:
1. Bugs or errors found
2. Performance issues
3. Best practice violations
4. Security concerns
5. Suggested improvements with code examples""",
                description="Code review template",
                variables=["language", "code"],
                tags=["code", "review"],
            ),
            PromptTemplate(
                name="summarize",
                template="""Summarize the following text in $length:

$text

Requirements:
- Capture the main points
- Be concise and clear
- Maintain the original meaning""",
                description="Text summarization template",
                variables=["text", "length"],
                tags=["summarize", "text"],
            ),
            PromptTemplate(
                name="explain_concept",
                template="""Explain $concept to someone with $level experience.

Requirements:
- Use clear, simple language appropriate for the audience
- Include a practical example
- Mention common misconceptions
- Keep it under $word_limit words""",
                description="Concept explanation template",
                variables=["concept", "level", "word_limit"],
                tags=["explain", "education"],
            ),
            PromptTemplate(
                name="debug_error",
                template="""I'm getting this error in my $language code:

Error:
```
$error
```

Code:
```$language
$code
```

Help me:
1. Understand what's causing this error
2. Fix the code
3. Prevent similar errors""",
                description="Debug error template",
                variables=["language", "error", "code"],
                tags=["code", "debug"],
            ),
            PromptTemplate(
                name="translate",
                template="""Translate the following from $source_lang to $target_lang:

$text

Requirements:
- Maintain the original meaning
- Use natural-sounding $target_lang
- Preserve formatting""",
                description="Translation template",
                variables=["source_lang", "target_lang", "text"],
                tags=["translate", "language"],
            ),
        ]

        for tpl in builtins:
            self.templates[tpl.name] = tpl

    def add_template(self, template: PromptTemplate) -> None:
        """Add a custom template."""
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)

    def search(self, query: str) -> List[PromptTemplate]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for tpl in self.templates.values():
            if (
                query_lower in tpl.name.lower()
                or query_lower in tpl.description.lower()
                or any(query_lower in tag.lower() for tag in tpl.tags)
            ):
                results.append(tpl)

        return results

    def render(self, name: str, **kwargs) -> str:
        """Get and render a template."""
        tpl = self.get_template(name)
        if not tpl:
            raise ValueError(f"Template '{name}' not found")
        return tpl.render(**kwargs)

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())


# =============================================================================
# PART 6: Meta-Prompt Optimizer
# =============================================================================


class PromptOptimizer:
    """Prompt optimization using meta-prompting."""

    def __init__(self, model=None):
        """Initialize with optional model."""
        self.model = model or setup_gemini()

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and return metrics."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        return {
            "word_count": word_count,
            "has_role": any(x in prompt_lower for x in ["you are", "act as"]),
            "has_format": any(
                x in prompt_lower for x in ["format", "json", "markdown"]
            ),
            "has_examples": "example" in prompt_lower,
            "has_constraints": any(
                x in prompt_lower for x in ["must", "should", "limit"]
            ),
            "specificity_score": min(5, 1 + word_count // 20),
        }

    def create_improvement_meta_prompt(self, original: str, issues: List[str]) -> str:
        """Create a meta-prompt to fix specific issues."""
        issues_text = "\n".join(f"- {issue}" for issue in issues)

        return f"""Improve this prompt to address these issues:

Issues to fix:
{issues_text}

Original prompt:
{original}

Provide only the improved prompt."""

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Get improvement suggestions (without LLM call)."""
        analysis = self.analyze(prompt)
        suggestions = []

        if not analysis["has_role"]:
            suggestions.append(
                "Add a role/persona (e.g., 'You are a senior developer...')"
            )

        if not analysis["has_format"]:
            suggestions.append("Specify output format (JSON, markdown, list, etc.)")

        if not analysis["has_examples"]:
            suggestions.append("Consider adding examples for clarity")

        if not analysis["has_constraints"]:
            suggestions.append("Add constraints (length limits, style requirements)")

        if analysis["word_count"] < 20:
            suggestions.append("Add more detail and context")

        return suggestions


# =============================================================================
# PART 7: Complete Toolkit
# =============================================================================


class PromptToolkit:
    """Complete prompt engineering toolkit."""

    def __init__(self, model=None):
        """Initialize all toolkit components."""
        self.examples = ExampleManager()
        self.templates = TemplateLibrary()
        self.optimizer = PromptOptimizer(model)

    def craft_prompt(self) -> CRAFTPrompt:
        """Get a new CRAFT prompt builder."""
        return CRAFTPrompt()

    def from_template(self, template_name: str, **kwargs) -> str:
        """Create prompt from template."""
        return self.templates.render(template_name, **kwargs)

    def with_examples(self, category: str, task: str, input_text: str) -> str:
        """Create few-shot prompt with examples."""
        return self.examples.build_few_shot_prompt(category, task, input_text)

    def with_cot(self, prompt: str, steps: Optional[List[str]] = None) -> str:
        """Add Chain-of-Thought to any prompt."""
        if steps:
            return CoTWrapper.wrap_structured(prompt, steps)
        return CoTWrapper.wrap_basic(prompt)

    def analyze_and_improve(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and suggest improvements."""
        analysis = self.optimizer.analyze(prompt)
        suggestions = self.optimizer.suggest_improvements(prompt)

        return {"analysis": analysis, "suggestions": suggestions}

    def validate_response(
        self, response: str, expected_format: str = "text"
    ) -> Dict[str, Any]:
        """Validate an LLM response."""
        result = {"format": expected_format, "valid": True}

        if expected_format == "json":
            extracted = OutputValidator.extract_json(response)
            result["valid"] = extracted is not None
            result["data"] = extracted

        elif expected_format == "list":
            items = OutputValidator.extract_list(response)
            result["valid"] = len(items) > 0
            result["items"] = items
            result["count"] = len(items)

        else:  # text
            result["valid"] = len(response.strip()) > 0
            result["length"] = len(response)

        return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Week 7 Project: Prompt Engineering Toolkit - SOLUTIONS")
    print("=" * 60)

    toolkit = PromptToolkit()

    # Demo CRAFT
    print("\n1. CRAFT Builder:")
    prompt = (
        toolkit.craft_prompt()
        .set_context("E-commerce application")
        .set_role("Senior Python developer")
        .set_action("Review checkout flow code")
        .set_format("Numbered list of issues")
        .set_tone("Constructive")
        .build()
    )
    print(prompt[:200] + "...")

    # Demo templates
    print("\n2. Template Library:")
    print(f"Available: {toolkit.templates.list_templates()}")

    # Demo CoT
    print("\n3. Chain-of-Thought:")
    cot_prompt = toolkit.with_cot("Is 17 * 24 greater than 400?")
    print(cot_prompt[:100] + "...")

    # Demo analyzer
    print("\n4. Prompt Analyzer:")
    analysis = toolkit.analyze_and_improve("write code")
    print(f"Suggestions: {analysis['suggestions']}")

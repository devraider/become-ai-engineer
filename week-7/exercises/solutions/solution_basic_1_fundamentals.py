"""
Week 7 Exercise 1 (Basic): Prompt Engineering Fundamentals - SOLUTIONS
=====================================================================
"""

import json
from typing import Dict, List, Optional


def craft_prompt(
    context: str, role: str, action: str, format_spec: str, tone: str
) -> str:
    """Build a prompt using the CRAFT framework."""
    return f"""Context: {context}

Role: You are {role}.

Task: {action}

Output Format: {format_spec}

Tone: {tone}

Please proceed with the task."""


def create_specific_prompt(vague_request: str) -> str:
    """Transform a vague request into a specific prompt."""
    return f"""Task: {vague_request}

Requirements:
- Be specific and detailed in your response
- Include practical examples where relevant
- Structure your response clearly with headings or bullet points
- Keep your response focused and concise (under 500 words)
- Explain any technical terms you use

Output Format: Use markdown formatting with clear sections."""


def add_constraints(base_prompt: str, constraints: Dict[str, any]) -> str:
    """Add constraints to an existing prompt."""
    constraint_lines = [base_prompt, "\nConstraints:"]

    if "max_words" in constraints:
        constraint_lines.append(f"- Maximum length: {constraints['max_words']} words")

    if "audience" in constraints:
        constraint_lines.append(f"- Target audience: {constraints['audience']}")

    if "format" in constraints:
        constraint_lines.append(f"- Output format: {constraints['format']}")

    if "language" in constraints:
        constraint_lines.append(f"- Programming language: {constraints['language']}")

    if "tone" in constraints:
        constraint_lines.append(f"- Tone: {constraints['tone']}")

    return "\n".join(constraint_lines)


def create_role_prompt(role: str, expertise: str, task: str) -> str:
    """Create a prompt with a specific role/persona."""
    return f"""You are a {role} with {expertise}.

Your Task: {task}

Please provide a thorough and professional response based on your expertise.
Be specific and provide actionable insights where relevant."""


def create_step_by_step_prompt(task: str, steps: List[str]) -> str:
    """Create a prompt that guides through specific steps."""
    step_lines = [f"Task: {task}\n", "Please follow these steps:"]

    for i, step in enumerate(steps, 1):
        step_lines.append(f"{i}. {step}")

    step_lines.append("\nProvide your response following each step in order.")

    return "\n".join(step_lines)


def create_example_based_prompt(
    task: str, example_input: str, example_output: str
) -> str:
    """Create a prompt with a single example (one-shot)."""
    return f"""Task: {task}

Example:
Input: {example_input}
Output: {example_output}

Now apply the same pattern:
Input: """


def create_output_format_prompt(task: str, format_type: str, fields: List[str]) -> str:
    """Create a prompt that specifies exact output format."""
    fields_str = ", ".join(f'"{f}"' for f in fields)

    if format_type.lower() == "json":
        schema_example = (
            "{\n" + ",\n".join(f'    "{f}": <value>' for f in fields) + "\n}"
        )
        return f"""{task}

Respond ONLY with valid JSON containing these fields: {fields_str}

Expected format:
{schema_example}

No explanations, no markdown, just the JSON object."""

    elif format_type.lower() == "markdown":
        return f"""{task}

Format your response in markdown with the following sections:
{chr(10).join(f'## {f}' for f in fields)}"""

    elif format_type.lower() == "list":
        return f"""{task}

Provide your response as a list with these items:
{chr(10).join(f'- {f}' for f in fields)}"""

    else:
        return f"""{task}

Include the following in your response: {fields_str}
Format: {format_type}"""


def create_negative_prompt(task: str, things_to_avoid: List[str]) -> str:
    """Create a prompt that specifies what NOT to do."""
    avoid_lines = [f"Task: {task}\n", "Important - DO NOT:"]

    for item in things_to_avoid:
        if item.lower().startswith("don't"):
            avoid_lines.append(f"- {item}")
        else:
            avoid_lines.append(f"- {item}")

    return "\n".join(avoid_lines)


def create_iterative_prompt(initial_output: str, feedback: str) -> str:
    """Create a prompt for iterating/improving on previous output."""
    return f"""Here is a previous response:

---
{initial_output}
---

Please improve this response based on the following feedback:
{feedback}

Provide the complete improved version."""


def analyze_prompt_quality(prompt: str) -> Dict[str, any]:
    """Analyze a prompt and return quality metrics."""
    prompt_lower = prompt.lower()

    # Check for various components
    has_context = any(
        word in prompt_lower for word in ["context", "background", "situation", "given"]
    )
    has_format = any(
        word in prompt_lower
        for word in ["format", "json", "markdown", "list", "bullet", "output"]
    )
    has_constraints = any(
        word in prompt_lower
        for word in [
            "limit",
            "maximum",
            "under",
            "words",
            "constraint",
            "must",
            "should",
        ]
    )
    has_examples = any(
        word in prompt_lower for word in ["example", "for instance", "such as", "e.g."]
    )
    has_role = any(
        word in prompt_lower
        for word in ["you are", "act as", "role", "expert", "specialist"]
    )

    # Calculate specificity score (1-5)
    word_count = len(prompt.split())
    specificity_score = 1

    if word_count > 10:
        specificity_score += 1
    if word_count > 30:
        specificity_score += 1
    if has_format or has_constraints:
        specificity_score += 1
    if has_examples or has_role:
        specificity_score += 1

    specificity_score = min(5, specificity_score)

    # Generate suggestions
    suggestions = []

    if not has_context:
        suggestions.append("Add context or background information")
    if not has_format:
        suggestions.append("Specify the desired output format")
    if not has_constraints:
        suggestions.append("Add constraints (length, style, etc.)")
    if not has_examples:
        suggestions.append("Include examples to clarify expectations")
    if not has_role:
        suggestions.append("Consider adding a role/persona for the AI")
    if word_count < 15:
        suggestions.append("Add more detail and specificity")

    return {
        "word_count": word_count,
        "has_context": has_context,
        "has_format": has_format,
        "has_constraints": has_constraints,
        "has_examples": has_examples,
        "has_role": has_role,
        "specificity_score": specificity_score,
        "suggestions": suggestions,
    }

"""
Week 7 Exercise 2 (Intermediate): Advanced Prompting Techniques - SOLUTIONS
==========================================================================
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


def create_chain_of_thought_prompt(question: str) -> str:
    """Create a Chain-of-Thought (CoT) prompt."""
    return f"""{question}

Think through this step by step:
1. First, identify what is being asked
2. Then, work through the problem systematically
3. Finally, provide your answer

Show your reasoning at each step before giving the final answer."""


def create_structured_cot_prompt(problem: str, reasoning_steps: List[str]) -> str:
    """Create a structured CoT prompt with specific reasoning steps."""
    steps_text = "\n".join(f"{i}. {step}" for i, step in enumerate(reasoning_steps, 1))

    return f"""Problem: {problem}

Work through this problem following these steps:
{steps_text}

Address each step in order, showing your work, then provide your final answer."""


def create_few_shot_prompt(
    task_description: str, examples: List[Dict[str, str]], new_input: str
) -> str:
    """Create a few-shot learning prompt."""
    prompt_parts = [f"Task: {task_description}\n", "Learn from these examples:\n"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['output']}")
        prompt_parts.append("")

    prompt_parts.append("Now process this new input:")
    prompt_parts.append(f"Input: {new_input}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)


def create_few_shot_with_reasoning(
    task_description: str, examples: List[Dict[str, str]], new_input: str
) -> str:
    """Create a few-shot prompt with reasoning explanations."""
    prompt_parts = [f"Task: {task_description}\n", "Learn from these examples:\n"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Reasoning: {ex.get('reasoning', 'N/A')}")
        prompt_parts.append(f"Output: {ex['output']}")
        prompt_parts.append("")

    prompt_parts.append("Now process this new input:")
    prompt_parts.append(f"Input: {new_input}")
    prompt_parts.append("Reasoning:")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)


def create_json_output_prompt(task: str, schema: Dict[str, str]) -> str:
    """Create a prompt requesting JSON output with a schema."""
    schema_lines = [f'    "{key}": {value}' for key, value in schema.items()]
    schema_example = "{\n" + ",\n".join(schema_lines) + "\n}"

    return f"""{task}

Respond with valid JSON matching this schema:
{schema_example}

Important:
- Return ONLY the JSON object
- No explanations or additional text
- Ensure proper JSON syntax"""


def create_list_output_prompt(task: str, num_items: int, item_format: str) -> str:
    """Create a prompt requesting a numbered list output."""
    return f"""{task}

Requirements:
- Provide exactly {num_items} items in a numbered list
- Each item should follow this format: {item_format}
- Number each item (1., 2., 3., etc.)

Example format:
1. [item following: {item_format}]
2. [item following: {item_format}]
...and so on"""


@dataclass
class FewShotExample:
    """Container for few-shot examples."""

    input_text: str
    output_text: str
    reasoning: Optional[str] = None


class FewShotManager:
    """Manager for few-shot examples."""

    def __init__(self):
        """Initialize the example store."""
        self.examples: Dict[str, List[FewShotExample]] = {}

    def add_example(self, task_name: str, example: FewShotExample) -> None:
        """Add an example for a specific task."""
        if task_name not in self.examples:
            self.examples[task_name] = []
        self.examples[task_name].append(example)

    def get_examples(
        self, task_name: str, num_examples: int = 3
    ) -> List[FewShotExample]:
        """Get examples for a task."""
        if task_name not in self.examples:
            return []
        return self.examples[task_name][:num_examples]

    def build_prompt(
        self, task_name: str, new_input: str, num_examples: int = 3
    ) -> str:
        """Build a complete few-shot prompt."""
        examples = self.get_examples(task_name, num_examples)

        if not examples:
            return f"Task: {task_name}\nInput: {new_input}\nOutput:"

        prompt_parts = [f"Task: {task_name}\n", "Examples:\n"]

        for i, ex in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {ex.input_text}")
            if ex.reasoning:
                prompt_parts.append(f"Reasoning: {ex.reasoning}")
            prompt_parts.append(f"Output: {ex.output_text}")
            prompt_parts.append("")

        prompt_parts.append("Now process:")
        prompt_parts.append(f"Input: {new_input}")
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)


def create_self_consistency_prompt(question: str, num_paths: int = 3) -> str:
    """Create a prompt for self-consistency checking."""
    return f"""{question}

Solve this problem {num_paths} different ways:

Approach 1:
[Work through the problem one way]

Approach 2:
[Work through the problem a different way]

Approach 3:
[Work through the problem yet another way]

After showing all {num_paths} approaches, compare your answers:
- Do they all agree?
- If not, which approach is most reliable and why?
- Provide your final, most confident answer."""


def create_verification_prompt(original_response: str, original_task: str) -> str:
    """Create a prompt to verify/check a previous response."""
    return f"""Original task: {original_task}

Response to verify:
---
{original_response}
---

Please verify this response:

1. Accuracy Check: Is the information factually correct?
2. Completeness Check: Does it fully address the original task?
3. Logic Check: Is the reasoning sound?
4. Error Check: Are there any mistakes or inconsistencies?

Verdict: Is this response correct? If not, what needs to be fixed?"""


def create_decomposition_prompt(complex_task: str) -> str:
    """Create a prompt to decompose a complex task into subtasks."""
    return f"""Complex Task: {complex_task}

Please break this down into smaller, manageable subtasks:

1. First, identify the main components or phases of this task
2. For each component, list the specific steps needed
3. Note any dependencies between steps (what must happen before what)
4. Estimate relative complexity of each subtask

Format your response as:

## Main Components
[List the major parts]

## Detailed Subtasks
### Component 1: [name]
- Step 1.1: [description]
- Step 1.2: [description]
...

### Component 2: [name]
- Step 2.1: [description]
...

## Dependencies
[What depends on what]

## Suggested Order
[Recommended sequence of execution]"""

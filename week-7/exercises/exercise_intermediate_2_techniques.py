"""
Week 7 Exercise 2 (Intermediate): Advanced Prompting Techniques
================================================================

Master Chain-of-Thought, Few-Shot, and structured output techniques.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv pydantic

Run this file:
    python exercise_intermediate_2_techniques.py

Run tests:
    python -m pytest tests/test_exercise_intermediate_2_techniques.py -v
"""

import os
import json
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

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    Field = None
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
# YOUR TASKS - Complete the functions below
# =============================================================================


def create_chain_of_thought_prompt(question: str) -> str:
    """
    Task 1: Create a Chain-of-Thought (CoT) prompt.

    CoT prompting asks the LLM to "show its work" by reasoning
    step by step before giving the final answer.

    Args:
        question: The question or problem to solve

    Returns:
        A prompt that encourages step-by-step reasoning

    Example:
        >>> prompt = create_chain_of_thought_prompt("Is 17 * 23 greater than 400?")
        >>> "step" in prompt.lower()
        True
    """
    # TODO: Create a CoT prompt that encourages step-by-step reasoning
    pass


def create_structured_cot_prompt(problem: str, reasoning_steps: List[str]) -> str:
    """
    Task 2: Create a structured CoT prompt with specific reasoning steps.

    Instead of general "think step by step", provide specific steps
    to guide the reasoning process.

    Args:
        problem: The problem to solve
        reasoning_steps: Specific steps to follow (e.g., ["Identify variables", "Set up equation"])

    Returns:
        A prompt with numbered reasoning steps

    Example:
        >>> prompt = create_structured_cot_prompt(
        ...     "Calculate the final price",
        ...     ["Find the base price", "Calculate discount", "Add tax"]
        ... )
        >>> "1." in prompt and "2." in prompt and "3." in prompt
        True
    """
    # TODO: Create structured CoT with specific steps
    pass


def create_few_shot_prompt(
    task_description: str, examples: List[Dict[str, str]], new_input: str
) -> str:
    """
    Task 3: Create a few-shot learning prompt.

    Few-shot prompting provides examples for the LLM to learn from.

    Args:
        task_description: Description of the task
        examples: List of {"input": ..., "output": ...} examples
        new_input: The new input to process

    Returns:
        A prompt with examples and the new input

    Example:
        >>> examples = [
        ...     {"input": "happy", "output": "positive"},
        ...     {"input": "sad", "output": "negative"}
        ... ]
        >>> prompt = create_few_shot_prompt("Classify emotion", examples, "excited")
        >>> "happy" in prompt and "positive" in prompt and "excited" in prompt
        True
    """
    # TODO: Create few-shot prompt with examples
    pass


def create_few_shot_with_reasoning(
    task_description: str,
    examples: List[Dict[str, str]],  # Now includes "reasoning" key
    new_input: str,
) -> str:
    """
    Task 4: Create a few-shot prompt with reasoning explanations.

    Combining few-shot with CoT by showing reasoning in examples.

    Args:
        task_description: Description of the task
        examples: List with {"input": ..., "reasoning": ..., "output": ...}
        new_input: The new input to process

    Returns:
        A prompt with examples showing reasoning

    Example:
        >>> examples = [
        ...     {"input": "2+2", "reasoning": "Adding 2 and 2 gives 4", "output": "4"},
        ... ]
        >>> prompt = create_few_shot_with_reasoning("Calculate", examples, "3+3")
        >>> "reasoning" in prompt.lower() or "Adding" in prompt
        True
    """
    # TODO: Create few-shot prompt with reasoning in examples
    pass


def create_json_output_prompt(task: str, schema: Dict[str, str]) -> str:
    """
    Task 5: Create a prompt requesting JSON output with a schema.

    Args:
        task: The task to accomplish
        schema: Dict describing expected JSON structure
                Keys are field names, values are type descriptions

    Returns:
        A prompt that requests JSON output matching the schema

    Example:
        >>> prompt = create_json_output_prompt(
        ...     "Analyze sentiment",
        ...     {"sentiment": "positive/negative/neutral", "score": "float 0-1"}
        ... )
        >>> "json" in prompt.lower() and "sentiment" in prompt
        True
    """
    # TODO: Create prompt requesting specific JSON format
    pass


def create_list_output_prompt(task: str, num_items: int, item_format: str) -> str:
    """
    Task 6: Create a prompt requesting a numbered list output.

    Args:
        task: The task to accomplish
        num_items: Exact number of items to return
        item_format: Description of each item's format

    Returns:
        A prompt requesting a specific list format

    Example:
        >>> prompt = create_list_output_prompt(
        ...     "Give Python tips",
        ...     num_items=5,
        ...     item_format="short phrase, then explanation"
        ... )
        >>> "5" in prompt and "list" in prompt.lower()
        True
    """
    # TODO: Create prompt for list output
    pass


@dataclass
class FewShotExample:
    """Container for few-shot examples."""

    input_text: str
    output_text: str
    reasoning: Optional[str] = None


class FewShotManager:
    """
    Task 7: Implement a manager for few-shot examples.

    This class should:
    - Store examples for different tasks
    - Select the best examples for a given input
    - Build few-shot prompts automatically
    """

    def __init__(self):
        """Initialize the example store."""
        # TODO: Initialize storage for examples
        # self.examples = {}  # task_name -> list of FewShotExample
        pass

    def add_example(self, task_name: str, example: FewShotExample) -> None:
        """
        Add an example for a specific task.

        Args:
            task_name: The name of the task
            example: A FewShotExample object
        """
        # TODO: Store the example
        pass

    def get_examples(
        self, task_name: str, num_examples: int = 3
    ) -> List[FewShotExample]:
        """
        Get examples for a task.

        Args:
            task_name: The name of the task
            num_examples: How many examples to return

        Returns:
            List of examples (up to num_examples)
        """
        # TODO: Return examples for the task
        pass

    def build_prompt(
        self, task_name: str, new_input: str, num_examples: int = 3
    ) -> str:
        """
        Build a complete few-shot prompt.

        Args:
            task_name: The task name
            new_input: The new input to process
            num_examples: How many examples to include

        Returns:
            A complete few-shot prompt
        """
        # TODO: Build the prompt using stored examples
        pass


def create_self_consistency_prompt(question: str, num_paths: int = 3) -> str:
    """
    Task 8: Create a prompt for self-consistency checking.

    Self-consistency asks the model to generate multiple reasoning paths
    and then choose the most consistent answer.

    Args:
        question: The question to answer
        num_paths: Number of different reasoning approaches

    Returns:
        A prompt asking for multiple approaches

    Example:
        >>> prompt = create_self_consistency_prompt("What is 15% of 80?", 3)
        >>> "3" in prompt and ("approach" in prompt.lower() or "way" in prompt.lower())
        True
    """
    # TODO: Create self-consistency prompt
    pass


def create_verification_prompt(original_response: str, original_task: str) -> str:
    """
    Task 9: Create a prompt to verify/check a previous response.

    Args:
        original_response: The response to verify
        original_task: What the original task was

    Returns:
        A prompt asking to verify the response

    Example:
        >>> prompt = create_verification_prompt(
        ...     "The answer is 42",
        ...     "Calculate 6 * 7"
        ... )
        >>> "verify" in prompt.lower() or "check" in prompt.lower()
        True
    """
    # TODO: Create verification prompt
    pass


def create_decomposition_prompt(complex_task: str) -> str:
    """
    Task 10: Create a prompt to decompose a complex task into subtasks.

    Useful for breaking down complex problems before solving them.

    Args:
        complex_task: A complex task description

    Returns:
        A prompt asking to break down the task

    Example:
        >>> prompt = create_decomposition_prompt("Build a web scraper")
        >>> "break" in prompt.lower() or "subtask" in prompt.lower() or "step" in prompt.lower()
        True
    """
    # TODO: Create task decomposition prompt
    pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 7 - Exercise 2: Advanced Prompting Techniques")
    print("=" * 60)

    # Test Task 1: Chain of Thought
    print("\n1. Testing Chain-of-Thought prompt...")
    cot = create_chain_of_thought_prompt("Is 17 * 23 greater than 400?")
    if cot:
        print("   ✓ CoT prompt created")
        print(f"   Preview: {cot[:80]}...")
    else:
        print("   ✗ create_chain_of_thought_prompt not implemented")

    # Test Task 2: Structured CoT
    print("\n2. Testing structured CoT prompt...")
    structured_cot = create_structured_cot_prompt(
        "Calculate total cost with 15% discount",
        ["Find original price", "Calculate 15% discount", "Subtract discount"],
    )
    if structured_cot:
        print("   ✓ Structured CoT created")
    else:
        print("   ✗ create_structured_cot_prompt not implemented")

    # Test Task 3: Few-shot
    print("\n3. Testing few-shot prompt...")
    examples = [
        {"input": "I love this!", "output": "positive"},
        {"input": "This is terrible", "output": "negative"},
        {"input": "It's okay", "output": "neutral"},
    ]
    few_shot = create_few_shot_prompt(
        "Classify sentiment", examples, "Amazing product!"
    )
    if few_shot:
        print("   ✓ Few-shot prompt created")
    else:
        print("   ✗ create_few_shot_prompt not implemented")

    # Test Task 4: Few-shot with reasoning
    print("\n4. Testing few-shot with reasoning...")
    reasoning_examples = [
        {
            "input": "Is 5 > 3?",
            "reasoning": "5 is greater than 3 because 5-3=2 which is positive",
            "output": "Yes",
        },
    ]
    few_shot_reasoning = create_few_shot_with_reasoning(
        "Compare numbers", reasoning_examples, "Is 7 > 10?"
    )
    if few_shot_reasoning:
        print("   ✓ Few-shot with reasoning created")
    else:
        print("   ✗ create_few_shot_with_reasoning not implemented")

    # Test Task 5: JSON output
    print("\n5. Testing JSON output prompt...")
    json_prompt = create_json_output_prompt(
        "Extract information",
        {"name": "string", "age": "integer", "occupation": "string"},
    )
    if json_prompt:
        print("   ✓ JSON output prompt created")
    else:
        print("   ✗ create_json_output_prompt not implemented")

    # Test Task 6: List output
    print("\n6. Testing list output prompt...")
    list_prompt = create_list_output_prompt(
        "Suggest books",
        num_items=5,
        item_format="Title by Author - one sentence description",
    )
    if list_prompt:
        print("   ✓ List output prompt created")
    else:
        print("   ✗ create_list_output_prompt not implemented")

    # Test Task 7: Few-shot manager
    print("\n7. Testing FewShotManager...")
    manager = FewShotManager()
    try:
        manager.add_example("sentiment", FewShotExample("Great!", "positive"))
        manager.add_example("sentiment", FewShotExample("Awful", "negative"))
        examples = manager.get_examples("sentiment", 2)
        if examples and len(examples) == 2:
            print("   ✓ FewShotManager working")
        else:
            print("   ✗ FewShotManager not fully implemented")
    except (AttributeError, TypeError):
        print("   ✗ FewShotManager not implemented")

    # Test Task 8: Self-consistency
    print("\n8. Testing self-consistency prompt...")
    consistency = create_self_consistency_prompt("What is 25% of 80?", 3)
    if consistency:
        print("   ✓ Self-consistency prompt created")
    else:
        print("   ✗ create_self_consistency_prompt not implemented")

    # Test Task 9: Verification
    print("\n9. Testing verification prompt...")
    verification = create_verification_prompt(
        "The population of France is 67 million", "What is France's population?"
    )
    if verification:
        print("   ✓ Verification prompt created")
    else:
        print("   ✗ create_verification_prompt not implemented")

    # Test Task 10: Decomposition
    print("\n10. Testing decomposition prompt...")
    decomp = create_decomposition_prompt("Build a recommendation system for movies")
    if decomp:
        print("   ✓ Decomposition prompt created")
    else:
        print("   ✗ create_decomposition_prompt not implemented")

    print("\n" + "=" * 60)
    print(
        "Run tests: python -m pytest tests/test_exercise_intermediate_2_techniques.py -v"
    )
    print("=" * 60)

"""
Week 7 Exercise 1 (Basic): Prompt Engineering Fundamentals
==========================================================

Learn the core principles of effective prompt engineering.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv

Run this file:
    python exercise_basic_1_fundamentals.py

Run tests:
    python -m pytest tests/test_exercise_basic_1_fundamentals.py -v
"""

import os
import json
from typing import Dict, List, Optional

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
# YOUR TASKS - Complete the functions below
# =============================================================================


def craft_prompt(
    context: str, role: str, action: str, format_spec: str, tone: str
) -> str:
    """
    Task 1: Build a prompt using the CRAFT framework.

    CRAFT stands for:
    - Context: Background information
    - Role: Who the AI should act as
    - Action: The specific task
    - Format: How to structure the output
    - Tone: The communication style

    Args:
        context: Background information for the task
        role: The role/persona for the AI
        action: The specific task to accomplish
        format_spec: How to format the output
        tone: The tone/style of communication

    Returns:
        A well-structured prompt string combining all elements

    Example:
        >>> prompt = craft_prompt(
        ...     context="I'm building a web application",
        ...     role="Senior software architect",
        ...     action="Review my database schema",
        ...     format_spec="Numbered list of suggestions",
        ...     tone="Professional and constructive"
        ... )
        >>> "Context:" in prompt and "Role:" in prompt
        True
    """
    # TODO: Implement CRAFT prompt builder
    # Combine all elements into a clear, structured prompt
    pass


def create_specific_prompt(vague_request: str) -> str:
    """
    Task 2: Transform a vague request into a specific prompt.

    Take an unclear/vague request and make it specific by adding:
    - Clear task definition
    - Output format requirements
    - Constraints (length, style, etc.)
    - Context if needed

    Args:
        vague_request: A vague/unclear request like "write about Python"

    Returns:
        A specific, well-structured prompt

    Example:
        >>> prompt = create_specific_prompt("write about Python")
        >>> len(prompt) > len("write about Python")
        True
        >>> any(word in prompt.lower() for word in ["format", "requirements", "specific"])
        True
    """
    # TODO: Transform vague request into specific prompt
    # Add specificity, format requirements, and constraints
    pass


def add_constraints(base_prompt: str, constraints: Dict[str, any]) -> str:
    """
    Task 3: Add constraints to an existing prompt.

    Common constraints include:
    - max_words: Maximum word count
    - format: Output format (json, markdown, plain)
    - language: Programming language if applicable
    - audience: Target audience level
    - tone: Communication tone

    Args:
        base_prompt: The original prompt
        constraints: Dict of constraints to apply

    Returns:
        The prompt with constraints added

    Example:
        >>> prompt = add_constraints(
        ...     "Explain recursion",
        ...     {"max_words": 100, "audience": "beginners", "format": "markdown"}
        ... )
        >>> "100 words" in prompt or "100" in prompt
        True
    """
    # TODO: Add constraints section to the prompt
    pass


def create_role_prompt(role: str, expertise: str, task: str) -> str:
    """
    Task 4: Create a prompt with a specific role/persona.

    Role prompting helps get more accurate, contextual responses by
    having the AI "act as" a specific expert.

    Args:
        role: The job title/role (e.g., "Senior Python Developer")
        expertise: Specific expertise (e.g., "10 years in backend development")
        task: The task to accomplish

    Returns:
        A prompt with clear role definition

    Example:
        >>> prompt = create_role_prompt(
        ...     "Security Expert",
        ...     "specializing in web application security",
        ...     "Review this authentication code"
        ... )
        >>> "Security Expert" in prompt
        True
    """
    # TODO: Create a role-based prompt
    pass


def create_step_by_step_prompt(task: str, steps: List[str]) -> str:
    """
    Task 5: Create a prompt that guides through specific steps.

    Breaking tasks into explicit steps helps get more thorough responses.

    Args:
        task: The overall task
        steps: List of specific steps to follow

    Returns:
        A prompt with numbered steps

    Example:
        >>> prompt = create_step_by_step_prompt(
        ...     "Analyze this code",
        ...     ["Check for bugs", "Review performance", "Suggest improvements"]
        ... )
        >>> "1." in prompt and "2." in prompt
        True
    """
    # TODO: Create a step-by-step prompt
    pass


def create_example_based_prompt(
    task: str, example_input: str, example_output: str
) -> str:
    """
    Task 6: Create a prompt with a single example (one-shot).

    Showing one example helps the LLM understand the expected format.

    Args:
        task: The task description
        example_input: An example input
        example_output: The expected output for that input

    Returns:
        A prompt with the task and one example

    Example:
        >>> prompt = create_example_based_prompt(
        ...     "Classify sentiment",
        ...     "I love this product!",
        ...     "positive"
        ... )
        >>> "I love this product!" in prompt and "positive" in prompt
        True
    """
    # TODO: Create a one-shot prompt with example
    pass


def create_output_format_prompt(task: str, format_type: str, fields: List[str]) -> str:
    """
    Task 7: Create a prompt that specifies exact output format.

    Args:
        task: The task to accomplish
        format_type: Type of format ("json", "markdown", "csv", "list")
        fields: List of fields/sections to include

    Returns:
        A prompt with explicit format instructions

    Example:
        >>> prompt = create_output_format_prompt(
        ...     "Analyze customer feedback",
        ...     "json",
        ...     ["sentiment", "key_points", "action_items"]
        ... )
        >>> "json" in prompt.lower() and "sentiment" in prompt
        True
    """
    # TODO: Create a prompt with format specification
    pass


def create_negative_prompt(task: str, things_to_avoid: List[str]) -> str:
    """
    Task 8: Create a prompt that specifies what NOT to do.

    Negative prompting helps avoid common unwanted behaviors.

    Args:
        task: The task to accomplish
        things_to_avoid: List of things the AI should NOT do

    Returns:
        A prompt with clear "do not" instructions

    Example:
        >>> prompt = create_negative_prompt(
        ...     "Write a product description",
        ...     ["Don't use superlatives", "Don't make claims without evidence"]
        ... )
        >>> "Don't" in prompt or "do not" in prompt.lower()
        True
    """
    # TODO: Create a prompt with negative constraints
    pass


def create_iterative_prompt(initial_output: str, feedback: str) -> str:
    """
    Task 9: Create a prompt for iterating/improving on previous output.

    This is useful for refining outputs in a conversation.

    Args:
        initial_output: The previous response to improve
        feedback: Specific feedback on what to change

    Returns:
        A prompt asking for improvements based on feedback

    Example:
        >>> prompt = create_iterative_prompt(
        ...     "Python is a programming language.",
        ...     "Make it more detailed and add examples"
        ... )
        >>> "Python is a programming language" in prompt
        True
    """
    # TODO: Create an iteration/improvement prompt
    pass


def analyze_prompt_quality(prompt: str) -> Dict[str, any]:
    """
    Task 10: Analyze a prompt and return quality metrics.

    Check for:
    - has_context: Does it provide context?
    - has_format: Does it specify output format?
    - has_constraints: Does it have constraints (length, style)?
    - has_examples: Does it include examples?
    - specificity_score: Rate from 1-5 how specific it is
    - suggestions: List of improvement suggestions

    Args:
        prompt: The prompt to analyze

    Returns:
        Dict with quality analysis

    Example:
        >>> analysis = analyze_prompt_quality("Write about Python")
        >>> analysis["specificity_score"] < 3
        True
        >>> len(analysis["suggestions"]) > 0
        True
    """
    # TODO: Analyze prompt quality and return metrics
    pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 7 - Exercise 1: Prompt Engineering Fundamentals")
    print("=" * 60)

    # Test Task 1: CRAFT prompt
    print("\n1. Testing CRAFT prompt builder...")
    craft_result = craft_prompt(
        context="Building an e-commerce website",
        role="Senior UX designer",
        action="Review my checkout flow design",
        format_spec="Bullet points with priorities",
        tone="Friendly but professional",
    )
    if craft_result:
        print("   ✓ CRAFT prompt created")
        print(f"   Preview: {craft_result[:100]}...")
    else:
        print("   ✗ craft_prompt not implemented")

    # Test Task 2: Specific prompt
    print("\n2. Testing specific prompt creation...")
    specific = create_specific_prompt("write about databases")
    if specific:
        print("   ✓ Specific prompt created")
        print(f"   Length increased from 21 to {len(specific)} chars")
    else:
        print("   ✗ create_specific_prompt not implemented")

    # Test Task 3: Constraints
    print("\n3. Testing constraints addition...")
    constrained = add_constraints(
        "Explain machine learning",
        {
            "max_words": 150,
            "audience": "business executives",
            "format": "bullet points",
        },
    )
    if constrained:
        print("   ✓ Constraints added")
    else:
        print("   ✗ add_constraints not implemented")

    # Test Task 4: Role prompt
    print("\n4. Testing role prompt...")
    role_prompt = create_role_prompt(
        "Data Scientist",
        "PhD in statistics with 8 years in ML",
        "Explain overfitting to my team",
    )
    if role_prompt:
        print("   ✓ Role prompt created")
    else:
        print("   ✗ create_role_prompt not implemented")

    # Test Task 5: Step by step
    print("\n5. Testing step-by-step prompt...")
    steps_prompt = create_step_by_step_prompt(
        "Debug this Python function",
        ["Read the code", "Identify the bug", "Explain the fix", "Show corrected code"],
    )
    if steps_prompt:
        print("   ✓ Step-by-step prompt created")
    else:
        print("   ✗ create_step_by_step_prompt not implemented")

    # Test Task 6: Example-based
    print("\n6. Testing example-based prompt...")
    example_prompt = create_example_based_prompt(
        "Convert to past tense", "I walk to the store", "I walked to the store"
    )
    if example_prompt:
        print("   ✓ Example-based prompt created")
    else:
        print("   ✗ create_example_based_prompt not implemented")

    # Test Task 7: Output format
    print("\n7. Testing output format prompt...")
    format_prompt = create_output_format_prompt(
        "Analyze this review", "json", ["sentiment", "confidence", "key_themes"]
    )
    if format_prompt:
        print("   ✓ Output format prompt created")
    else:
        print("   ✗ create_output_format_prompt not implemented")

    # Test Task 8: Negative prompt
    print("\n8. Testing negative prompt...")
    negative = create_negative_prompt(
        "Write a news summary",
        ["Don't include opinions", "Don't use sensational language"],
    )
    if negative:
        print("   ✓ Negative prompt created")
    else:
        print("   ✗ create_negative_prompt not implemented")

    # Test Task 9: Iterative prompt
    print("\n9. Testing iterative prompt...")
    iterative = create_iterative_prompt(
        "Machine learning is a subset of AI.", "Add more detail and practical examples"
    )
    if iterative:
        print("   ✓ Iterative prompt created")
    else:
        print("   ✗ create_iterative_prompt not implemented")

    # Test Task 10: Quality analysis
    print("\n10. Testing prompt quality analysis...")
    analysis = analyze_prompt_quality("tell me about python")
    if analysis:
        print(
            f"   ✓ Analysis complete: specificity={analysis.get('specificity_score', 'N/A')}"
        )
        print(f"   Suggestions: {analysis.get('suggestions', [])[:2]}")
    else:
        print("   ✗ analyze_prompt_quality not implemented")

    print("\n" + "=" * 60)
    print("Run tests: python -m pytest tests/test_exercise_basic_1_fundamentals.py -v")
    print("=" * 60)

"""
Week 5 Exercise 1 (Basic): Prompt Engineering
=============================================

Learn to write effective prompts for LLM APIs.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv

Run this file:
    python exercise_basic_1_prompts.py

Run tests:
    python -m pytest tests/test_exercise_basic_1_prompts.py -v
"""

import os
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
# SETUP - This part is provided for you
# =============================================================================

def setup_gemini():
    """Set up Gemini API client. Returns None if API key not found."""
    if not GENAI_AVAILABLE:
        print("Please install: uv add google-generativeai python-dotenv")
        return None
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found in environment")
        return None
        
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# =============================================================================
# YOUR TASKS - Complete the functions below
# =============================================================================


def basic_prompt(topic: str) -> str:
    """
    Task 1: Create a basic prompt asking for an explanation of a topic.
    
    Args:
        topic: The topic to explain (e.g., "recursion", "APIs")
    
    Returns:
        A prompt string that asks for a clear explanation
        
    Requirements:
        - Ask for a beginner-friendly explanation
        - Request an example
        - Limit response to 100 words
    """
    # TODO: Implement
    pass


def role_based_prompt(role: str, task: str) -> str:
    """
    Task 2: Create a prompt that assigns a specific role to the AI.
    
    Args:
        role: The expert role (e.g., "Python developer", "data scientist")
        task: The task to perform (e.g., "review this code", "explain this concept")
    
    Returns:
        A prompt string with role assignment
        
    Example:
        role = "senior Python developer"
        task = "explain list comprehensions"
        
        Should return something like:
        "You are a senior Python developer. Please explain list comprehensions..."
    """
    # TODO: Implement
    pass


def few_shot_prompt(examples: List[Dict[str, str]], new_input: str) -> str:
    """
    Task 3: Create a few-shot prompt with examples.
    
    Few-shot prompting shows the model examples of desired input/output pairs.
    
    Args:
        examples: List of dicts with 'input' and 'output' keys
        new_input: The new input to process
    
    Returns:
        A prompt string with examples followed by the new input
        
    Example:
        examples = [
            {"input": "happy", "output": "POSITIVE"},
            {"input": "sad", "output": "NEGATIVE"},
        ]
        new_input = "excited"
        
        Should format like:
        "Input: happy
        Output: POSITIVE
        
        Input: sad
        Output: NEGATIVE
        
        Input: excited
        Output:"
    """
    # TODO: Implement
    pass


def structured_output_prompt(data: str, fields: List[str]) -> str:
    """
    Task 4: Create a prompt requesting structured JSON output.
    
    Args:
        data: The text data to analyze
        fields: List of field names to extract (e.g., ["name", "date", "amount"])
    
    Returns:
        A prompt that requests JSON output with specific fields
        
    Requirements:
        - Clearly request JSON format
        - List all required fields
        - Provide the data to analyze
    """
    # TODO: Implement
    pass


def chain_of_thought_prompt(problem: str) -> str:
    """
    Task 5: Create a chain-of-thought prompt for complex reasoning.
    
    Chain-of-thought prompting asks the model to show its reasoning step by step.
    
    Args:
        problem: A complex problem or question
    
    Returns:
        A prompt that encourages step-by-step reasoning
        
    Requirements:
        - Ask the model to think step by step
        - Request showing work/reasoning
        - Ask for a final answer after the reasoning
    """
    # TODO: Implement
    pass


def constrained_prompt(task: str, constraints: Dict[str, str]) -> str:
    """
    Task 6: Create a prompt with specific constraints.
    
    Args:
        task: The main task to perform
        constraints: Dict of constraint name to value
                    e.g., {"max_words": "50", "tone": "professional", "format": "bullet points"}
    
    Returns:
        A prompt string with all constraints clearly stated
    """
    # TODO: Implement
    pass


def code_generation_prompt(
    language: str,
    task_description: str,
    requirements: List[str],
) -> str:
    """
    Task 7: Create a prompt for code generation.
    
    Args:
        language: Programming language (e.g., "Python", "JavaScript")
        task_description: What the code should do
        requirements: List of specific requirements
    
    Returns:
        A well-structured prompt for generating code
        
    Requirements:
        - Specify the language
        - Describe the task clearly
        - List all requirements
        - Ask for comments in the code
        - Request error handling
    """
    # TODO: Implement
    pass


def comparison_prompt(items: List[str], criteria: List[str]) -> str:
    """
    Task 8: Create a prompt for comparing multiple items.
    
    Args:
        items: List of items to compare (e.g., ["Python", "JavaScript", "Go"])
        criteria: List of comparison criteria (e.g., ["speed", "ease of learning"])
    
    Returns:
        A prompt asking for a structured comparison
        
    Requirements:
        - List all items and criteria
        - Request a table or structured format
        - Ask for a summary recommendation
    """
    # TODO: Implement
    pass


def refinement_prompt(original_text: str, feedback: str) -> str:
    """
    Task 9: Create a prompt for refining/improving text based on feedback.
    
    Args:
        original_text: The text to improve
        feedback: Specific feedback about what to change
    
    Returns:
        A prompt asking for improvements while maintaining context
    """
    # TODO: Implement
    pass


def safety_prompt(user_input: str) -> str:
    """
    Task 10: Create a prompt with safety guardrails.
    
    Args:
        user_input: Raw user input that might need sanitization
    
    Returns:
        A prompt with safety instructions to prevent prompt injection
        
    Requirements:
        - Include instructions to stay on topic
        - Ask model to refuse harmful requests
        - Clearly separate system instructions from user input
    """
    # TODO: Implement
    pass


# =============================================================================
# BONUS: API Integration
# =============================================================================


def test_prompt_with_api(prompt: str) -> Optional[str]:
    """
    Bonus: Test a prompt with the actual Gemini API.
    
    Args:
        prompt: The prompt to send
    
    Returns:
        The model's response text, or None if API unavailable
    """
    model = setup_gemini()
    if model is None:
        return None
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return None


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Exercise 1: Prompt Engineering")
    print("=" * 60)
    
    print("\n1. Basic prompt...")
    prompt = basic_prompt("machine learning")
    if prompt:
        print(f"   Prompt: {prompt[:100]}...")
    
    print("\n2. Role-based prompt...")
    prompt = role_based_prompt("data scientist", "explain feature engineering")
    if prompt:
        print(f"   Prompt: {prompt[:100]}...")
    
    print("\n3. Few-shot prompt...")
    examples = [
        {"input": "The movie was great!", "output": "POSITIVE"},
        {"input": "Terrible service.", "output": "NEGATIVE"},
    ]
    prompt = few_shot_prompt(examples, "I love this product!")
    if prompt:
        print(f"   Prompt:\n{prompt}")
    
    print("\n4. Structured output prompt...")
    data = "Meeting with John on March 15th to discuss the $50,000 contract."
    fields = ["person", "date", "amount"]
    prompt = structured_output_prompt(data, fields)
    if prompt:
        print(f"   Prompt: {prompt[:150]}...")
    
    print("\n5. Chain-of-thought prompt...")
    problem = "If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire trip?"
    prompt = chain_of_thought_prompt(problem)
    if prompt:
        print(f"   Prompt: {prompt[:150]}...")
    
    # Test with API if available
    print("\n" + "=" * 60)
    print("Testing with Gemini API (if available)...")
    print("=" * 60)
    
    test_prompt = basic_prompt("APIs")
    if test_prompt:
        response = test_prompt_with_api(test_prompt)
        if response:
            print(f"\nAPI Response:\n{response[:500]}...")
        else:
            print("\nAPI not available. Complete the prompts and test manually!")
    
    print("\nComplete all TODOs and run tests to verify!")

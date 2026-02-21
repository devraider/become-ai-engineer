"""
Week 5 Exercise 3 (Advanced): Tool Use & Function Calling
=========================================================

Implement function calling to let LLMs interact with external systems.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv pydantic

Run this file:
    python exercise_advanced_3_tools.py

Run tests:
    python -m pytest tests/test_exercise_advanced_3_tools.py -v
"""

import os
import json
import re
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

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
    return genai


# =============================================================================
# SAMPLE TOOLS (These simulate external services)
# =============================================================================


def get_weather(location: str, unit: str = "celsius") -> Dict:
    """
    Simulated weather API.

    Args:
        location: City name
        unit: "celsius" or "fahrenheit"

    Returns:
        Weather data dictionary
    """
    # Simulated data
    weather_data = {
        "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45},
        "london": {"temp_c": 15, "condition": "cloudy", "humidity": 78},
        "tokyo": {"temp_c": 28, "condition": "partly cloudy", "humidity": 65},
        "paris": {"temp_c": 18, "condition": "rainy", "humidity": 82},
    }

    data = weather_data.get(
        location.lower(), {"temp_c": 20, "condition": "unknown", "humidity": 50}
    )

    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"],
    }


def search_database(query: str, limit: int = 5) -> List[Dict]:
    """
    Simulated database search.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of matching records
    """
    # Simulated product database
    products = [
        {"id": 1, "name": "Laptop Pro", "price": 999, "category": "electronics"},
        {"id": 2, "name": "Wireless Mouse", "price": 49, "category": "electronics"},
        {"id": 3, "name": "Python Book", "price": 35, "category": "books"},
        {"id": 4, "name": "Standing Desk", "price": 599, "category": "furniture"},
        {
            "id": 5,
            "name": "Mechanical Keyboard",
            "price": 150,
            "category": "electronics",
        },
    ]

    query_lower = query.lower()
    results = [
        p
        for p in products
        if query_lower in p["name"].lower() or query_lower in p["category"].lower()
    ]

    return results[:limit]


def calculate_math(expression: str) -> Dict:
    """
    Safe math calculator.

    Args:
        expression: Mathematical expression (e.g., "2 + 2 * 3")

    Returns:
        Result dictionary
    """
    # Safe evaluation of math expressions
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return {"error": "Invalid characters in expression", "expression": expression}

    try:
        result = eval(expression)  # Safe because we validated input
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def send_email(to: str, subject: str, body: str) -> Dict:
    """
    Simulated email sender.

    Args:
        to: Recipient email
        subject: Email subject
        body: Email body

    Returns:
        Status dictionary
    """
    # Simulated - just returns success
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": "msg_12345",
    }


# =============================================================================
# YOUR TASKS - Complete the classes and functions below
# =============================================================================


@dataclass
class ToolDefinition:
    """
    Task 1: Define the structure for a tool/function definition.

    This represents a function that can be called by the LLM.
    """

    name: str
    description: str
    parameters: Dict[
        str, Dict
    ]  # {"param_name": {"type": "string", "description": "..."}}
    required_params: List[str]
    function: Callable

    def to_gemini_format(self) -> Dict:
        """
        Convert to Gemini function declaration format.

        Returns:
            Dictionary in Gemini's expected format
        """
        # TODO: Implement
        pass

    def to_openai_format(self) -> Dict:
        """
        Convert to OpenAI function declaration format.

        Returns:
            Dictionary in OpenAI's expected format
        """
        # TODO: Implement
        pass


class ToolRegistry:
    """
    Task 2: Implement a registry to manage available tools.

    This allows dynamic registration and lookup of tools.
    """

    def __init__(self):
        """Initialize the registry."""
        # TODO: Initialize attributes
        pass

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict],
        required_params: List[str],
        function: Callable,
    ) -> None:
        """
        Register a new tool.

        Args:
            name: Tool name
            description: What the tool does
            parameters: Parameter definitions
            required_params: List of required parameter names
            function: The actual function to call
        """
        # TODO: Implement
        pass

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        # TODO: Implement
        pass

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        # TODO: Implement
        pass

    def get_all_definitions(self, format_type: str = "gemini") -> List[Dict]:
        """
        Get all tool definitions in API format.

        Args:
            format_type: "gemini" or "openai"

        Returns:
            List of tool definitions
        """
        # TODO: Implement
        pass


def execute_tool_call(
    registry: ToolRegistry,
    tool_name: str,
    arguments: Dict,
) -> Dict:
    """
    Task 3: Execute a tool call from the LLM.

    Args:
        registry: Tool registry
        tool_name: Name of tool to call
        arguments: Arguments from LLM

    Returns:
        Result dictionary with 'success' and 'result' or 'error'
    """
    # TODO: Implement with error handling
    pass


def parse_function_call(response_text: str) -> Optional[Dict]:
    """
    Task 4: Parse a function call from LLM text response.

    Some APIs return function calls as JSON in text. Parse it.

    Args:
        response_text: Raw response text that might contain a function call

    Returns:
        Dictionary with 'name' and 'arguments' if found, None otherwise

    Example input:
        "I'll help you with that. <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>"

    Example output:
        {"name": "get_weather", "arguments": {"location": "Paris"}}
    """
    # TODO: Implement
    pass


class ToolAgent:
    """
    Task 5: Implement an agent that can use tools to answer questions.

    This agent:
    1. Receives a user question
    2. Decides if a tool is needed
    3. Calls the tool if needed
    4. Formulates a response
    """

    def __init__(self, registry: ToolRegistry, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the agent.

        Args:
            registry: Tool registry with available tools
            model_name: LLM model to use
        """
        # TODO: Initialize attributes
        pass

    def _should_use_tool(self, question: str) -> Optional[str]:
        """
        Determine if a tool should be used and which one.

        Args:
            question: User's question

        Returns:
            Tool name if a tool should be used, None otherwise

        Hint: Use keyword matching or another heuristic
        """
        # TODO: Implement
        pass

    def _generate_tool_arguments(self, tool_name: str, question: str) -> Dict:
        """
        Generate arguments for a tool call based on the question.

        Args:
            tool_name: Tool to call
            question: User's question

        Returns:
            Arguments dictionary
        """
        # TODO: Implement
        pass

    def process_question(self, question: str) -> str:
        """
        Process a user question, potentially using tools.

        Args:
            question: User's question

        Returns:
            Response string
        """
        # TODO: Implement the full flow
        pass


def validate_tool_arguments(
    tool: ToolDefinition,
    arguments: Dict,
) -> Tuple[bool, Optional[str]]:
    """
    Task 6: Validate that arguments match tool requirements.

    Args:
        tool: Tool definition
        arguments: Provided arguments

    Returns:
        Tuple of (is_valid, error_message)
    """
    # TODO: Implement
    pass


def create_tool_prompt(
    question: str,
    available_tools: List[ToolDefinition],
) -> str:
    """
    Task 7: Create a prompt that tells the LLM about available tools.

    Args:
        question: User's question
        available_tools: List of available tools

    Returns:
        Prompt string with tool descriptions
    """
    # TODO: Implement
    pass


# Need to import Tuple for type hints
from typing import Tuple


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Exercise 3: Tool Use & Function Calling")
    print("=" * 60)

    print("\n1. Setting up tool registry...")
    registry = ToolRegistry()

    # Register tools
    registry.register(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "description": "celsius or fahrenheit"},
        },
        required_params=["location"],
        function=get_weather,
    )

    registry.register(
        name="search_database",
        description="Search the product database",
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        required_params=["query"],
        function=search_database,
    )

    registry.register(
        name="calculate",
        description="Calculate a math expression",
        parameters={
            "expression": {"type": "string", "description": "Math expression"},
        },
        required_params=["expression"],
        function=calculate_math,
    )

    tools = registry.list_tools()
    if tools:
        print(f"   Registered tools: {tools}")

    print("\n2. Testing tool execution...")
    result = execute_tool_call(registry, "get_weather", {"location": "Tokyo"})
    if result:
        print(f"   Weather result: {result}")

    result = execute_tool_call(registry, "calculate", {"expression": "2 + 2 * 3"})
    if result:
        print(f"   Math result: {result}")

    print("\n3. Testing tool definitions...")
    tool = registry.get_tool("get_weather")
    if tool:
        gemini_format = tool.to_gemini_format()
        if gemini_format:
            print(f"   Gemini format: {json.dumps(gemini_format, indent=2)[:200]}...")

    print("\n4. Testing function call parsing...")
    test_response = 'Let me check that. <function_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</function_call>'
    parsed = parse_function_call(test_response)
    if parsed:
        print(f"   Parsed function call: {parsed}")

    print("\n5. Testing ToolAgent...")
    agent = ToolAgent(registry)

    questions = [
        "What's the weather in London?",
        "Calculate 15 * 7 + 23",
        "Search for electronics products",
    ]

    for q in questions:
        response = agent.process_question(q)
        if response:
            print(f"   Q: {q}")
            print(f"   A: {response[:100]}...")

    print("\nComplete all TODOs and run tests to verify!")

"""
Week 5 Exercise 3 (Advanced): Tool Use & Function Calling - SOLUTIONS
====================================================================
"""

import os
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# Sample tools
def get_weather(location: str, unit: str = "celsius") -> Dict:
    """Simulated weather API."""
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
    """Simulated database search."""
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
    """Safe math calculator."""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return {"error": "Invalid characters in expression", "expression": expression}

    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@dataclass
class ToolDefinition:
    """Structure for a tool/function definition."""

    name: str
    description: str
    parameters: Dict[str, Dict]
    required_params: List[str]
    function: Callable

    def to_gemini_format(self) -> Dict:
        """Convert to Gemini function declaration format."""
        properties = {}
        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": self.required_params,
            },
        }

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI function declaration format."""
        properties = {}
        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_params,
                },
            },
        }


class ToolRegistry:
    """Registry to manage available tools."""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict],
        required_params: List[str],
        function: Callable,
    ) -> None:
        """Register a new tool."""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required_params=required_params,
            function=function,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_all_definitions(self, format_type: str = "gemini") -> List[Dict]:
        """Get all tool definitions in API format."""
        if format_type == "gemini":
            return [tool.to_gemini_format() for tool in self.tools.values()]
        return [tool.to_openai_format() for tool in self.tools.values()]


def execute_tool_call(
    registry: ToolRegistry,
    tool_name: str,
    arguments: Dict,
) -> Dict:
    """Execute a tool call from the LLM."""
    tool = registry.get_tool(tool_name)

    if tool is None:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}

    try:
        result = tool.function(**arguments)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_function_call(response_text: str) -> Optional[Dict]:
    """Parse a function call from LLM text response."""
    # Try to find function call in tags
    pattern = r"<function_call>(.*?)</function_call>"
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON with function call structure
    json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group())
            if "name" in data and "arguments" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


class ToolAgent:
    """Agent that can use tools to answer questions."""

    def __init__(self, registry: ToolRegistry, model_name: str = "gemini-1.5-flash"):
        self.registry = registry
        self.model_name = model_name
        self.model = None

        if GENAI_AVAILABLE:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)

    def _should_use_tool(self, question: str) -> Optional[str]:
        """Determine if a tool should be used and which one."""
        question_lower = question.lower()

        keywords = {
            "get_weather": [
                "weather",
                "temperature",
                "forecast",
                "hot",
                "cold",
                "rain",
            ],
            "search_database": ["search", "find", "product", "database", "look up"],
            "calculate": [
                "calculate",
                "math",
                "compute",
                "+",
                "-",
                "*",
                "/",
                "sum",
                "multiply",
            ],
        }

        for tool_name, tool_keywords in keywords.items():
            if any(kw in question_lower for kw in tool_keywords):
                if self.registry.get_tool(tool_name):
                    return tool_name

        return None

    def _generate_tool_arguments(self, tool_name: str, question: str) -> Dict:
        """Generate arguments for a tool call based on the question."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {}

        args = {}
        question_lower = question.lower()

        if tool_name == "get_weather":
            # Extract location
            cities = ["london", "paris", "tokyo", "new york", "san francisco"]
            for city in cities:
                if city in question_lower:
                    args["location"] = city.title()
                    break
            else:
                args["location"] = "Unknown"
            args["unit"] = "celsius"

        elif tool_name == "calculate":
            # Extract expression
            patterns = [
                r"(\d+[\s]*[\+\-\*\/][\s]*\d+)",
                r"calculate\s+(.+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, question_lower)
                if match:
                    args["expression"] = match.group(1).strip()
                    break

        elif tool_name == "search_database":
            words = question_lower.split()
            search_terms = ["electronics", "books", "laptop", "keyboard", "mouse"]
            for term in search_terms:
                if term in question_lower:
                    args["query"] = term
                    break
            else:
                args["query"] = words[-1] if words else "product"
            args["limit"] = 5

        return args

    def process_question(self, question: str) -> str:
        """Process a user question, potentially using tools."""
        tool_name = self._should_use_tool(question)

        if tool_name:
            args = self._generate_tool_arguments(tool_name, question)
            result = execute_tool_call(self.registry, tool_name, args)

            if result.get("success"):
                return (
                    f"Result from {tool_name}: {json.dumps(result['result'], indent=2)}"
                )
            else:
                return f"Error using {tool_name}: {result.get('error')}"

        return "I couldn't determine which tool to use for your question."


def validate_tool_arguments(
    tool: ToolDefinition,
    arguments: Dict,
) -> Tuple[bool, Optional[str]]:
    """Validate that arguments match tool requirements."""
    # Check required parameters
    for required in tool.required_params:
        if required not in arguments:
            return False, f"Missing required parameter: {required}"

    # Check parameter types (basic validation)
    for param_name, value in arguments.items():
        if param_name in tool.parameters:
            expected_type = tool.parameters[param_name].get("type")

            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter {param_name} should be a string"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"Parameter {param_name} should be an integer"

    return True, None


def create_tool_prompt(
    question: str,
    available_tools: List[ToolDefinition],
) -> str:
    """Create a prompt that tells the LLM about available tools."""
    tools_desc = []
    for tool in available_tools:
        params = ", ".join(
            f"{k}: {v.get('description', '')}" for k, v in tool.parameters.items()
        )
        tools_desc.append(f"- {tool.name}: {tool.description}\n  Parameters: {params}")

    tools_text = "\n".join(tools_desc)

    return f"""You have access to the following tools:

{tools_text}

Question: {question}

If you need to use a tool, respond with:
<function_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</function_call>

Otherwise, answer the question directly."""

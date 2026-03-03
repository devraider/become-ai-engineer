"""
Solutions for Week 10 - Exercise 2: Conditional Routing & Cycles
================================================================
"""

from typing import TypedDict, Literal
from enum import Enum
import re


# =============================================================================
# TASK 1: Define Routing State
# =============================================================================
class RoutingState(TypedDict):
    """State for routing decisions."""

    query: str
    category: str
    response: str
    confidence: float


# =============================================================================
# TASK 2: Implement Query Classifier
# =============================================================================
def classify_query(state: RoutingState) -> dict:
    """Classify the query into categories."""
    query = state["query"].lower()

    # Check for math
    math_indicators = [
        "+",
        "-",
        "*",
        "/",
        "=",
        "calculate",
        "compute",
        "sum",
        "multiply",
    ]
    math_count = sum(1 for ind in math_indicators if ind in query)

    # Check for weather
    weather_indicators = [
        "weather",
        "temperature",
        "rain",
        "sunny",
        "cloudy",
        "forecast",
    ]
    weather_count = sum(1 for ind in weather_indicators if ind in query)

    # Check for code
    code_indicators = ["function", "class", "code", "python", "error", "bug", "program"]
    code_count = sum(1 for ind in code_indicators if ind in query)

    # Determine category and confidence
    counts = {"math": math_count, "weather": weather_count, "code": code_count}

    max_category = max(counts, key=counts.get)
    max_count = counts[max_category]

    if max_count == 0:
        return {"category": "general", "confidence": 0.5}
    elif max_count >= 2:
        return {"category": max_category, "confidence": 0.9}
    else:
        return {"category": max_category, "confidence": 0.7}


# =============================================================================
# TASK 3: Create Route Function
# =============================================================================
def route_by_category(state: RoutingState) -> str:
    """Return next node name based on category."""
    mapping = {
        "math": "math_handler",
        "weather": "weather_handler",
        "code": "code_handler",
        "general": "general_handler",
    }
    return mapping.get(state["category"], "general_handler")


# =============================================================================
# TASK 4: Implement Handler Nodes
# =============================================================================
def math_handler(state: RoutingState) -> dict:
    """Handle math queries."""
    query = state["query"]
    try:
        # Try to find and evaluate expression
        expr_match = re.search(r"[\d\s\+\-\*/\(\)\.]+", query)
        if expr_match:
            expr = expr_match.group().strip()
            if expr and any(c.isdigit() for c in expr):
                result = eval(expr)
                return {"response": f"The result is: {result}"}
    except:
        pass
    return {
        "response": "I can help with math calculations. Please provide a clear expression like '2 + 3 * 4'."
    }


def weather_handler(state: RoutingState) -> dict:
    """Handle weather queries."""
    return {
        "response": "Currently, it's partly cloudy with a temperature of 72°F (22°C). Expect clear skies later today."
    }


def code_handler(state: RoutingState) -> dict:
    """Handle code queries."""
    return {
        "response": "I can help with coding questions. Please describe your problem or share your code, and I'll assist you."
    }


def general_handler(state: RoutingState) -> dict:
    """Handle general queries."""
    return {
        "response": "I'm here to help! Could you provide more details about what you'd like to know?"
    }


# =============================================================================
# TASK 5: Build Conditional Router Graph
# =============================================================================
class ConditionalRouterGraph:
    """Graph with conditional routing based on classification."""

    def __init__(self):
        self.classifier = classify_query
        self.router = route_by_category
        self.handlers = {
            "math_handler": math_handler,
            "weather_handler": weather_handler,
            "code_handler": code_handler,
            "general_handler": general_handler,
        }

    def invoke(self, query: str) -> dict:
        """Execute the routing graph."""
        # Initialize state
        state: RoutingState = {
            "query": query,
            "category": "",
            "response": "",
            "confidence": 0.0,
        }

        # Classify
        update = self.classifier(state)
        state["category"] = update["category"]
        state["confidence"] = update["confidence"]

        # Route
        handler_name = self.router(state)

        # Handle
        handler = self.handlers.get(handler_name, general_handler)
        response_update = handler(state)
        state["response"] = response_update["response"]

        return state


# =============================================================================
# TASK 6: Implement Confidence-Based Routing
# =============================================================================
def route_with_confidence(state: RoutingState) -> str:
    """Route based on both category and confidence."""
    if state["confidence"] >= 0.7:
        return route_by_category(state)
    return "clarify"


def clarify_node(state: RoutingState) -> dict:
    """Ask for clarification."""
    return {
        "response": "Could you please clarify your question? I want to make sure I understand what you're looking for."
    }


# =============================================================================
# TASK 7: Define Retry State
# =============================================================================
class RetryState(TypedDict):
    """State for retry pattern."""

    input: str
    output: str
    error: str | None
    retry_count: int
    max_retries: int
    success: bool


# =============================================================================
# TASK 8: Implement Retry Loop
# =============================================================================
def unreliable_operation(state: RetryState) -> dict:
    """Simulate an operation that might fail."""
    if state["retry_count"] < 2:
        return {"error": "Temporary failure", "retry_count": state["retry_count"] + 1}
    return {"output": "Success!", "success": True, "error": None}


def should_retry(state: RetryState) -> str:
    """Determine if we should retry."""
    if state.get("error") and state["retry_count"] < state["max_retries"]:
        return "retry"
    elif state.get("error"):
        return "fail"
    return "success"


def handle_failure(state: RetryState) -> dict:
    """Handle final failure."""
    return {"output": f"Failed after {state['retry_count']} retries"}


class RetryGraph:
    """Graph with retry loop."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def invoke(self, input_text: str) -> dict:
        """Execute the retry graph."""
        state: RetryState = {
            "input": input_text,
            "output": "",
            "error": None,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "success": False,
        }

        while True:
            # Run operation
            update = unreliable_operation(state)
            state = {**state, **update}

            # Check what to do next
            decision = should_retry(state)

            if decision == "success":
                break
            elif decision == "fail":
                update = handle_failure(state)
                state = {**state, **update}
                break
            # If "retry", loop continues

        return state


# =============================================================================
# TASK 9: Implement Multi-Condition Router
# =============================================================================
class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MultiConditionState(TypedDict):
    """State for multi-condition routing."""

    request: str
    priority: str
    category: str
    assigned_to: str
    response: str


def analyze_request(state: MultiConditionState) -> dict:
    """Analyze request priority and category."""
    request = state["request"].lower()

    # Priority
    if "urgent" in request or "asap" in request:
        priority = "high"
    elif "important" in request:
        priority = "medium"
    else:
        priority = "low"

    # Category (reuse logic)
    routing_state: RoutingState = {
        "query": request,
        "category": "",
        "response": "",
        "confidence": 0.0,
    }
    classification = classify_query(routing_state)

    return {"priority": priority, "category": classification["category"]}


def multi_route(state: MultiConditionState) -> str:
    """Route based on multiple conditions."""
    priority = state["priority"]
    category = state["category"]

    if priority == "high":
        if category == "math":
            return "senior_math"
        return "senior_general"
    elif priority == "medium":
        return f"standard_{category}"
    else:
        return "queue"


# =============================================================================
# TASK 10: Implement Cycle Detection
# =============================================================================
class CycleState(TypedDict):
    """State for cycle detection."""

    value: int
    history: list
    iteration: int


def increment_node(state: CycleState) -> dict:
    """Increment value and track."""
    return {
        "value": state["value"] + 1,
        "history": state["history"] + ["increment"],
        "iteration": state["iteration"] + 1,
    }


def check_threshold(state: CycleState) -> str:
    """Check if should continue or stop."""
    if state["value"] < 10 and state["iteration"] < 20:
        return "continue"
    return "stop"


class SafeCycleGraph:
    """Graph with controlled cycles."""

    def __init__(self, max_iterations: int = 20):
        self.max_iterations = max_iterations

    def invoke(self, start_value: int = 0) -> dict:
        """Execute the cycle graph safely."""
        state: CycleState = {"value": start_value, "history": [], "iteration": 0}

        while state["iteration"] < self.max_iterations:
            # Increment
            update = increment_node(state)
            state = {**state, **update}

            # Check
            decision = check_threshold(state)
            if decision == "stop":
                break

        return state


if __name__ == "__main__":
    print("=" * 60)
    print("Week 10 - Solution 2: Conditional Routing & Cycles")
    print("=" * 60)

    # Test classifier
    print("\n--- Query Classification ---")
    queries = [
        "What is 2 + 2?",
        "Weather today?",
        "How to write a function?",
        "Tell me a joke",
    ]
    for q in queries:
        result = classify_query(
            {"query": q, "category": "", "response": "", "confidence": 0.0}
        )
        print(f"'{q}' -> {result}")

    # Test router
    print("\n--- Conditional Router ---")
    router = ConditionalRouterGraph()
    result = router.invoke("Calculate 15 * 3")
    print(f"Result: {result}")

    # Test retry
    print("\n--- Retry Graph ---")
    retry = RetryGraph(max_retries=5)
    result = retry.invoke("test")
    print(f"Retry result: {result}")

    # Test cycles
    print("\n--- Safe Cycles ---")
    cycle = SafeCycleGraph()
    result = cycle.invoke(0)
    print(f"Cycle result: value={result['value']}, iterations={result['iteration']}")

    print("\n✅ All solutions implemented!")

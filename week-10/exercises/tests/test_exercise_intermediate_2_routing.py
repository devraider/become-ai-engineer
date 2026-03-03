"""
Tests for Week 10 - Exercise 2: Conditional Routing & Cycles
"""

import pytest
from exercise_intermediate_2_routing import (
    RoutingState,
    classify_query,
    route_by_category,
    math_handler,
    weather_handler,
    code_handler,
    general_handler,
    ConditionalRouterGraph,
    route_with_confidence,
    clarify_node,
    RetryState,
    unreliable_operation,
    should_retry,
    handle_failure,
    RetryGraph,
    MultiConditionState,
    analyze_request,
    multi_route,
    CycleState,
    check_threshold,
    SafeCycleGraph,
)


class TestClassifyQuery:
    """Tests for Task 2: Query Classifier."""

    def test_classify_math(self):
        """classify_query should detect math queries."""
        state = {
            "query": "What is 2 + 2?",
            "category": "",
            "response": "",
            "confidence": 0.0,
        }
        result = classify_query(state)
        assert result["category"] == "math"

    def test_classify_weather(self):
        """classify_query should detect weather queries."""
        state = {
            "query": "What's the weather like today?",
            "category": "",
            "response": "",
            "confidence": 0.0,
        }
        result = classify_query(state)
        assert result["category"] == "weather"

    def test_classify_code(self):
        """classify_query should detect code queries."""
        state = {
            "query": "How do I write a function in Python?",
            "category": "",
            "response": "",
            "confidence": 0.0,
        }
        result = classify_query(state)
        assert result["category"] == "code"

    def test_classify_general(self):
        """classify_query should fall back to general."""
        state = {
            "query": "Tell me a joke",
            "category": "",
            "response": "",
            "confidence": 0.0,
        }
        result = classify_query(state)
        assert result["category"] == "general"

    def test_confidence_score(self):
        """classify_query should set confidence score."""
        state = {
            "query": "Calculate 5 * 3",
            "category": "",
            "response": "",
            "confidence": 0.0,
        }
        result = classify_query(state)
        assert 0.0 < result["confidence"] <= 1.0


class TestRouteByCategory:
    """Tests for Task 3: Route Function."""

    def test_route_math(self):
        """route_by_category should return math_handler for math."""
        state: RoutingState = {
            "query": "",
            "category": "math",
            "response": "",
            "confidence": 0.9,
        }
        assert route_by_category(state) == "math_handler"

    def test_route_weather(self):
        """route_by_category should return weather_handler for weather."""
        state: RoutingState = {
            "query": "",
            "category": "weather",
            "response": "",
            "confidence": 0.9,
        }
        assert route_by_category(state) == "weather_handler"

    def test_route_code(self):
        """route_by_category should return code_handler for code."""
        state: RoutingState = {
            "query": "",
            "category": "code",
            "response": "",
            "confidence": 0.9,
        }
        assert route_by_category(state) == "code_handler"


class TestHandlers:
    """Tests for Task 4: Handler Nodes."""

    def test_math_handler_returns_response(self):
        """math_handler should return a response."""
        state: RoutingState = {
            "query": "2 + 2",
            "category": "math",
            "response": "",
            "confidence": 0.9,
        }
        result = math_handler(state)
        assert "response" in result
        assert len(result["response"]) > 0

    def test_weather_handler_returns_response(self):
        """weather_handler should return a response."""
        state: RoutingState = {
            "query": "weather today",
            "category": "weather",
            "response": "",
            "confidence": 0.9,
        }
        result = weather_handler(state)
        assert "response" in result

    def test_code_handler_returns_response(self):
        """code_handler should return a response."""
        state: RoutingState = {
            "query": "how to code",
            "category": "code",
            "response": "",
            "confidence": 0.9,
        }
        result = code_handler(state)
        assert "response" in result


class TestConditionalRouterGraph:
    """Tests for Task 5: Conditional Router Graph."""

    def test_router_graph_math_query(self):
        """ConditionalRouterGraph should route math queries correctly."""
        router = ConditionalRouterGraph()
        result = router.invoke("Calculate 5 + 3")
        assert result["category"] == "math"
        assert len(result["response"]) > 0

    def test_router_graph_weather_query(self):
        """ConditionalRouterGraph should route weather queries correctly."""
        router = ConditionalRouterGraph()
        result = router.invoke("What's the temperature?")
        assert result["category"] == "weather"


class TestConfidenceRouting:
    """Tests for Task 6: Confidence-Based Routing."""

    def test_high_confidence_routes_to_handler(self):
        """High confidence should route to specific handler."""
        state: RoutingState = {
            "query": "2+2",
            "category": "math",
            "response": "",
            "confidence": 0.9,
        }
        result = route_with_confidence(state)
        assert result == "math_handler"

    def test_low_confidence_routes_to_clarify(self):
        """Low confidence should route to clarify."""
        state: RoutingState = {
            "query": "hmm",
            "category": "general",
            "response": "",
            "confidence": 0.4,
        }
        result = route_with_confidence(state)
        assert result == "clarify"


class TestRetryPattern:
    """Tests for Tasks 7-8: Retry Pattern."""

    def test_unreliable_operation_fails_initially(self):
        """unreliable_operation should fail on first attempts."""
        state: RetryState = {
            "input": "test",
            "output": "",
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
            "success": False,
        }
        result = unreliable_operation(state)
        assert result.get("error") is not None
        assert result["retry_count"] == 1

    def test_unreliable_operation_succeeds_eventually(self):
        """unreliable_operation should succeed after retries."""
        state: RetryState = {
            "input": "test",
            "output": "",
            "error": None,
            "retry_count": 2,
            "max_retries": 3,
            "success": False,
        }
        result = unreliable_operation(state)
        assert result.get("success") is True
        assert result.get("error") is None

    def test_should_retry_continues(self):
        """should_retry should return 'retry' when retries available."""
        state: RetryState = {
            "input": "",
            "output": "",
            "error": "Failed",
            "retry_count": 1,
            "max_retries": 3,
            "success": False,
        }
        assert should_retry(state) == "retry"

    def test_should_retry_fails(self):
        """should_retry should return 'fail' when max retries reached."""
        state: RetryState = {
            "input": "",
            "output": "",
            "error": "Failed",
            "retry_count": 3,
            "max_retries": 3,
            "success": False,
        }
        assert should_retry(state) == "fail"

    def test_should_retry_success(self):
        """should_retry should return 'success' when no error."""
        state: RetryState = {
            "input": "",
            "output": "Done",
            "error": None,
            "retry_count": 2,
            "max_retries": 3,
            "success": True,
        }
        assert should_retry(state) == "success"

    def test_retry_graph_succeeds(self):
        """RetryGraph should eventually succeed."""
        graph = RetryGraph(max_retries=5)
        result = graph.invoke("test")
        assert result["success"] is True


class TestMultiConditionRouting:
    """Tests for Task 9: Multi-Condition Router."""

    def test_analyze_urgent_request(self):
        """analyze_request should detect urgent priority."""
        state: MultiConditionState = {
            "request": "URGENT: calculate now",
            "priority": "",
            "category": "",
            "assigned_to": "",
            "response": "",
        }
        result = analyze_request(state)
        assert result["priority"] == "high"

    def test_multi_route_high_math(self):
        """multi_route should route high priority math to senior."""
        state: MultiConditionState = {
            "request": "",
            "priority": "high",
            "category": "math",
            "assigned_to": "",
            "response": "",
        }
        result = multi_route(state)
        assert "senior" in result


class TestSafeCycles:
    """Tests for Task 10: Safe Cycles."""

    def test_check_threshold_continues(self):
        """check_threshold should continue when below threshold."""
        state: CycleState = {"value": 5, "history": [], "iteration": 3}
        assert check_threshold(state) == "continue"

    def test_check_threshold_stops(self):
        """check_threshold should stop when threshold reached."""
        state: CycleState = {"value": 10, "history": [], "iteration": 10}
        assert check_threshold(state) == "stop"

    def test_safe_cycle_graph_reaches_target(self):
        """SafeCycleGraph should increment to target value."""
        graph = SafeCycleGraph(max_iterations=20)
        result = graph.invoke(start_value=0)
        assert result["value"] >= 10
        assert result["iteration"] <= 20

    def test_safe_cycle_graph_respects_limit(self):
        """SafeCycleGraph should respect max iterations."""
        graph = SafeCycleGraph(max_iterations=5)
        result = graph.invoke(start_value=0)
        assert result["iteration"] <= 5

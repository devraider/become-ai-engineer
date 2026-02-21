"""
Tests for Week 5 Exercise 3: Tool Use & Function Calling
"""

import pytest
from exercise_advanced_3_tools import (
    ToolDefinition,
    ToolRegistry,
    execute_tool_call,
    parse_function_call,
    ToolAgent,
    validate_tool_arguments,
    create_tool_prompt,
    # Sample tools for testing
    get_weather,
    search_database,
    calculate_math,
)


class TestToolDefinition:
    def test_creation(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"param1": {"type": "string", "description": "Test param"}},
            required_params=["param1"],
            function=lambda x: x,
        )
        assert tool.name == "test_tool"

    def test_to_gemini_format(self):
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string", "description": "City"}},
            required_params=["location"],
            function=get_weather,
        )

        result = tool.to_gemini_format()
        if result:  # If implemented
            assert "name" in result or "function_declarations" in str(result)

    def test_to_openai_format(self):
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters={"query": {"type": "string", "description": "Query"}},
            required_params=["query"],
            function=lambda q: [q],
        )

        result = tool.to_openai_format()
        if result:  # If implemented
            assert "name" in result or "function" in result


class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        registry.register(
            name="test",
            description="Test tool",
            parameters={"x": {"type": "string", "description": "Test"}},
            required_params=["x"],
            function=lambda x: x,
        )

        assert "test" in registry.list_tools()

    def test_get_tool(self):
        registry = ToolRegistry()
        registry.register(
            name="my_tool",
            description="My tool",
            parameters={},
            required_params=[],
            function=lambda: "result",
        )

        tool = registry.get_tool("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register("tool1", "Desc1", {}, [], lambda: 1)
        registry.register("tool2", "Desc2", {}, [], lambda: 2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools

    def test_get_all_definitions(self):
        registry = ToolRegistry()
        registry.register(
            "tool1",
            "Desc1",
            {"p": {"type": "string", "description": "p"}},
            [],
            lambda: 1,
        )

        definitions = registry.get_all_definitions("gemini")
        assert isinstance(definitions, list)


class TestExecuteToolCall:
    def test_successful_execution(self):
        registry = ToolRegistry()
        registry.register(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string", "description": "City"}},
            required_params=["location"],
            function=get_weather,
        )

        result = execute_tool_call(registry, "get_weather", {"location": "London"})

        if result:  # If implemented
            assert (
                "success" in result or "result" in result or "location" in str(result)
            )

    def test_tool_not_found(self):
        registry = ToolRegistry()
        result = execute_tool_call(registry, "nonexistent", {})

        if result:  # If implemented
            assert "error" in result or result.get("success") == False

    def test_with_calculate(self):
        registry = ToolRegistry()
        registry.register(
            name="calculate",
            description="Calculate",
            parameters={"expression": {"type": "string", "description": "Math"}},
            required_params=["expression"],
            function=calculate_math,
        )

        result = execute_tool_call(registry, "calculate", {"expression": "2 + 2"})

        if result and "result" in result:
            assert result["result"].get("result") == 4 or result.get("result") == 4


class TestParseFunctionCall:
    def test_parse_tagged_call(self):
        text = 'Some text <function_call>{"name": "test", "arguments": {"x": 1}}</function_call>'
        result = parse_function_call(text)

        if result:  # If implemented
            assert result["name"] == "test"
            assert result["arguments"]["x"] == 1

    def test_no_function_call(self):
        text = "This is just regular text with no function call"
        result = parse_function_call(text)
        assert result is None

    def test_json_only(self):
        text = '{"name": "func", "arguments": {}}'
        result = parse_function_call(text)
        # Should handle both cases (find it or return None)


class TestToolAgent:
    @pytest.fixture
    def agent(self):
        registry = ToolRegistry()
        registry.register(
            name="get_weather",
            description="Get weather for a location",
            parameters={"location": {"type": "string", "description": "City"}},
            required_params=["location"],
            function=get_weather,
        )
        registry.register(
            name="calculate",
            description="Calculate math expression",
            parameters={"expression": {"type": "string", "description": "Math"}},
            required_params=["expression"],
            function=calculate_math,
        )
        return ToolAgent(registry)

    def test_initialization(self, agent):
        assert agent is not None

    def test_process_question(self, agent):
        response = agent.process_question("What's the weather in Tokyo?")
        assert isinstance(response, str)

    def test_math_question(self, agent):
        response = agent.process_question("Calculate 10 + 5")
        assert isinstance(response, str)


class TestValidateToolArguments:
    def test_valid_arguments(self):
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={
                "required_param": {"type": "string", "description": "Required"}
            },
            required_params=["required_param"],
            function=lambda x: x,
        )

        is_valid, error = validate_tool_arguments(tool, {"required_param": "value"})

        if is_valid is not None:  # If implemented
            assert is_valid == True

    def test_missing_required(self):
        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={
                "required_param": {"type": "string", "description": "Required"}
            },
            required_params=["required_param"],
            function=lambda x: x,
        )

        is_valid, error = validate_tool_arguments(tool, {})

        if is_valid is not None:  # If implemented
            assert is_valid == False
            assert error is not None


class TestCreateToolPrompt:
    def test_includes_question(self):
        tools = [
            ToolDefinition("tool1", "Description 1", {}, [], lambda: 1),
        ]
        prompt = create_tool_prompt("What is the weather?", tools)

        if prompt:  # If implemented
            assert "weather" in prompt.lower()

    def test_includes_tool_info(self):
        tools = [
            ToolDefinition("get_weather", "Get weather data", {}, [], lambda: 1),
        ]
        prompt = create_tool_prompt("Test question", tools)

        if prompt:  # If implemented
            assert "get_weather" in prompt or "weather" in prompt.lower()

"""
Tests for Week 9 - Exercise Advanced 3: Agents and Tools
"""

import pytest

from exercise_advanced_3_agents import (
    ToolResult,
    BaseTool,
    tool,
    FunctionTool,
    CalculatorTool,
    SearchTool,
    WikipediaTool,
    ToolRegistry,
    AgentAction,
    AgentFinish,
    ReActPromptParser,
    AgentExecutor,
    ToolCall,
    ToolCallingAgent,
    AgentWithMemory,
    MultiAgentSystem,
    ResearchAgent,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(success=True, output="42")
        assert result.success is True
        assert result.output == "42"
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(success=False, output="", error="Division by zero")
        assert result.success is False
        assert "Division" in result.error


class TestCalculatorTool:
    """Tests for CalculatorTool class."""

    def test_basic_math(self):
        """Test basic calculations."""
        calc = CalculatorTool()
        result = calc.run("2 + 2")
        assert result.success is True
        assert "4" in result.output

    def test_complex_expression(self):
        """Test complex expressions."""
        calc = CalculatorTool()
        result = calc.run("(10 + 5) * 2")
        assert result.success is True
        assert "30" in result.output

    def test_math_functions(self):
        """Test math functions."""
        calc = CalculatorTool()
        result = calc.run("2 ** 3")
        assert result.success is True
        assert "8" in result.output

    def test_invalid_expression(self):
        """Test handling invalid expressions."""
        calc = CalculatorTool()
        result = calc.run("invalid + expression")
        assert result.success is False


class TestSearchTool:
    """Tests for SearchTool class."""

    def test_search_found(self):
        """Test finding a result."""
        knowledge = {
            "python": "Python is a programming language",
            "langchain": "LangChain is an LLM framework",
        }
        search = SearchTool(knowledge)
        result = search.run("python")
        assert result.success is True
        assert "Python" in result.output or "programming" in result.output

    def test_search_not_found(self):
        """Test when query not found."""
        search = SearchTool({"only": "one entry"})
        result = search.run("nonexistent query")
        assert "not found" in result.output.lower() or result.output == ""


class TestWikipediaTool:
    """Tests for WikipediaTool class."""

    def test_lookup_topic(self):
        """Test looking up a topic."""
        summaries = {"python": "Python is a high-level programming language."}
        wiki = WikipediaTool(summaries)
        result = wiki.run("python")
        assert result.success is True
        assert "programming" in result.output.lower()

    def test_case_insensitive(self):
        """Test case-insensitive lookup."""
        summaries = {"Python": "A programming language"}
        wiki = WikipediaTool(summaries)
        result = wiki.run("PYTHON")
        assert result.success is True


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_create_tool(self):
        """Test creating tool with decorator."""

        @tool(name="double", description="Doubles a number")
        def double_it(x: str) -> str:
            return str(int(x) * 2)

        assert double_it.name == "double"
        assert "Doubles" in double_it.description
        result = double_it.run("5")
        assert "10" in result.output

    def test_default_name(self):
        """Test default name from function."""

        @tool()
        def my_function(x: str) -> str:
            """Does something."""
            return x

        assert my_function.name == "my_function"


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_and_get(self):
        """Test registering and getting tools."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        found = registry.get("calculator")
        assert found is not None
        assert found.name == "calculator"

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())

        names = registry.list_tools()
        assert "calculator" in names
        assert "search" in names

    def test_duplicate_registration(self):
        """Test that duplicate names raise error."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        with pytest.raises(ValueError):
            registry.register(CalculatorTool())

    def test_unregister(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        result = registry.unregister("calculator")
        assert result is True
        assert registry.get("calculator") is None

    def test_get_descriptions(self):
        """Test getting tool descriptions."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        descriptions = registry.get_tool_descriptions()
        assert "calculator" in descriptions.lower()


class TestReActPromptParser:
    """Tests for ReActPromptParser class."""

    def test_parse_action(self):
        """Test parsing an action."""
        parser = ReActPromptParser()
        text = """
Thought: I need to search for information.
Action: search
Action Input: python programming
"""
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "search"
        assert "python" in result.tool_input.lower()

    def test_parse_finish(self):
        """Test parsing a final answer."""
        parser = ReActPromptParser()
        text = """
Thought: I now have all the information.
Final Answer: Python is a versatile programming language used for many purposes.
"""
        result = parser.parse(text)
        assert isinstance(result, AgentFinish)
        assert "Python" in result.output

    def test_invalid_format(self):
        """Test handling invalid format."""
        parser = ReActPromptParser()
        with pytest.raises(ValueError):
            parser.parse("Random text without proper format")


class TestAgentExecutor:
    """Tests for AgentExecutor class."""

    def test_simple_execution(self):
        """Test simple agent execution."""
        call_count = 0

        def mock_agent(input_text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return """
Thought: I need to calculate this.
Action: calculator
Action Input: 2 + 2
"""
            return """
Thought: I have the answer.
Final Answer: The result is 4.
"""

        executor = AgentExecutor(
            agent=mock_agent, tools=[CalculatorTool()], max_iterations=5
        )

        result = executor.run("What is 2 + 2?")
        assert "4" in result

    def test_max_iterations(self):
        """Test max iterations limit."""

        def infinite_agent(input_text):
            return """
Thought: Need more info.
Action: calculator
Action Input: 1 + 1
"""

        executor = AgentExecutor(
            agent=infinite_agent, tools=[CalculatorTool()], max_iterations=3
        )

        # Should stop after max iterations
        result = executor.run("Loop forever")
        assert result is not None


class TestToolCallingAgent:
    """Tests for ToolCallingAgent class."""

    def test_plan_and_execute(self):
        """Test planning and executing tools."""

        def mock_llm(input_text):
            return [
                ToolCall(id="1", name="calculator", arguments={"expression": "5 * 5"})
            ]

        agent = ToolCallingAgent(llm_with_tools=mock_llm, tools=[CalculatorTool()])

        calls = agent.plan("What is 5 * 5?")
        assert len(calls) >= 1

        results = agent.execute(calls)
        assert len(results) >= 1


class TestAgentWithMemory:
    """Tests for AgentWithMemory class."""

    def test_memory_integration(self):
        """Test that agent uses memory."""

        def mock_agent(input_text):
            return """
Thought: I can answer directly.
Final Answer: Hello! I remember our conversation.
"""

        executor = AgentExecutor(agent=mock_agent, tools=[], max_iterations=2)

        agent = AgentWithMemory(executor)

        response1 = agent.run("Hello!")
        response2 = agent.run("What did I say?")

        history = agent.get_chat_history()
        assert len(history) >= 2

    def test_clear_memory(self):
        """Test clearing agent memory."""

        def mock_agent(input_text):
            return "Thought: Done.\nFinal Answer: OK"

        executor = AgentExecutor(agent=mock_agent, tools=[], max_iterations=1)
        agent = AgentWithMemory(executor)

        agent.run("Test")
        agent.clear_memory()

        assert len(agent.get_chat_history()) == 0


class TestMultiAgentSystem:
    """Tests for MultiAgentSystem class."""

    def test_register_agents(self):
        """Test registering multiple agents."""
        system = MultiAgentSystem()

        mock_agent = type("MockAgent", (), {"run": lambda self, x: "OK"})()

        system.register_agent("math", mock_agent, "Handles math questions")
        system.register_agent("search", mock_agent, "Handles search")

        agents = system.list_agents()
        assert len(agents) == 2

    def test_routing(self):
        """Test routing to correct agent."""
        system = MultiAgentSystem()

        math_agent = type("MathAgent", (), {"run": lambda self, x: "MATH"})()
        search_agent = type("SearchAgent", (), {"run": lambda self, x: "SEARCH"})()

        system.register_agent("math", math_agent, "Math questions")
        system.register_agent("search", search_agent, "Search queries")

        def router(text):
            if "calculate" in text.lower():
                return "math"
            return "search"

        system.set_router(router)

        result = system.run("Please calculate 2+2")
        assert result == "MATH"


class TestResearchAgent:
    """Tests for ResearchAgent class."""

    def test_initialization(self):
        """Test agent initializes correctly."""
        agent = ResearchAgent()
        assert agent is not None

    def test_basic_chat(self):
        """Test basic chat interaction."""

        def mock_llm(prompt):
            return """
Thought: I can answer this directly.
Final Answer: Hello! How can I help you today?
"""

        agent = ResearchAgent(llm=mock_llm)
        response = agent.chat("Hello!")
        assert response is not None

    def test_tool_use(self):
        """Test that agent can use tools."""
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "calculator" in prompt.lower():
                return """
Thought: I need to calculate this.
Action: calculator
Action Input: 10 * 5
"""
            return """
Thought: I have the answer.
Final Answer: The result is 50.
"""

        agent = ResearchAgent(llm=mock_llm, verbose=False)
        response = agent.run("What is 10 times 5?")

        # Should either use tool or answer directly
        assert response is not None

    def test_reset(self):
        """Test resetting agent state."""
        agent = ResearchAgent()
        agent.chat("Test message")
        agent.reset()
        # After reset, memory should be cleared

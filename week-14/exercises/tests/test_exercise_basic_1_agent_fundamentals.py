"""
Tests for Week 14 - Exercise 1: Agent Fundamentals

Run with: pytest tests/test_exercise_basic_1_agent_fundamentals.py -v
"""

import pytest
from datetime import datetime

from exercise_basic_1_agent_fundamentals import (
    MessageRole,
    AgentMessage,
    ToolResult,
    AgentContext,
    ToolParameter,
    Tool,
    ToolRegistry,
    AgentStateType,
    AgentState,
    SimpleAgent,
    MemoryItem,
    AgentMemory,
    LogLevel,
    LogEntry,
    AgentLogger,
    AgentExecutor,
)


# =============================================================================
# Part 1: Agent Message Tests
# =============================================================================
class TestAgentMessage:
    """Tests for AgentMessage class."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = AgentMessage(role=MessageRole.USER, content="Hello!")

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert isinstance(msg.timestamp, datetime)

    def test_is_from_user(self):
        """Test is_from_user method."""
        user_msg = AgentMessage(role=MessageRole.USER, content="Hi")
        assistant_msg = AgentMessage(role=MessageRole.ASSISTANT, content="Hello")

        assert user_msg.is_from_user() is True
        assert assistant_msg.is_from_user() is False

    def test_is_from_assistant(self):
        """Test is_from_assistant method."""
        user_msg = AgentMessage(role=MessageRole.USER, content="Hi")
        assistant_msg = AgentMessage(role=MessageRole.ASSISTANT, content="Hello")

        assert user_msg.is_from_assistant() is False
        assert assistant_msg.is_from_assistant() is True

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = AgentMessage(role=MessageRole.USER, content="Test")
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Test"
        assert "timestamp" in result


# =============================================================================
# Part 2: Tool Result Tests
# =============================================================================
class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult.success("calculator", 42, time_ms=10.5)

        assert result.success is True
        assert result.result == 42
        assert result.tool_name == "calculator"
        assert result.execution_time_ms == 10.5

    def test_failure_result(self):
        """Test creating a failed result."""
        result = ToolResult.failure("api_call", "Connection timeout")

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.tool_name == "api_call"

    def test_to_message(self):
        """Test converting result to message."""
        result = ToolResult.success("search", "Found 5 results")
        msg = result.to_message()

        assert msg.role == MessageRole.TOOL
        assert "Found 5 results" in msg.content


# =============================================================================
# Part 3: Agent Context Tests
# =============================================================================
class TestAgentContext:
    """Tests for AgentContext class."""

    def test_add_messages(self):
        """Test adding messages to context."""
        ctx = AgentContext(max_messages=10)

        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi there!")

        assert len(ctx.messages) == 2

    def test_max_messages_limit(self):
        """Test message limit enforcement."""
        ctx = AgentContext(max_messages=3)

        for i in range(5):
            ctx.add_user_message(f"Message {i}")

        assert len(ctx.messages) == 3

    def test_get_recent(self):
        """Test getting recent messages."""
        ctx = AgentContext()
        ctx.add_user_message("First")
        ctx.add_assistant_message("Second")
        ctx.add_user_message("Third")

        recent = ctx.get_recent(2)

        assert len(recent) == 2
        assert recent[-1].content == "Third"

    def test_to_prompt_format(self):
        """Test converting to prompt format."""
        ctx = AgentContext()
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi!")

        prompt = ctx.to_prompt_format()

        assert len(prompt) == 2
        assert prompt[0]["role"] == "user"
        assert prompt[1]["role"] == "assistant"

    def test_clear(self):
        """Test clearing context."""
        ctx = AgentContext()
        ctx.add_user_message("Test")
        ctx.clear()

        assert len(ctx.messages) == 0


# =============================================================================
# Part 4: Tool Tests
# =============================================================================
class TestTool:
    """Tests for Tool class."""

    def test_execute_tool(self):
        """Test executing a tool."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters=[
                ToolParameter("a", "int", "First number"),
                ToolParameter("b", "int", "Second number"),
            ],
        )

        result = tool.execute({"a": 5, "b": 3})

        assert result == 8

    def test_to_schema(self):
        """Test generating tool schema."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = Tool(
            name="greet",
            description="Greet someone",
            function=greet,
            parameters=[ToolParameter("name", "string", "Name to greet")],
        )

        schema = tool.to_schema()

        assert schema["name"] == "greet"
        assert "parameters" in schema

    def test_validate_args(self):
        """Test argument validation."""
        tool = Tool(
            name="test",
            description="Test tool",
            function=lambda x: x,
            parameters=[
                ToolParameter("required", "string", "Required param"),
                ToolParameter("optional", "string", "Optional", required=False),
            ],
        )

        valid, error = tool.validate_args({"required": "value"})
        assert valid is True

        valid, error = tool.validate_args({})
        assert valid is False


# =============================================================================
# Part 5: Tool Registry Tests
# =============================================================================
class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_decorator(self):
        """Test registering a tool with decorator."""
        registry = ToolRegistry()

        @registry.register
        def search(query: str) -> str:
            """Search for something."""
            return f"Results for: {query}"

        assert "search" in registry.list_tools()

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: int) -> int:
            """A tool."""
            return x * 2

        tool = registry.get("my_tool")

        assert tool is not None
        assert tool.name == "my_tool"

    def test_execute_registered_tool(self):
        """Test executing a registered tool."""
        registry = ToolRegistry()

        @registry.register
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        tool = registry.get("multiply")
        result = tool.execute({"a": 3, "b": 4})

        assert result == 12


# =============================================================================
# Part 6: Agent State Tests
# =============================================================================
class TestAgentState:
    """Tests for AgentState class."""

    def test_initial_state(self):
        """Test initial state is IDLE."""
        state = AgentState()

        assert state.current == AgentStateType.IDLE

    def test_valid_transition(self):
        """Test valid state transitions."""
        state = AgentState()

        result = state.transition_to(AgentStateType.THINKING)

        assert result is True
        assert state.current == AgentStateType.THINKING

    def test_invalid_transition(self):
        """Test invalid state transitions."""
        state = AgentState()

        # Can't go directly from IDLE to DONE
        result = state.transition_to(AgentStateType.DONE)

        assert result is False
        assert state.current == AgentStateType.IDLE

    def test_can_transition(self):
        """Test checking if transition is valid."""
        state = AgentState()

        assert state.can_transition(AgentStateType.THINKING) is True
        assert state.can_transition(AgentStateType.DONE) is False

    def test_reset(self):
        """Test resetting state."""
        state = AgentState()
        state.transition_to(AgentStateType.THINKING)
        state.reset()

        assert state.current == AgentStateType.IDLE


# =============================================================================
# Part 7: Simple Agent Tests
# =============================================================================
class TestSimpleAgent:
    """Tests for SimpleAgent class."""

    def test_create_agent(self):
        """Test creating an agent."""
        agent = SimpleAgent(name="assistant")

        assert agent.name == "assistant"

    def test_add_tool(self):
        """Test adding tools to agent."""
        agent = SimpleAgent(name="test")

        @agent.tool
        def calculator(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        assert "calculator" in agent._tools.list_tools()

    def test_execute_tool(self):
        """Test executing an agent tool."""
        agent = SimpleAgent(name="test")

        @agent.tool
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        result = agent.execute_tool("double", {"x": 5})

        assert result.success is True
        assert result.result == 10


# =============================================================================
# Part 8: Agent Memory Tests
# =============================================================================
class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_store_memory(self):
        """Test storing memories."""
        memory = AgentMemory(capacity=100)

        memory.store("User likes Python", importance=0.8)
        memory.store("User is learning ML", importance=0.9)

        assert memory.count() == 2

    def test_retrieve_by_query(self):
        """Test retrieving memories by query."""
        memory = AgentMemory()
        memory.store("Python programming tips")
        memory.store("Machine learning basics")
        memory.store("Python best practices")

        results = memory.retrieve("Python", k=2)

        assert len(results) <= 2
        assert all("Python" in r.content for r in results)

    def test_get_recent(self):
        """Test getting recent memories."""
        memory = AgentMemory()
        memory.store("First")
        memory.store("Second")
        memory.store("Third")

        recent = memory.get_recent(2)

        assert len(recent) == 2

    def test_get_important(self):
        """Test getting important memories."""
        memory = AgentMemory()
        memory.store("Low importance", importance=0.2)
        memory.store("High importance", importance=0.9)
        memory.store("Medium importance", importance=0.5)

        important = memory.get_important(2, min_importance=0.4)

        assert len(important) == 2
        assert all(m.importance >= 0.4 for m in important)

    def test_capacity_limit(self):
        """Test memory capacity limits."""
        memory = AgentMemory(capacity=3)

        for i in range(5):
            memory.store(f"Memory {i}", importance=float(i) / 10)

        assert memory.count() == 3


# =============================================================================
# Part 9: Agent Logger Tests
# =============================================================================
class TestAgentLogger:
    """Tests for AgentLogger class."""

    def test_log_messages(self):
        """Test logging messages at different levels."""
        logger = AgentLogger("test-agent")

        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        entries = logger.get_entries()

        assert len(entries) == 3

    def test_min_level_filter(self):
        """Test minimum level filtering."""
        logger = AgentLogger("test", min_level=LogLevel.WARNING)

        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")

        entries = logger.get_entries()

        # Only WARNING and ERROR should be logged
        assert len(entries) == 2

    def test_tool_call_logging(self):
        """Test logging tool calls."""
        logger = AgentLogger("test")

        logger.tool_call("search", {"query": "test"})

        entries = logger.get_entries()

        assert len(entries) == 1
        assert "search" in entries[0].message

    def test_filter_by_level(self):
        """Test filtering entries by level."""
        logger = AgentLogger("test")

        logger.info("Info 1")
        logger.error("Error 1")
        logger.info("Info 2")

        errors = logger.get_entries(level=LogLevel.ERROR)

        assert len(errors) == 1
        assert errors[0].level == LogLevel.ERROR


# =============================================================================
# Part 10: Agent Executor Tests
# =============================================================================
class TestAgentExecutor:
    """Tests for AgentExecutor class."""

    def test_run_agent(self):
        """Test running an agent."""
        agent = SimpleAgent(name="test")

        @agent.tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        executor = AgentExecutor(max_retries=3)
        result = executor.run(agent, "echo hello")

        assert result is not None

    def test_batch_execution(self):
        """Test batch execution."""
        agent = SimpleAgent(name="test")

        @agent.tool
        def process(x: str) -> str:
            """Process input."""
            return f"Processed: {x}"

        executor = AgentExecutor()
        results = executor.run_batch(agent, ["input1", "input2", "input3"])

        assert len(results) == 3


# =============================================================================
# Integration Tests
# =============================================================================
class TestAgentIntegration:
    """Integration tests for agent components."""

    def test_full_agent_workflow(self):
        """Test complete agent workflow."""
        # Create agent with tools
        agent = SimpleAgent(name="calculator-agent")

        @agent.tool
        def calculate(expression: str) -> float:
            """Evaluate a math expression."""
            # Simple mock - just return 42
            return 42.0

        # Execute
        executor = AgentExecutor()
        result = executor.run(agent, "Calculate 2 + 2")

        # Verify
        assert result is not None

    def test_agent_with_memory(self):
        """Test agent using memory."""
        agent = SimpleAgent(name="memory-agent")
        memory = AgentMemory()

        # Store some memories
        memory.store("User prefers short responses", importance=0.9)
        memory.store("User works in data science", importance=0.8)

        # Retrieve relevant memories
        relevant = memory.retrieve("data", k=1)

        assert len(relevant) == 1
        assert "data science" in relevant[0].content

    def test_agent_with_logging(self):
        """Test agent with logging."""
        agent = SimpleAgent(name="logged-agent")
        logger = AgentLogger(agent.name)

        @agent.tool
        def search(query: str) -> str:
            """Search for information."""
            logger.tool_call("search", {"query": query})
            return f"Results for: {query}"

        # Run agent
        agent.run("Search for Python")

        # Check logs
        entries = logger.get_entries()
        assert len(entries) >= 1

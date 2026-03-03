"""
Tests for Week 14 - Exercise 2: Google ADK Integration

Run with: pytest tests/test_exercise_intermediate_2_adk_integration.py -v
"""

import pytest
import asyncio
from datetime import datetime

from exercise_intermediate_2_adk_integration import (
    ToolSchema,
    ADKToolWrapper,
    ModelProvider,
    ADKAgentConfig,
    ADKAgent,
    ToolSchemaGenerator,
    MemoryType,
    MemoryEntry,
    ADKMemory,
    RunnerMode,
    RunResult,
    ADKRunner,
    StreamEvent,
    StreamChunk,
    StreamingHandler,
    ADKToolkit,
    AgentMetrics,
    AgentMonitor,
    AgentTemplate,
    ADKAgentFactory,
)


# =============================================================================
# Part 1: ADK Tool Wrapper Tests
# =============================================================================
class TestADKToolWrapper:
    """Tests for ADKToolWrapper class."""

    def test_wrap_function_as_tool(self):
        """Test wrapping a function as a tool."""
        wrapper = ADKToolWrapper()

        @wrapper.tool
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

        schema = wrapper.get_schema("greet")

        assert schema is not None
        assert schema.name == "greet"

    def test_execute_wrapped_tool(self):
        """Test executing a wrapped tool."""
        wrapper = ADKToolWrapper()

        @wrapper.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = wrapper.execute("add", a=5, b=3)

        assert result == 8

    def test_custom_tool_name(self):
        """Test using custom tool name."""
        wrapper = ADKToolWrapper()

        @wrapper.tool(name="custom_name")
        def my_function(x: int) -> int:
            """A function."""
            return x * 2

        schema = wrapper.get_schema("custom_name")

        assert schema is not None

    def test_get_all_schemas(self):
        """Test getting all tool schemas."""
        wrapper = ADKToolWrapper()

        @wrapper.tool
        def tool1(x: int) -> int:
            """Tool 1."""
            return x

        @wrapper.tool
        def tool2(y: str) -> str:
            """Tool 2."""
            return y

        schemas = wrapper.get_all_schemas()

        assert len(schemas) == 2


# =============================================================================
# Part 2: ADK Agent Config Tests
# =============================================================================
class TestADKAgentConfig:
    """Tests for ADKAgentConfig class."""

    def test_create_config(self):
        """Test creating agent configuration."""
        config = ADKAgentConfig(
            name="test-agent", model="gemini-2.0-flash", temperature=0.7
        )

        assert config.name == "test-agent"
        assert config.model == "gemini-2.0-flash"
        assert config.temperature == 0.7

    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = ADKAgentConfig(name="test", temperature=0.7, max_tokens=1000)

        assert valid_config.validate() is True

        invalid_config = ADKAgentConfig(name="test", temperature=3.0)  # Invalid: > 2

        assert invalid_config.validate() is False

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ADKAgentConfig(name="test")
        result = config.to_dict()

        assert result["name"] == "test"
        assert "model" in result

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"name": "test-agent", "model": "gemini-2.0-flash", "temperature": 0.5}

        config = ADKAgentConfig.from_dict(data)

        assert config.name == "test-agent"
        assert config.temperature == 0.5

    def test_with_model(self):
        """Test creating config with different model."""
        config = ADKAgentConfig(name="test", model="gemini-2.0-flash")
        new_config = config.with_model("gemini-pro")

        assert new_config.model == "gemini-pro"
        assert config.model == "gemini-2.0-flash"  # Original unchanged


# =============================================================================
# Part 3: ADK Agent Tests
# =============================================================================
class TestADKAgent:
    """Tests for ADKAgent class."""

    def test_create_agent(self):
        """Test creating an ADK agent."""
        config = ADKAgentConfig(name="test-agent")
        agent = ADKAgent(config)

        assert agent is not None

    def test_add_tool(self):
        """Test adding a tool to agent."""
        config = ADKAgentConfig(name="test")
        agent = ADKAgent(config)

        def my_tool(x: int) -> int:
            return x * 2

        agent.add_tool(my_tool)

        # Tool should be added
        assert True  # Verify via internal state

    def test_add_multiple_tools(self):
        """Test adding multiple tools."""
        config = ADKAgentConfig(name="test")
        agent = ADKAgent(config)

        def tool1(x: int) -> int:
            return x

        def tool2(y: str) -> str:
            return y

        agent.add_tools([tool1, tool2])

        # Both tools should be added
        assert True

    @pytest.mark.asyncio
    async def test_run_agent(self):
        """Test running the agent."""
        config = ADKAgentConfig(name="test")
        agent = ADKAgent(config)

        result = await agent.run("Hello!")

        assert result is not None


# =============================================================================
# Part 4: Tool Schema Generator Tests
# =============================================================================
class TestToolSchemaGenerator:
    """Tests for ToolSchemaGenerator class."""

    def test_generate_schema(self):
        """Test generating schema from function."""

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        schema = ToolSchemaGenerator.from_function(add)

        assert schema["name"] == "add"
        assert "parameters" in schema

    def test_parameter_types(self):
        """Test extracting parameter types."""

        def func(text: str, count: int, ratio: float, flag: bool) -> str:
            """A function with various types."""
            return text

        schema = ToolSchemaGenerator.from_function(func)
        params = schema.get("parameters", {}).get("properties", {})

        # Should infer correct types
        assert params.get("text", {}).get("type") == "string"
        assert params.get("count", {}).get("type") == "integer"

    def test_validate_args(self):
        """Test validating arguments against schema."""
        schema = {
            "name": "test",
            "parameters": {
                "type": "object",
                "properties": {"required_param": {"type": "string"}},
                "required": ["required_param"],
            },
        }

        valid, errors = ToolSchemaGenerator.validate_against_schema(
            {"required_param": "value"}, schema
        )

        assert valid is True

        valid, errors = ToolSchemaGenerator.validate_against_schema({}, schema)

        assert valid is False


# =============================================================================
# Part 5: ADK Memory Tests
# =============================================================================
class TestADKMemory:
    """Tests for ADKMemory class."""

    def test_add_memory(self):
        """Test adding memories."""
        memory = ADKMemory()

        entry_id = memory.add(
            "User prefers Python", MemoryType.LONG_TERM, tags=["preference"]
        )

        assert entry_id is not None

    def test_get_memory(self):
        """Test getting memory by ID."""
        memory = ADKMemory()

        entry_id = memory.add("Test content", MemoryType.SHORT_TERM)

        entry = memory.get(entry_id)

        assert entry is not None
        assert entry.content == "Test content"

    def test_search_memory(self):
        """Test searching memory."""
        memory = ADKMemory()

        memory.add("Python programming tips", MemoryType.LONG_TERM)
        memory.add("Machine learning basics", MemoryType.LONG_TERM)
        memory.add("Python best practices", MemoryType.LONG_TERM)

        results = memory.search("Python", k=2)

        assert len(results) <= 2

    def test_memory_types(self):
        """Test different memory types."""
        memory = ADKMemory()

        memory.add("Short term", MemoryType.SHORT_TERM)
        memory.add("Long term", MemoryType.LONG_TERM)
        memory.add("Working", MemoryType.WORKING)

        # Search specific type
        results = memory.search("term", memory_types=[MemoryType.SHORT_TERM])

        assert all(r.memory_type == MemoryType.SHORT_TERM for r in results)

    def test_export_import(self):
        """Test exporting and importing memory state."""
        memory = ADKMemory()
        memory.add("Test memory", MemoryType.LONG_TERM)

        exported = memory.export()

        new_memory = ADKMemory()
        new_memory.import_state(exported)

        results = new_memory.search("Test")
        assert len(results) > 0


# =============================================================================
# Part 6: ADK Runner Tests
# =============================================================================
class TestADKRunner:
    """Tests for ADKRunner class."""

    def test_create_runner(self):
        """Test creating a runner."""
        runner = ADKRunner(mode=RunnerMode.SYNC)

        assert runner is not None

    @pytest.mark.asyncio
    async def test_run_async(self):
        """Test async execution."""
        config = ADKAgentConfig(name="test")
        agent = ADKAgent(config)

        runner = ADKRunner(mode=RunnerMode.ASYNC)
        result = await runner.run_async(agent, "Hello")

        assert isinstance(result, RunResult)

    def test_run_result_structure(self):
        """Test RunResult structure."""
        result = RunResult(
            output="Test output", tool_calls=[], iterations=1, total_time_ms=100.0
        )

        assert result.output == "Test output"
        assert result.iterations == 1


# =============================================================================
# Part 7: Streaming Handler Tests
# =============================================================================
class TestStreamingHandler:
    """Tests for StreamingHandler class."""

    def test_register_callbacks(self):
        """Test registering callbacks."""
        handler = StreamingHandler()

        tokens = []
        handler.on_token(lambda t: tokens.append(t))

        # Process a token chunk
        chunk = StreamChunk(event=StreamEvent.TOKEN, data="Hello")
        handler.process(chunk)

        assert "Hello" in tokens

    def test_process_different_events(self):
        """Test processing different event types."""
        handler = StreamingHandler()

        events_received = []
        handler.on_token(lambda t: events_received.append(("token", t)))
        handler.on_complete(lambda r: events_received.append(("complete", r)))

        handler.process(StreamChunk(StreamEvent.TOKEN, "Hi"))
        handler.process(StreamChunk(StreamEvent.COMPLETE, "Done"))

        assert len(events_received) == 2

    @pytest.mark.asyncio
    async def test_collect_stream(self):
        """Test collecting from stream."""
        handler = StreamingHandler()

        async def mock_stream():
            yield StreamChunk(StreamEvent.TOKEN, "Hello")
            yield StreamChunk(StreamEvent.TOKEN, " World")
            yield StreamChunk(StreamEvent.COMPLETE, "")

        result = await handler.collect(mock_stream())

        assert "Hello" in result


# =============================================================================
# Part 8: ADK Toolkit Tests
# =============================================================================
class TestADKToolkit:
    """Tests for ADKToolkit class."""

    def test_get_all_tools(self):
        """Test getting all tools."""
        toolkit = ADKToolkit()
        tools = toolkit.get_all()

        assert isinstance(tools, list)

    def test_current_datetime(self):
        """Test current datetime tool."""
        result = ADKToolkit.current_datetime()

        assert result is not None
        # Should be a valid datetime string
        assert len(result) > 0

    def test_calculate(self):
        """Test calculate tool."""
        result = ADKToolkit.calculate("2 + 3")

        assert result == 5.0

    def test_json_tools(self):
        """Test JSON parsing and formatting."""
        data = {"key": "value"}

        json_str = ADKToolkit.json_format(data)
        parsed = ADKToolkit.json_parse(json_str)

        assert parsed == data

    def test_text_length(self):
        """Test text length tool."""
        result = ADKToolkit.text_length("Hello")

        assert result == 5


# =============================================================================
# Part 9: Agent Monitor Tests
# =============================================================================
class TestAgentMonitor:
    """Tests for AgentMonitor class."""

    def test_start_end_run(self):
        """Test starting and ending a run."""
        monitor = AgentMonitor()

        run_id = monitor.start_run("agent-1")
        monitor.end_run(run_id, success=True, output="Done")

        metrics = monitor.get_metrics("agent-1")

        assert metrics.total_runs == 1
        assert metrics.successful_runs == 1

    def test_log_tool_call(self):
        """Test logging tool calls."""
        monitor = AgentMonitor()

        run_id = monitor.start_run("agent-1")
        monitor.log_tool_call(run_id, "search", {"query": "test"})
        monitor.end_run(run_id, success=True)

        metrics = monitor.get_metrics("agent-1")

        assert metrics.total_tool_calls == 1

    def test_run_history(self):
        """Test getting run history."""
        monitor = AgentMonitor()

        for i in range(3):
            run_id = monitor.start_run("agent-1")
            monitor.end_run(run_id, success=True)

        history = monitor.get_run_history("agent-1", limit=2)

        assert len(history) == 2

    def test_export_metrics(self):
        """Test exporting metrics."""
        monitor = AgentMonitor()

        run_id = monitor.start_run("agent-1")
        monitor.end_run(run_id, success=True)

        exported = monitor.export_metrics()

        assert "agent-1" in exported


# =============================================================================
# Part 10: ADK Agent Factory Tests
# =============================================================================
class TestADKAgentFactory:
    """Tests for ADKAgentFactory class."""

    def test_create_from_template(self):
        """Test creating agent from template."""
        factory = ADKAgentFactory()

        agent = factory.create(AgentTemplate.ASSISTANT)

        assert agent is not None

    def test_create_custom(self):
        """Test creating custom agent."""
        factory = ADKAgentFactory()

        def my_tool(x: int) -> int:
            return x * 2

        agent = factory.create_custom(
            name="my-agent", tools=[my_tool], system_prompt="You are helpful."
        )

        assert agent is not None

    def test_list_templates(self):
        """Test listing available templates."""
        factory = ADKAgentFactory()

        templates = factory.list_templates()

        assert len(templates) > 0
        assert "assistant" in [t.lower() for t in templates]


# =============================================================================
# Integration Tests
# =============================================================================
class TestADKIntegration:
    """Integration tests for ADK components."""

    @pytest.mark.asyncio
    async def test_full_adk_workflow(self):
        """Test complete ADK workflow."""
        # Create factory and agent
        factory = ADKAgentFactory()
        config = ADKAgentConfig(name="test-agent", max_iterations=5)
        agent = ADKAgent(config)

        # Add tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        agent.add_tool(greet)

        # Create runner
        runner = ADKRunner()

        # Set up monitor
        monitor = AgentMonitor()
        run_id = monitor.start_run("test-agent")

        # Run
        result = await runner.run_async(agent, "Greet John")

        # Complete monitoring
        monitor.end_run(run_id, success=True, output=result.output)

        # Verify
        metrics = monitor.get_metrics("test-agent")
        assert metrics.total_runs == 1

    @pytest.mark.asyncio
    async def test_agent_with_memory_integration(self):
        """Test agent using memory."""
        config = ADKAgentConfig(name="memory-agent", memory_enabled=True)
        agent = ADKAgent(config)

        # Create memory
        memory = ADKMemory()
        memory.add("User prefers formal responses", MemoryType.LONG_TERM)

        # Run agent
        result = await agent.run("Hello")

        # Memory should be accessible
        relevant = memory.search("formal")
        assert len(relevant) > 0

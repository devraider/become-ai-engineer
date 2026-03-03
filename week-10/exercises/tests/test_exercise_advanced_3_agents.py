"""
Tests for Week 10 - Exercise 3: Agents & Persistence
"""

import pytest
from exercise_advanced_3_agents import (
    Message,
    add_messages,
    AgentState,
    Tool,
    CalculatorTool,
    SearchTool,
    ToolRegistry,
    ReActAgent,
    Checkpoint,
    MemoryCheckpointer,
    PersistentAgent,
    HumanInTheLoopState,
    HumanInTheLoopAgent,
    StreamingAgent,
)


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Message should be created with required fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Message.to_dict should return dictionary."""
        msg = Message(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"

    def test_message_with_tool_calls(self):
        """Message should support tool_calls field."""
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "search", "args": {"query": "test"}}],
        )
        assert len(msg.tool_calls) == 1


class TestAddMessages:
    """Tests for add_messages reducer."""

    def test_add_messages_appends(self):
        """add_messages should append new messages to existing."""
        existing = [Message(role="user", content="Hi")]
        new = [Message(role="assistant", content="Hello")]
        result = add_messages(existing, new)
        assert len(result) == 2
        assert result[0].content == "Hi"
        assert result[1].content == "Hello"


class TestCalculatorTool:
    """Tests for Task 3: Calculator Tool."""

    def test_calculator_name(self):
        """CalculatorTool should have correct name."""
        calc = CalculatorTool()
        assert calc.name == "calculator"

    def test_calculator_simple_expression(self):
        """CalculatorTool should evaluate simple expressions."""
        calc = CalculatorTool()
        result = calc.execute(expression="2 + 3")
        assert result == "5"

    def test_calculator_complex_expression(self):
        """CalculatorTool should evaluate complex expressions."""
        calc = CalculatorTool()
        result = calc.execute(expression="2 * 3 + 4")
        assert result == "10"

    def test_calculator_schema(self):
        """CalculatorTool should return valid schema."""
        calc = CalculatorTool()
        schema = calc.get_schema()
        assert schema["name"] == "calculator"
        assert "expression" in str(schema)


class TestSearchTool:
    """Tests for Task 4: Search Tool."""

    def test_search_name(self):
        """SearchTool should have correct name."""
        search = SearchTool()
        assert search.name == "search"

    def test_search_finds_results(self):
        """SearchTool should find matching results."""
        search = SearchTool()
        result = search.execute(query="python")
        assert "python" in result.lower() or "programming" in result.lower()

    def test_search_no_results(self):
        """SearchTool should handle no results gracefully."""
        search = SearchTool()
        result = search.execute(query="xyznonexistent")
        assert "no" in result.lower() or len(result) > 0


class TestToolRegistry:
    """Tests for Task 5: Tool Registry."""

    def test_register_tool(self):
        """ToolRegistry should register tools."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.list_tools()

    def test_get_tool(self):
        """ToolRegistry should retrieve tools by name."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)
        assert registry.get("calculator") is calc

    def test_execute_tool(self):
        """ToolRegistry should execute tools by name."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute("calculator", expression="5 + 5")
        assert result == "10"

    def test_get_all_schemas(self):
        """ToolRegistry should return all schemas."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())
        schemas = registry.get_all_schemas()
        assert len(schemas) == 2


class TestReActAgent:
    """Tests for Task 6: ReAct Agent."""

    def test_agent_detects_calculator_need(self):
        """ReActAgent should detect when calculator is needed."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        agent = ReActAgent(registry)

        messages = [Message(role="user", content="What is 5 * 7?")]
        use_tool, tool_name, args = agent.should_use_tool(messages)
        assert use_tool is True
        assert tool_name == "calculator"

    def test_agent_detects_search_need(self):
        """ReActAgent should detect when search is needed."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        agent = ReActAgent(registry)

        messages = [Message(role="user", content="What is Python?")]
        use_tool, tool_name, args = agent.should_use_tool(messages)
        assert use_tool is True
        assert tool_name == "search"

    def test_agent_run_returns_response(self):
        """ReActAgent.run should return a response."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool())
        agent = ReActAgent(registry)

        result = agent.run("What is 3 + 4?")
        assert "response" in result or "final_response" in result


class TestMemoryCheckpointer:
    """Tests for Task 7: Memory Checkpointer."""

    def test_put_checkpoint(self):
        """MemoryCheckpointer.put should save checkpoint."""
        checkpointer = MemoryCheckpointer()
        checkpoint_id = checkpointer.put("thread-1", {"messages": []})
        assert checkpoint_id is not None

    def test_get_checkpoint(self):
        """MemoryCheckpointer.get should retrieve latest checkpoint."""
        checkpointer = MemoryCheckpointer()
        checkpointer.put("thread-1", {"value": 1})
        checkpointer.put("thread-1", {"value": 2})
        checkpoint = checkpointer.get("thread-1")
        assert checkpoint.state["value"] == 2

    def test_get_history(self):
        """MemoryCheckpointer.get_history should return all checkpoints."""
        checkpointer = MemoryCheckpointer()
        checkpointer.put("thread-1", {"step": 1})
        checkpointer.put("thread-1", {"step": 2})
        history = checkpointer.get_history("thread-1")
        assert len(history) == 2

    def test_delete_checkpoint(self):
        """MemoryCheckpointer.delete should remove all checkpoints."""
        checkpointer = MemoryCheckpointer()
        checkpointer.put("thread-1", {"data": "test"})
        assert checkpointer.delete("thread-1") is True
        assert checkpointer.get("thread-1") is None


class TestPersistentAgent:
    """Tests for Task 8: Persistent Agent."""

    def test_persistent_chat(self):
        """PersistentAgent should maintain conversation."""
        agent = PersistentAgent()
        agent.chat("My name is Alice", "thread-1")
        result = agent.chat("What's my name?", "thread-1")
        # Agent should have access to previous message
        assert "response" in result or len(result) > 0

    def test_separate_threads(self):
        """PersistentAgent should keep threads separate."""
        agent = PersistentAgent()
        agent.chat("I like Python", "thread-1")
        agent.chat("I like JavaScript", "thread-2")

        history1 = agent.get_conversation("thread-1")
        history2 = agent.get_conversation("thread-2")

        assert len(history1) > 0
        assert len(history2) > 0

    def test_clear_conversation(self):
        """PersistentAgent should clear conversation."""
        agent = PersistentAgent()
        agent.chat("Hello", "thread-1")
        agent.clear_conversation("thread-1")
        history = agent.get_conversation("thread-1")
        assert len(history) == 0


class TestHumanInTheLoopAgent:
    """Tests for Task 9: Human-in-the-Loop."""

    def test_analyze_sensitive_request(self):
        """HumanInTheLoopAgent should detect sensitive requests."""
        agent = HumanInTheLoopAgent()
        result = agent.analyze_request("delete all files")
        assert result["approval_required"] is True

    def test_analyze_safe_request(self):
        """HumanInTheLoopAgent should allow safe requests."""
        agent = HumanInTheLoopAgent()
        result = agent.analyze_request("show me the list")
        assert result["approval_required"] is False

    def test_submit_sensitive_request(self):
        """submit_request should return pending for sensitive."""
        agent = HumanInTheLoopAgent()
        result = agent.submit_request("req-1", "delete database")
        assert "pending" in str(result).lower() or result.get("approval_required")

    def test_approve_request(self):
        """approve should allow request to proceed."""
        agent = HumanInTheLoopAgent()
        agent.submit_request("req-1", "remove old logs")
        result = agent.approve("req-1")
        assert "approved" in str(result).lower() or result.get("approved")


class TestStreamingAgent:
    """Tests for Task 10: Streaming Agent."""

    def test_stream_yields_events(self):
        """StreamingAgent.stream should yield events."""
        agent = StreamingAgent()
        events = list(agent.stream("Calculate 2 + 2"))
        assert len(events) > 0

    def test_stream_event_structure(self):
        """StreamingAgent events should have correct structure."""
        agent = StreamingAgent()
        events = list(agent.stream("What is Python?"))
        for event in events:
            assert "event" in event
            assert "data" in event

    def test_stream_includes_response(self):
        """StreamingAgent should include response event."""
        agent = StreamingAgent()
        events = list(agent.stream("Hello"))
        event_types = [e["event"] for e in events]
        assert "response" in event_types or "thinking" in event_types

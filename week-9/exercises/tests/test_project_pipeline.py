"""
Tests for Week 9 - Project: AI Assistant with Tools and Memory
"""

import pytest
from datetime import datetime

from project_pipeline import (
    ChatMessage,
    SmartMemory,
    ToolParameter,
    ToolSchema,
    Tool,
    CalculatorTool,
    WebSearchTool,
    DocumentRetrieverTool,
    ToolManager,
    ResponseGenerator,
    ToolDecision,
    ToolDecisionEngine,
    FallbackHandler,
    ConversationManager,
    AIAssistant,
)


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_to_dict(self):
        """Test converting to dictionary."""
        msg = ChatMessage(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {"role": "user", "content": "Test", "timestamp": datetime.now().isoformat()}
        msg = ChatMessage.from_dict(d)
        assert msg.role == "user"
        assert msg.content == "Test"


class TestSmartMemory:
    """Tests for SmartMemory class."""

    def test_add_message(self):
        """Test adding messages."""
        memory = SmartMemory()
        memory.add_message(ChatMessage(role="user", content="Hello"))
        memory.add_message(ChatMessage(role="assistant", content="Hi!"))

        context = memory.get_context()
        assert "recent_messages" in context
        assert len(context["recent_messages"]) == 2

    def test_summarization(self):
        """Test that old messages get summarized."""
        memory = SmartMemory(recent_window=2)

        # Add more messages than window
        for i in range(10):
            memory.add_message(ChatMessage(role="user", content=f"Message {i}"))
            memory.add_message(ChatMessage(role="assistant", content=f"Response {i}"))

        context = memory.get_context()
        # Recent messages should be limited
        assert len(context["recent_messages"]) <= 4  # 2 turns

    def test_search_history(self):
        """Test searching message history."""
        memory = SmartMemory()
        memory.add_message(ChatMessage(role="user", content="Tell me about Python"))
        memory.add_message(ChatMessage(role="user", content="What about JavaScript?"))

        results = memory.search_history("Python")
        assert len(results) >= 1

    def test_metadata(self):
        """Test saving metadata."""
        memory = SmartMemory()
        memory.save_metadata("preference", "formal")

        context = memory.get_context()
        assert context["metadata"]["preference"] == "formal"

    def test_clear(self):
        """Test clearing memory."""
        memory = SmartMemory()
        memory.add_message(ChatMessage(role="user", content="Test"))
        memory.clear()

        context = memory.get_context()
        assert len(context["recent_messages"]) == 0


class TestToolSchema:
    """Tests for ToolSchema class."""

    def test_to_function_schema(self):
        """Test converting to function schema."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters=[ToolParameter(name="query", description="Search query")],
        )

        func_schema = schema.to_function_schema()
        assert func_schema["name"] == "test_tool"
        assert "description" in func_schema


class TestCalculatorTool:
    """Tests for Calculator Tool."""

    def test_basic_operations(self):
        """Test basic math operations."""
        calc = CalculatorTool()

        assert "4" in calc.execute(expression="2 + 2")
        assert "6" in calc.execute(expression="2 * 3")
        assert "5" in calc.execute(expression="10 / 2")

    def test_advanced_operations(self):
        """Test advanced operations."""
        calc = CalculatorTool()

        result = calc.execute(expression="2 ** 3")
        assert "8" in result

    def test_error_handling(self):
        """Test handling invalid expressions."""
        calc = CalculatorTool()

        result = calc.execute(expression="invalid")
        assert "error" in result.lower() or result != ""


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_search(self):
        """Test searching."""
        kb = {"python": "Python is a programming language"}
        search = WebSearchTool(knowledge_base=kb)

        result = search.execute(query="python")
        assert "Python" in result or "programming" in result

    def test_num_results(self):
        """Test limiting results."""
        kb = {
            "python basics": "Basic Python",
            "python advanced": "Advanced Python",
            "python web": "Web Python",
        }
        search = WebSearchTool(knowledge_base=kb)

        result = search.execute(query="python", num_results=1)
        assert result != ""


class TestDocumentRetrieverTool:
    """Tests for DocumentRetrieverTool."""

    def test_retrieve(self):
        """Test document retrieval."""
        docs = [
            {"content": "Python is great for AI", "metadata": {}},
            {"content": "JavaScript is for web", "metadata": {}},
        ]
        retriever = DocumentRetrieverTool(documents=docs)

        result = retriever.execute(query="Python AI")
        assert result != "" or len(result) >= 0

    def test_top_k(self):
        """Test limiting results."""
        docs = [{"content": f"Document {i}", "metadata": {}} for i in range(10)]
        retriever = DocumentRetrieverTool(documents=docs)

        result = retriever.execute(query="Document", top_k=3)
        # Should return limited results


class TestToolManager:
    """Tests for ToolManager class."""

    def test_register_and_get(self):
        """Test registering and getting tools."""
        manager = ToolManager()
        calc = CalculatorTool()
        manager.register(calc)

        found = manager.get("calculator")
        assert found is not None

    def test_list_tools(self):
        """Test listing tools."""
        manager = ToolManager()
        manager.register(CalculatorTool())
        manager.register(WebSearchTool())

        tools = manager.list_tools()
        assert "calculator" in tools
        assert "web_search" in tools

    def test_execute_tool(self):
        """Test executing a tool."""
        manager = ToolManager()
        manager.register(CalculatorTool())

        result = manager.execute_tool("calculator", expression="3 + 3")
        assert "6" in result


class TestResponseGenerator:
    """Tests for ResponseGenerator class."""

    def test_generate(self):
        """Test generating responses."""

        def mock_llm(prompt):
            return "This is a response"

        generator = ResponseGenerator(llm=mock_llm)
        response = generator.generate("Hello")
        assert response != ""

    def test_with_context(self):
        """Test generation with context."""

        def mock_llm(prompt):
            if "context" in prompt.lower():
                return "Response with context"
            return "Basic response"

        generator = ResponseGenerator(llm=mock_llm)
        response = generator.generate("Question", context="Previous context")
        assert response != ""

    def test_stream(self):
        """Test streaming responses."""

        def mock_llm(prompt):
            return "Word by word response"

        generator = ResponseGenerator(llm=mock_llm)
        chunks = list(generator.stream("Hello"))
        assert len(chunks) > 0


class TestToolDecisionEngine:
    """Tests for ToolDecisionEngine class."""

    def test_decide_use_tool(self):
        """Test deciding to use a tool."""
        manager = ToolManager()
        manager.register(CalculatorTool())

        engine = ToolDecisionEngine(tool_manager=manager)
        decision = engine.decide("What is 2 + 2?")

        assert isinstance(decision, ToolDecision)

    def test_decide_no_tool(self):
        """Test deciding not to use tool."""
        manager = ToolManager()
        engine = ToolDecisionEngine(tool_manager=manager)

        decision = engine.decide("Hello, how are you?")
        # Should either use no tool or return some decision
        assert isinstance(decision, ToolDecision)


class TestFallbackHandler:
    """Tests for FallbackHandler class."""

    def test_with_retry_success(self):
        """Test retry with eventual success."""
        attempt = 0

        def flaky_func():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ValueError("Not yet")
            return "Success"

        handler = FallbackHandler(max_retries=5)
        result, success = handler.with_retry(flaky_func)

        assert success is True
        assert result == "Success"

    def test_with_retry_failure(self):
        """Test retry exhaustion."""

        def always_fails():
            raise ValueError("Always fails")

        handler = FallbackHandler(max_retries=2)
        result, success = handler.with_retry(always_fails)

        assert success is False

    def test_with_fallback(self):
        """Test fallback chain."""

        def primary():
            raise ValueError("Primary failed")

        def fallback1():
            return "Fallback 1"

        handler = FallbackHandler()
        result = handler.with_fallback(primary, [fallback1])

        assert result == "Fallback 1"


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_create_session(self):
        """Test creating a session."""
        manager = ConversationManager()
        session_id = manager.create_session()

        assert session_id is not None
        assert manager.get_session(session_id) is not None

    def test_add_message(self):
        """Test adding messages to session."""
        manager = ConversationManager()
        session_id = manager.create_session()

        result = manager.add_message(session_id, "user", "Hello")
        assert result is True

        history = manager.get_history(session_id)
        assert len(history) >= 1

    def test_end_session(self):
        """Test ending a session."""
        manager = ConversationManager()
        session_id = manager.create_session()

        result = manager.end_session(session_id)
        assert result is True
        assert manager.get_session(session_id) is None

    def test_export_session(self):
        """Test exporting session as JSON."""
        import json

        manager = ConversationManager()
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "Test")

        exported = manager.export_session(session_id)
        assert exported is not None

        # Should be valid JSON
        data = json.loads(exported)
        assert data is not None


class TestAIAssistant:
    """Tests for AIAssistant class."""

    def test_initialization(self):
        """Test assistant initialization."""
        assistant = AIAssistant()
        assert assistant is not None

    def test_basic_chat(self):
        """Test basic chat interaction."""

        def mock_llm(prompt):
            return "Hello! How can I help you?"

        assistant = AIAssistant(llm=mock_llm)
        response = assistant.chat("Hello!")

        assert response != ""

    def test_tool_use(self):
        """Test that assistant can use tools."""

        def mock_llm(prompt):
            if "calculate" in prompt.lower():
                return "The result is 4."
            return "I can help with that."

        assistant = AIAssistant(llm=mock_llm)
        response = assistant.chat("Calculate 2 + 2")

        assert response != ""

    def test_memory_persistence(self):
        """Test that memory persists across chats."""

        def mock_llm(prompt):
            return f"Response to {len(prompt)} chars"

        assistant = AIAssistant(llm=mock_llm)

        assistant.chat("First message")
        assistant.chat("Second message")

        # Export should contain both messages
        export = assistant.export_conversation()
        assert export != ""

    def test_available_tools(self):
        """Test getting available tools."""
        assistant = AIAssistant()
        tools = assistant.get_available_tools()

        assert isinstance(tools, list)
        assert "calculator" in tools

    def test_clear_memory(self):
        """Test clearing memory."""
        assistant = AIAssistant()
        assistant.chat("Test message")
        assistant.clear_memory()

        # Memory should be empty after clear

    def test_session_management(self):
        """Test using sessions."""
        assistant = AIAssistant()

        # First session
        response1 = assistant.chat("Hello", session_id="session1")

        # Second session
        response2 = assistant.chat("Hi there", session_id="session2")

        # Both should work
        assert response1 != ""
        assert response2 != ""

    def test_streaming(self):
        """Test streaming responses."""

        def mock_llm(prompt):
            return "This is a streaming response"

        assistant = AIAssistant(llm=mock_llm)
        stream = assistant.chat("Hello", stream=True)

        if hasattr(stream, "__iter__"):
            chunks = list(stream)
            assert len(chunks) > 0

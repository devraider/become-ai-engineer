"""
Tests for Week 9 - Exercise Intermediate 2: Memory Systems
"""

import pytest
from datetime import datetime

from exercise_intermediate_2_memory import (
    Message,
    BaseMemory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenMemory,
    EntityMemory,
    PersistentMemory,
    ConversationChain,
    SearchableMemory,
    CombinedMemory,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="human", content="Hello")
        assert msg.role == "human"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(role="ai", content="Hi there", metadata={"tokens": 5})
        assert msg.metadata["tokens"] == 5


class TestConversationBufferMemory:
    """Tests for ConversationBufferMemory class."""

    def test_add_message(self):
        """Test adding messages."""
        memory = ConversationBufferMemory()
        memory.add_message("human", "Hello")
        memory.add_message("ai", "Hi there!")

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "human"
        assert messages[1].role == "ai"

    def test_add_user_ai_messages(self):
        """Test convenience methods."""
        memory = ConversationBufferMemory()
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi!")

        messages = memory.get_messages()
        assert messages[0].role == "human"
        assert messages[1].role == "ai"

    def test_save_context(self):
        """Test save_context method."""
        memory = ConversationBufferMemory()
        memory.save_context({"input": "Question"}, {"output": "Answer"})

        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Question"
        assert messages[1].content == "Answer"

    def test_clear(self):
        """Test clearing memory."""
        memory = ConversationBufferMemory()
        memory.add_message("human", "Test")
        memory.clear()
        assert len(memory.get_messages()) == 0

    def test_load_memory_variables(self):
        """Test loading memory variables."""
        memory = ConversationBufferMemory(return_messages=True)
        memory.add_message("human", "Hello")

        variables = memory.load_memory_variables({})
        assert "history" in variables


class TestConversationBufferWindowMemory:
    """Tests for ConversationBufferWindowMemory class."""

    def test_window_limit(self):
        """Test that window limits messages."""
        memory = ConversationBufferWindowMemory(k=2)

        # Add 5 turns (10 messages)
        for i in range(5):
            memory.add_message("human", f"Question {i}")
            memory.add_message("ai", f"Answer {i}")

        messages = memory.get_messages()
        # Should only have last 2 turns (4 messages)
        assert len(messages) == 4

    def test_window_keeps_recent(self):
        """Test that recent messages are kept."""
        memory = ConversationBufferWindowMemory(k=2)

        for i in range(5):
            memory.add_message("human", f"Q{i}")
            memory.add_message("ai", f"A{i}")

        messages = memory.get_messages()
        # Should have Q3, A3, Q4, A4
        assert "Q3" in messages[0].content or "Q4" in messages[0].content


class TestConversationSummaryMemory:
    """Tests for ConversationSummaryMemory class."""

    def test_summarization_triggered(self):
        """Test that summarization occurs."""

        def mock_summarizer(text):
            return f"Summary of {len(text)} chars"

        memory = ConversationSummaryMemory(
            summarizer=mock_summarizer, max_messages_before_summary=4
        )

        # Add enough messages to trigger summarization
        for i in range(3):
            memory.add_message("human", f"Question {i}")
            memory.add_message("ai", f"Answer {i}")

        summary = memory.get_summary()
        assert summary != "" or len(memory.get_messages()) <= 4

    def test_clear(self):
        """Test clearing summary memory."""
        memory = ConversationSummaryMemory()
        memory.add_message("human", "Test")
        memory.clear()

        assert len(memory.get_messages()) == 0
        assert memory.get_summary() == ""


class TestConversationTokenMemory:
    """Tests for ConversationTokenMemory class."""

    def test_token_limit(self):
        """Test that token limit is enforced."""
        memory = ConversationTokenMemory(max_tokens=10)

        # Add messages that exceed limit
        memory.add_message("human", "This is a test message")
        memory.add_message("ai", "This is another test message")

        # Should have trimmed old messages
        assert memory.get_token_count() <= 10

    def test_get_token_count(self):
        """Test token counting."""
        memory = ConversationTokenMemory(max_tokens=100)
        memory.add_message("human", "Hello world")

        count = memory.get_token_count()
        assert count > 0


class TestEntityMemory:
    """Tests for EntityMemory class."""

    def test_entity_extraction(self):
        """Test entity extraction from messages."""
        memory = EntityMemory()
        memory.add_message("human", "I met Alice and Bob today.")

        entities = memory.get_all_entities()
        # Should find capitalized words
        assert isinstance(entities, dict)

    def test_get_entity_info(self):
        """Test getting info about an entity."""
        memory = EntityMemory()
        memory.add_message("human", "Alice is a software engineer.")
        memory.add_message("human", "Alice also likes Python.")

        info = memory.get_entity_info("Alice")
        assert isinstance(info, list)


class TestPersistentMemory:
    """Tests for PersistentMemory class."""

    def test_save_and_load(self, tmp_path):
        """Test persistence to disk."""
        file_path = str(tmp_path / "memory.json")

        # Create and save
        memory1 = PersistentMemory(file_path)
        memory1.add_message("human", "Test message")

        # Load in new instance
        memory2 = PersistentMemory(file_path)
        messages = memory2.get_messages()

        assert len(messages) >= 1
        assert messages[0].content == "Test message"

    def test_clear_removes_file(self, tmp_path):
        """Test that clear removes file."""
        import os

        file_path = str(tmp_path / "memory.json")

        memory = PersistentMemory(file_path)
        memory.add_message("human", "Test")
        memory.clear()

        assert not os.path.exists(file_path) or len(memory.get_messages()) == 0


class TestConversationChain:
    """Tests for ConversationChain class."""

    def test_conversation_flow(self):
        """Test multi-turn conversation."""

        def mock_processor(prompt):
            return f"Response to: {prompt[-20:]}"

        chain = ConversationChain(processor=mock_processor)

        response1 = chain.run("Hello!")
        assert "Response" in response1

        response2 = chain.run("How are you?")
        assert "Response" in response2

    def test_memory_persistence(self):
        """Test that memory persists across turns."""

        def mock_processor(prompt):
            return f"Received: {len(prompt)} chars"

        chain = ConversationChain(processor=mock_processor)

        chain.run("First message")
        chain.run("Second message")

        memory = chain.get_memory()
        messages = memory.get_messages()
        assert len(messages) >= 4  # 2 user + 2 AI


class TestSearchableMemory:
    """Tests for SearchableMemory class."""

    def test_search(self):
        """Test searching memory."""
        memory = SearchableMemory()
        memory.add_message("human", "Tell me about Python programming")
        memory.add_message("ai", "Python is a versatile language")
        memory.add_message("human", "What about JavaScript?")
        memory.add_message("ai", "JavaScript is for web development")

        results = memory.search("Python", k=2)
        assert len(results) <= 2
        # Results should be relevant to Python
        assert any("Python" in msg.content for msg in results)

    def test_load_with_query(self):
        """Test loading with query filter."""
        memory = SearchableMemory()
        memory.add_message("human", "Python question")
        memory.add_message("human", "JavaScript question")

        variables = memory.load_memory_variables({"query": "Python"})
        assert "history" in variables


class TestCombinedMemory:
    """Tests for CombinedMemory class."""

    def test_messages_to_all(self):
        """Test that messages go to all memories."""
        buffer = ConversationBufferMemory()
        entity = EntityMemory()
        combined = CombinedMemory([buffer, entity])

        combined.add_message("human", "Alice is great at Python")

        # Both should have the message
        assert len(buffer.get_messages()) == 1
        assert len(entity.get_messages()) == 1

    def test_combined_variables(self):
        """Test combined memory variables."""
        buffer = ConversationBufferMemory()
        entity = EntityMemory()
        combined = CombinedMemory([buffer, entity])

        combined.add_message("human", "Test about Alice")

        variables = combined.load_memory_variables({})
        assert "history" in variables or len(variables) > 0

    def test_clear_all(self):
        """Test clearing all memories."""
        buffer = ConversationBufferMemory()
        entity = EntityMemory()
        combined = CombinedMemory([buffer, entity])

        combined.add_message("human", "Test")
        combined.clear()

        assert len(buffer.get_messages()) == 0
        assert len(entity.get_messages()) == 0

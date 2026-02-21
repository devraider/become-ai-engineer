"""
Tests for Week 5 Exercise 2: Chat & Conversations
"""

import pytest
from exercise_intermediate_2_chat import (
    Message,
    ConversationHistory,
    ChatBot,
    summarize_conversation,
    format_for_api,
    create_conversation_fork,
)


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = Message(role="assistant", content="Hi there")
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_from_dict(self):
        data = {"role": "user", "content": "Test", "metadata": {"key": "value"}}
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Test"


class TestConversationHistory:
    def test_add_message(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        assert len(history.get_messages()) == 1

    def test_get_messages(self):
        history = ConversationHistory()
        history.add_message("user", "First")
        history.add_message("assistant", "Second")

        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "First"

    def test_get_n_messages(self):
        history = ConversationHistory()
        for i in range(10):
            history.add_message("user", f"Message {i}")

        messages = history.get_messages(n=3)
        assert len(messages) == 3

    def test_max_messages_limit(self):
        history = ConversationHistory(max_messages=5)
        for i in range(10):
            history.add_message("user", f"Message {i}")

        messages = history.get_messages()
        assert len(messages) <= 5

    def test_clear(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.clear()
        assert len(history.get_messages()) == 0

    def test_context_window(self):
        history = ConversationHistory()
        history.add_message("user", "Short")
        history.add_message("assistant", "Also short")

        context = history.get_context_window(max_tokens=100)
        assert len(context) >= 1

    def test_to_list(self):
        history = ConversationHistory()
        history.add_message("user", "Test")

        result = history.to_list()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_from_list(self):
        data = [
            {"role": "user", "content": "Hello", "metadata": {}},
            {"role": "assistant", "content": "Hi", "metadata": {}},
        ]
        history = ConversationHistory.from_list(data)
        assert len(history.get_messages()) == 2


class TestChatBot:
    def test_initialization(self):
        bot = ChatBot(system_prompt="Test assistant")
        assert bot is not None

    def test_chat_returns_response(self):
        bot = ChatBot()
        response = bot.chat("Hello")
        assert isinstance(response, str)

    def test_history_maintained(self):
        bot = ChatBot()
        bot.chat("First message")
        bot.chat("Second message")

        history = bot.get_history()
        assert len(history) >= 2

    def test_reset_conversation(self):
        bot = ChatBot()
        bot.chat("Hello")
        bot.reset_conversation()

        history = bot.get_history()
        assert len(history) == 0


class TestSummarizeConversation:
    def test_returns_string(self):
        history = ConversationHistory()
        history.add_message("user", "What is Python?")
        history.add_message("assistant", "Python is a programming language.")

        result = summarize_conversation(history)
        assert isinstance(result, str)


class TestFormatForApi:
    def test_gemini_format(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")

        result = format_for_api(history, "gemini")
        assert isinstance(result, list)
        if result:  # If implemented
            assert "parts" in result[0] or "content" in result[0]

    def test_openai_format(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")

        result = format_for_api(history, "openai")
        assert isinstance(result, list)
        if result:  # If implemented
            assert "content" in result[0]


class TestCreateConversationFork:
    def test_fork_at_index(self):
        history = ConversationHistory()
        history.add_message("user", "Message 0")
        history.add_message("assistant", "Response 0")
        history.add_message("user", "Message 1")
        history.add_message("assistant", "Response 1")

        forked = create_conversation_fork(history, 2)

        if forked:  # If implemented
            messages = forked.get_messages()
            assert len(messages) <= 2

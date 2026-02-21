"""
Week 5 Exercise 2 (Intermediate): Chat & Conversations - SOLUTIONS
=================================================================
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str
    content: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create Message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class ConversationHistory:
    """Conversation history manager."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.messages: List[Message] = []

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> None:
        """Add a message to the history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)

        # Trim if over limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_messages(self, n: Optional[int] = None) -> List[Message]:
        """Get messages from history."""
        if n is None:
            return self.messages.copy()
        return self.messages[-n:]

    def get_context_window(
        self, max_tokens: int, chars_per_token: int = 4
    ) -> List[Message]:
        """Get messages that fit within a token budget."""
        max_chars = max_tokens * chars_per_token
        result = []
        total_chars = 0

        for msg in reversed(self.messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > max_chars:
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

    def to_list(self) -> List[Dict]:
        """Convert history to list of dicts for serialization."""
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_list(
        cls, data: List[Dict], max_messages: int = 100
    ) -> "ConversationHistory":
        """Create history from list of dicts."""
        history = cls(max_messages=max_messages)
        for item in data:
            history.messages.append(Message.from_dict(item))
        return history


class ChatBot:
    """Chatbot with conversation management."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 50,
        max_context_tokens: int = 4000,
    ):
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.history = ConversationHistory(max_messages=max_history)

    def _build_prompt(self, user_message: str) -> str:
        """Build the full prompt including system prompt and history."""
        parts = [f"System: {self.system_prompt}\n"]

        context = self.history.get_context_window(self.max_context_tokens)
        for msg in context:
            parts.append(f"{msg.role.capitalize()}: {msg.content}")

        parts.append(f"User: {user_message}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def chat(self, message: str) -> str:
        """Send a message and get a response (mock implementation)."""
        self.history.add_message("user", message)

        # Mock response - in real implementation, call API
        response = (
            f"I received your message: '{message[:50]}...'. This is a mock response."
        )

        self.history.add_message("assistant", response)
        return response

    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self.history.get_messages()

    def reset_conversation(self) -> None:
        """Start a new conversation."""
        self.history.clear()


class GeminiChatBot(ChatBot):
    """Chatbot using Gemini API."""

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 50,
        max_context_tokens: int = 4000,
        model_name: str = "gemini-1.5-flash",
    ):
        super().__init__(system_prompt, max_history, max_context_tokens)
        self.model = None
        self.chat_session = None

        if GENAI_AVAILABLE:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")

            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt,
                )
                self.chat_session = self.model.start_chat()

    def chat(self, message: str) -> str:
        """Send a message to Gemini and get a response."""
        self.history.add_message("user", message)

        if self.chat_session:
            try:
                response = self.chat_session.send_message(message)
                response_text = response.text
            except Exception as e:
                response_text = f"Error: {e}"
        else:
            response_text = "Gemini API not configured. Please set GOOGLE_API_KEY."

        self.history.add_message("assistant", response_text)
        return response_text


def summarize_conversation(
    history: ConversationHistory, max_summary_length: int = 200
) -> str:
    """Create a summary of a conversation."""
    messages = history.get_messages()

    if not messages:
        return "No conversation to summarize."

    # Create a prompt for summarization
    conversation_text = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

    return f"""Please summarize the following conversation in {max_summary_length} characters or less:

{conversation_text}

Summary:"""


def extract_key_points(history: ConversationHistory) -> List[str]:
    """Extract key discussion points from a conversation."""
    messages = history.get_messages()

    # Simple keyword extraction (real implementation would use NLP or LLM)
    key_points = []
    for msg in messages:
        if msg.role == "user":
            # Extract questions or main topics
            if "?" in msg.content:
                key_points.append(f"Question: {msg.content[:100]}")

    return key_points if key_points else ["No key points extracted"]


def format_for_api(
    history: ConversationHistory, api_type: str = "gemini"
) -> List[Dict]:
    """Format conversation history for different API providers."""
    messages = history.get_messages()

    if api_type == "gemini":
        return [{"role": msg.role, "parts": [msg.content]} for msg in messages]
    elif api_type == "openai":
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    elif api_type == "anthropic":
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    else:
        return [msg.to_dict() for msg in messages]


def create_conversation_fork(
    history: ConversationHistory,
    fork_at_index: int,
) -> ConversationHistory:
    """Create a fork of the conversation at a specific point."""
    new_history = ConversationHistory(max_messages=history.max_messages)

    messages = history.get_messages()
    for msg in messages[:fork_at_index]:
        new_history.add_message(msg.role, msg.content, msg.metadata)

    return new_history

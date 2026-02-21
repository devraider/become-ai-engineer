"""
Week 5 Exercise 2 (Intermediate): Chat & Conversations
======================================================

Build multi-turn conversations with LLM APIs.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv

Run this file:
    python exercise_intermediate_2_chat.py

Run tests:
    python -m pytest tests/test_exercise_intermediate_2_chat.py -v
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# =============================================================================
# SETUP
# =============================================================================

def setup_gemini():
    """Set up Gemini API client."""
    if not GENAI_AVAILABLE:
        return None
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return None
        
    genai.configure(api_key=api_key)
    return genai


# =============================================================================
# YOUR TASKS - Complete the classes and functions below
# =============================================================================


@dataclass
class Message:
    """
    Task 1: Complete the Message dataclass.
    
    Represents a single message in a conversation.
    
    Attributes:
        role: Either "user" or "assistant"
        content: The message text
        metadata: Optional dict for extra info (timestamp, tokens, etc.)
    """
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        # TODO: Implement
        pass
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create Message from dictionary."""
        # TODO: Implement
        pass


class ConversationHistory:
    """
    Task 2: Implement a conversation history manager.
    
    This class manages the history of a conversation with features like:
    - Adding messages
    - Retrieving recent messages
    - Truncating history to fit context window
    - Serialization for storage
    """
    
    def __init__(self, max_messages: int = 100):
        """
        Initialize conversation history.
        
        Args:
            max_messages: Maximum messages to keep in history
        """
        # TODO: Initialize attributes
        pass
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to the history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata dict
        """
        # TODO: Implement
        pass
    
    def get_messages(self, n: Optional[int] = None) -> List[Message]:
        """
        Get messages from history.
        
        Args:
            n: Number of recent messages to return (None = all)
        
        Returns:
            List of Message objects
        """
        # TODO: Implement
        pass
    
    def get_context_window(self, max_tokens: int, chars_per_token: int = 4) -> List[Message]:
        """
        Get messages that fit within a token budget.
        
        Args:
            max_tokens: Maximum tokens allowed
            chars_per_token: Approximate characters per token
        
        Returns:
            List of most recent messages that fit
        """
        # TODO: Implement
        pass
    
    def clear(self) -> None:
        """Clear all messages."""
        # TODO: Implement
        pass
    
    def to_list(self) -> List[Dict]:
        """Convert history to list of dicts for serialization."""
        # TODO: Implement
        pass
    
    @classmethod
    def from_list(cls, data: List[Dict], max_messages: int = 100) -> "ConversationHistory":
        """Create history from list of dicts."""
        # TODO: Implement
        pass


class ChatBot:
    """
    Task 3: Implement a chatbot with conversation management.
    
    Features:
    - System prompt configuration
    - Conversation history management
    - Context window management
    - Response generation
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 50,
        max_context_tokens: int = 4000,
    ):
        """
        Initialize the chatbot.
        
        Args:
            system_prompt: Instructions for the assistant's behavior
            max_history: Maximum messages to keep in history
            max_context_tokens: Maximum tokens for context window
        """
        # TODO: Initialize attributes
        pass
    
    def _build_prompt(self, user_message: str) -> str:
        """
        Build the full prompt including system prompt and history.
        
        Args:
            user_message: The new user message
        
        Returns:
            Complete prompt string
        """
        # TODO: Implement
        pass
    
    def chat(self, message: str) -> str:
        """
        Send a message and get a response.
        
        This is a mock implementation that doesn't call the API.
        Override or extend for actual API integration.
        
        Args:
            message: User's message
        
        Returns:
            Assistant's response
        """
        # TODO: Implement (mock response for testing)
        pass
    
    def get_history(self) -> List[Message]:
        """Get conversation history."""
        # TODO: Implement
        pass
    
    def reset_conversation(self) -> None:
        """Start a new conversation."""
        # TODO: Implement
        pass


class GeminiChatBot(ChatBot):
    """
    Task 4: Implement a chatbot using Gemini API.
    
    Extends ChatBot with actual API integration.
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_history: int = 50,
        max_context_tokens: int = 4000,
        model_name: str = "gemini-1.5-flash",
    ):
        super().__init__(system_prompt, max_history, max_context_tokens)
        # TODO: Initialize Gemini model
        pass
    
    def chat(self, message: str) -> str:
        """
        Send a message to Gemini and get a response.
        
        Args:
            message: User's message
        
        Returns:
            Assistant's response
        """
        # TODO: Implement with actual API call
        pass


def summarize_conversation(history: ConversationHistory, max_summary_length: int = 200) -> str:
    """
    Task 5: Create a summary of a conversation.
    
    Args:
        history: Conversation history to summarize
        max_summary_length: Maximum characters in summary
    
    Returns:
        A brief summary of the conversation
        
    Note: This should return a prompt for summarization, 
    not actually call the API (unless you want to test it)
    """
    # TODO: Implement
    pass


def extract_key_points(history: ConversationHistory) -> List[str]:
    """
    Task 6: Extract key discussion points from a conversation.
    
    Args:
        history: Conversation history
    
    Returns:
        List of key points discussed
        
    Note: Return a prompt for extraction, or implement simple keyword extraction
    """
    # TODO: Implement
    pass


def format_for_api(history: ConversationHistory, api_type: str = "gemini") -> List[Dict]:
    """
    Task 7: Format conversation history for different API providers.
    
    Args:
        history: Conversation history
        api_type: "gemini", "openai", or "anthropic"
    
    Returns:
        List of message dicts formatted for the specified API
        
    Formats:
        - Gemini: [{"role": "user", "parts": ["text"]}]
        - OpenAI: [{"role": "user", "content": "text"}]
        - Anthropic: [{"role": "user", "content": "text"}]
    """
    # TODO: Implement
    pass


def create_conversation_fork(
    history: ConversationHistory,
    fork_at_index: int,
) -> ConversationHistory:
    """
    Task 8: Create a fork of the conversation at a specific point.
    
    Useful for exploring alternative conversation paths.
    
    Args:
        history: Original conversation
        fork_at_index: Index to fork at (keep messages 0 to fork_at_index)
    
    Returns:
        New ConversationHistory with truncated history
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Exercise 2: Chat & Conversations")
    print("=" * 60)
    
    print("\n1. Testing Message dataclass...")
    msg = Message(role="user", content="Hello!")
    if hasattr(msg, 'to_dict') and msg.to_dict():
        print(f"   Message dict: {msg.to_dict()}")
    
    print("\n2. Testing ConversationHistory...")
    history = ConversationHistory(max_messages=10)
    history.add_message("user", "What is Python?")
    history.add_message("assistant", "Python is a programming language.")
    history.add_message("user", "Tell me more.")
    
    messages = history.get_messages()
    if messages:
        print(f"   Messages in history: {len(messages)}")
        for msg in messages:
            print(f"   - {msg.role}: {msg.content[:30]}...")
    
    print("\n3. Testing ChatBot...")
    bot = ChatBot(system_prompt="You are a Python tutor.")
    
    response = bot.chat("What are decorators?")
    if response:
        print(f"   Bot response: {response[:100]}...")
    
    print("\n4. Testing context window...")
    context = history.get_context_window(max_tokens=100)
    if context:
        print(f"   Messages in context window: {len(context)}")
    
    print("\n5. Testing format_for_api...")
    formatted = format_for_api(history, "openai")
    if formatted:
        print(f"   OpenAI format: {formatted[0]}")
    
    print("\n6. Testing with Gemini API (if available)...")
    genai = setup_gemini()
    if genai:
        gemini_bot = GeminiChatBot(
            system_prompt="You are a helpful coding assistant."
        )
        try:
            response = gemini_bot.chat("What is a Python list?")
            print(f"   Gemini response: {response[:200]}...")
        except Exception as e:
            print(f"   API error: {e}")
    else:
        print("   Gemini API not available")
    
    print("\nComplete all TODOs and run tests to verify!")

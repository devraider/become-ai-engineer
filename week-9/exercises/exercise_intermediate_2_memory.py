"""
Week 9 - Exercise Intermediate 2: Memory Systems
================================================

Learn to implement conversation memory for LangChain applications.

Topics covered:
- Buffer memory (store all messages)
- Window memory (keep last k messages)
- Summary memory (compress old messages)
- Token-based memory
- Memory with custom storage
"""

from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


# =============================================================================
# Message Types
# =============================================================================
@dataclass
class Message:
    """Represents a chat message."""

    role: str  # "human", "ai", or "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# TASK 1: Implement Base Memory Class
# =============================================================================
class BaseMemory(ABC):
    """
    Abstract base class for memory implementations.

    All memory types inherit from this.
    """

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory.

        Args:
            role: Message role ("human" or "ai")
            content: Message content
        """
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """Get all messages in memory.

        Returns:
            List of Message objects
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory."""
        pass

    def add_user_message(self, content: str) -> None:
        """Convenience method to add human message.

        Args:
            content: User's message
        """
        # TODO: Call add_message with role="human"
        pass

    def add_ai_message(self, content: str) -> None:
        """Convenience method to add AI message.

        Args:
            content: AI's response
        """
        # TODO: Call add_message with role="ai"
        pass

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Save a conversation turn (input and output).

        Args:
            inputs: Dict with "input" key
            outputs: Dict with "output" key
        """
        # TODO: Add user message from inputs["input"]
        # Add AI message from outputs["output"]
        pass


# =============================================================================
# TASK 2: Implement Buffer Memory
# =============================================================================
class ConversationBufferMemory(BaseMemory):
    """
    Stores all messages without any limit.

    Simple but may grow unbounded.
    """

    def __init__(self, return_messages: bool = True):
        """Initialize buffer memory.

        Args:
            return_messages: If True, return Message objects; else string
        """
        # TODO: Initialize empty messages list and store return_messages flag
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message to buffer.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Create Message and append to messages list
        pass

    def get_messages(self) -> list[Message]:
        """Get all messages.

        Returns:
            Copy of messages list
        """
        # TODO: Return a copy of the messages list
        pass

    def clear(self) -> None:
        """Clear all messages."""
        # TODO: Empty the messages list
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load memory as variables for prompt.

        Args:
            inputs: Unused in buffer memory

        Returns:
            Dict with "history" key containing messages or string
        """
        # TODO: If return_messages, return {"history": self.get_messages()}
        # Else, format messages as string and return
        pass

    def _format_messages_as_string(self) -> str:
        """Format messages as human-readable string.

        Returns:
            Formatted conversation string
        """
        # TODO: Format each message as "Role: content" and join with newlines
        pass


# =============================================================================
# TASK 3: Implement Window Memory
# =============================================================================
class ConversationBufferWindowMemory(BaseMemory):
    """
    Keeps only the last k conversation turns.

    Useful for limiting context size.
    """

    def __init__(self, k: int = 5, return_messages: bool = True):
        """Initialize window memory.

        Args:
            k: Number of conversation turns to keep
            return_messages: If True, return Message objects
        """
        # TODO: Store k, return_messages, initialize messages list
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message, pruning old ones if needed.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add message
        # If messages > k*2 (k turns = k*2 messages), remove oldest pair
        pass

    def get_messages(self) -> list[Message]:
        """Get messages within window.

        Returns:
            List of last k turns of messages
        """
        # TODO: Return last k*2 messages
        pass

    def clear(self) -> None:
        """Clear all messages."""
        # TODO: Empty messages list
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load windowed memory as variables.

        Returns:
            Dict with "history" containing windowed messages
        """
        # TODO: Return {"history": self.get_messages()} or formatted string
        pass


# =============================================================================
# TASK 4: Implement Summary Memory
# =============================================================================
class ConversationSummaryMemory(BaseMemory):
    """
    Maintains a running summary of the conversation.

    Uses a summarizer function to compress history.
    """

    def __init__(self, summarizer=None, max_messages_before_summary: int = 6):
        """Initialize summary memory.

        Args:
            summarizer: Function that takes text and returns summary
            max_messages_before_summary: Trigger summarization after this many
        """
        # TODO: Store summarizer, max_messages_before_summary
        # Initialize messages list and summary string
        pass

    def _default_summarizer(self, text: str) -> str:
        """Default summarizer (just truncates).

        Args:
            text: Text to summarize

        Returns:
            Abbreviated summary
        """
        # TODO: Return first 200 chars with "..." if longer
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message, summarizing if threshold reached.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add message to list
        # If len(messages) >= max_messages_before_summary:
        #   - Summarize current messages
        #   - Clear messages and store summary
        pass

    def _summarize_messages(self) -> str:
        """Create summary of current messages.

        Returns:
            Summary string
        """
        # TODO: Format messages as text
        # Apply summarizer (or default) to get summary
        # Prepend existing summary if any
        pass

    def get_messages(self) -> list[Message]:
        """Get recent messages (not summarized ones).

        Returns:
            Current message buffer
        """
        # TODO: Return copy of messages
        pass

    def get_summary(self) -> str:
        """Get the current summary.

        Returns:
            Summary of older conversation
        """
        # TODO: Return summary string
        pass

    def clear(self) -> None:
        """Clear messages and summary."""
        # TODO: Clear both messages and summary
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load summary + recent messages.

        Returns:
            Dict with "history" and "summary" keys
        """
        # TODO: Return both summary and recent messages
        pass


# =============================================================================
# TASK 5: Implement Token-Based Memory
# =============================================================================
class ConversationTokenMemory(BaseMemory):
    """
    Keeps messages up to a token limit.

    More precise control over context size.
    """

    def __init__(self, max_tokens: int = 2000, token_counter=None):
        """Initialize token-limited memory.

        Args:
            max_tokens: Maximum tokens to keep
            token_counter: Function to count tokens (default: word count)
        """
        # TODO: Store max_tokens, token_counter (or default)
        # Initialize messages list
        pass

    def _default_token_counter(self, text: str) -> int:
        """Count tokens (simplified: word count).

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        # TODO: Return len(text.split())
        pass

    def _count_message_tokens(self, message: Message) -> int:
        """Count tokens in a message.

        Args:
            message: Message to count

        Returns:
            Token count
        """
        # TODO: Count tokens in role + content
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message, removing old ones if over token limit.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add new message
        # While total tokens > max_tokens, remove oldest message
        pass

    def get_messages(self) -> list[Message]:
        """Get messages within token limit.

        Returns:
            List of messages
        """
        # TODO: Return copy of messages
        pass

    def get_token_count(self) -> int:
        """Get current token count.

        Returns:
            Total tokens in memory
        """
        # TODO: Sum tokens of all messages
        pass

    def clear(self) -> None:
        """Clear all messages."""
        # TODO: Empty messages list
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load memory as variables.

        Returns:
            Dict with "history" and "token_count"
        """
        # TODO: Return messages and token count
        pass


# =============================================================================
# TASK 6: Implement Entity Memory
# =============================================================================
class EntityMemory(BaseMemory):
    """
    Tracks entities (people, places, things) mentioned in conversation.

    Helps maintain context about specific subjects.
    """

    def __init__(self, entity_extractor=None):
        """Initialize entity memory.

        Args:
            entity_extractor: Function to extract entities from text
        """
        # TODO: Store entity_extractor (or default)
        # Initialize messages list and entities dict
        pass

    def _default_entity_extractor(self, text: str) -> list[str]:
        """Extract entities (simplified: capitalized words).

        Args:
            text: Text to extract from

        Returns:
            List of potential entity names
        """
        import re

        # TODO: Find capitalized words that aren't at sentence start
        # Return list of potential entities
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message and update entity tracking.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add message
        # Extract entities from content
        # Update entities dict with message references
        pass

    def get_entity_info(self, entity: str) -> list[str]:
        """Get all messages mentioning an entity.

        Args:
            entity: Entity name

        Returns:
            List of message contents mentioning entity
        """
        # TODO: Return list of messages that mention entity
        pass

    def get_all_entities(self) -> dict:
        """Get all tracked entities.

        Returns:
            Dict mapping entities to their mentions
        """
        # TODO: Return copy of entities dict
        pass

    def get_messages(self) -> list[Message]:
        """Get all messages."""
        # TODO: Return copy of messages
        pass

    def clear(self) -> None:
        """Clear messages and entities."""
        # TODO: Clear both messages and entities
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load memory with entity context.

        Returns:
            Dict with "history" and "entities"
        """
        # TODO: Return messages and entities dict
        pass


# =============================================================================
# TASK 7: Implement Memory with Persistence
# =============================================================================
class PersistentMemory(BaseMemory):
    """
    Memory that persists to disk.

    Survives application restarts.
    """

    def __init__(self, file_path: str):
        """Initialize persistent memory.

        Args:
            file_path: Path to save/load memory
        """
        # TODO: Store file_path, initialize messages list
        # Load existing memory if file exists
        pass

    def _load(self) -> None:
        """Load memory from disk."""
        import json
        import os

        # TODO: If file exists, read and parse JSON
        # Convert dicts back to Message objects
        pass

    def _save(self) -> None:
        """Save memory to disk."""
        import json

        # TODO: Convert messages to dicts and save as JSON
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message and persist.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add message and call _save()
        pass

    def get_messages(self) -> list[Message]:
        """Get all messages.

        Returns:
            List of messages
        """
        # TODO: Return copy of messages
        pass

    def clear(self) -> None:
        """Clear memory and remove file."""
        import os

        # TODO: Clear messages, delete file if exists
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load memory as variables.

        Returns:
            Dict with "history"
        """
        # TODO: Return {"history": self.get_messages()}
        pass


# =============================================================================
# TASK 8: Implement Conversation Chain with Memory
# =============================================================================
class ConversationChain:
    """
    A chain that maintains conversation memory.

    Simulates LangChain's ConversationChain.
    """

    def __init__(
        self,
        processor,  # Function that takes prompt and returns response
        memory: BaseMemory = None,
        prompt_template: str = None,
    ):
        """Initialize conversation chain.

        Args:
            processor: Function to process prompts (simulates LLM)
            memory: Memory instance to use
            prompt_template: Template with {history} and {input} placeholders
        """
        # TODO: Store processor and memory (default to ConversationBufferMemory)
        # Store prompt template (provide default)
        pass

    def _build_prompt(self, user_input: str) -> str:
        """Build prompt with history and input.

        Args:
            user_input: Current user input

        Returns:
            Complete prompt with history
        """
        # TODO: Get history from memory
        # Format history as string if needed
        # Fill in prompt template
        pass

    def run(self, user_input: str) -> str:
        """Process user input and return response.

        Args:
            user_input: User's message

        Returns:
            AI response
        """
        # TODO: Build prompt, call processor, save context, return response
        pass

    def get_memory(self) -> BaseMemory:
        """Get the memory object.

        Returns:
            Memory instance
        """
        # TODO: Return memory
        pass


# =============================================================================
# TASK 9: Implement Memory Search
# =============================================================================
class SearchableMemory(BaseMemory):
    """
    Memory that supports semantic search over history.

    Useful for finding relevant past context.
    """

    def __init__(self, embedding_fn=None):
        """Initialize searchable memory.

        Args:
            embedding_fn: Function to create embeddings (default: bag of words)
        """
        # TODO: Store embedding_fn (or default)
        # Initialize messages list
        pass

    def _default_embedding(self, text: str) -> dict[str, int]:
        """Create simple bag-of-words embedding.

        Args:
            text: Text to embed

        Returns:
            Word frequency dict
        """
        # TODO: Return dict of word frequencies
        pass

    def _similarity(self, emb1: dict, emb2: dict) -> float:
        """Compute similarity between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        # TODO: Compute cosine-like similarity for bag-of-words
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message with embedding.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Create message, compute and store embedding
        pass

    def search(self, query: str, k: int = 3) -> list[Message]:
        """Find most relevant messages for query.

        Args:
            query: Search query
            k: Number of results

        Returns:
            Top k most relevant messages
        """
        # TODO: Compute query embedding
        # Score each message by similarity
        # Return top k messages
        pass

    def get_messages(self) -> list[Message]:
        """Get all messages.

        Returns:
            List of messages
        """
        # TODO: Return copy of messages
        pass

    def clear(self) -> None:
        """Clear messages and embeddings."""
        # TODO: Clear messages and any stored embeddings
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load memory, optionally filtered by query.

        Args:
            inputs: Optional dict with "query" for search

        Returns:
            Dict with "history" (filtered if query provided)
        """
        # TODO: If inputs has "query", search for relevant messages
        # Otherwise return all messages
        pass


# =============================================================================
# TASK 10: Implement Combined Memory
# =============================================================================
class CombinedMemory(BaseMemory):
    """
    Combines multiple memory types.

    E.g., buffer memory + entity memory.
    """

    def __init__(self, memories: list[BaseMemory]):
        """Initialize with multiple memories.

        Args:
            memories: List of memory instances
        """
        # TODO: Store the list of memories
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message to all memories.

        Args:
            role: Message role
            content: Message content
        """
        # TODO: Add message to each memory in list
        pass

    def get_messages(self) -> list[Message]:
        """Get messages from first memory.

        Returns:
            Messages from primary memory
        """
        # TODO: Return messages from first memory
        pass

    def clear(self) -> None:
        """Clear all memories."""
        # TODO: Clear each memory
        pass

    def load_memory_variables(self, inputs: dict = None) -> dict:
        """Load and combine variables from all memories.

        Returns:
            Combined dict from all memories
        """
        # TODO: Merge load_memory_variables from all memories
        # Handle key conflicts by prefixing with memory index
        pass


# =============================================================================
# Test your implementations
# =============================================================================
if __name__ == "__main__":
    # Test Task 2: Buffer Memory
    print("=== Testing Buffer Memory ===")
    buffer = ConversationBufferMemory()
    buffer.add_user_message("Hello!")
    buffer.add_ai_message("Hi there! How can I help?")
    buffer.add_user_message("Tell me about Python")
    buffer.add_ai_message("Python is a programming language...")
    print(f"Messages: {len(buffer.get_messages())}")
    print(buffer.load_memory_variables())

    # Test Task 3: Window Memory
    print("\n=== Testing Window Memory ===")
    window = ConversationBufferWindowMemory(k=2)
    for i in range(5):
        window.add_user_message(f"Question {i}")
        window.add_ai_message(f"Answer {i}")
    print(f"Window messages: {len(window.get_messages())}")  # Should be 4

    # Test Task 5: Token Memory
    print("\n=== Testing Token Memory ===")
    token_mem = ConversationTokenMemory(max_tokens=20)
    token_mem.add_user_message("This is a test message")
    token_mem.add_ai_message("This is a response message")
    token_mem.add_user_message("Another message here")
    print(f"Token count: {token_mem.get_token_count()}")

    # Test Task 8: Conversation Chain
    print("\n=== Testing Conversation Chain ===")

    def mock_llm(prompt):
        return f"I received: {prompt[-50:]}"

    chain = ConversationChain(processor=mock_llm)
    if chain:
        response1 = chain.run("Hello!")
        response2 = chain.run("What did I just say?")
        print(f"Response 2: {response2}")

    print("\n✅ Intermediate exercises completed!")

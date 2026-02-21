"""
Week 9 - Project: AI Assistant with Tools and Memory
====================================================

Build a complete AI assistant using LangChain components.

Features:
- Conversation memory (buffer + summary)
- Multiple tools (calculator, search, RAG)
- Streaming responses
- Error handling with fallbacks
- Conversation management

This project combines all concepts from Week 9.
"""

from typing import Any, Callable, Generator, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json


# =============================================================================
# PART 1: Message Types and Memory
# =============================================================================
@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: Optional[str] = None  # Tool name for tool messages
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        # TODO: Return dict with role, content, name, timestamp as ISO string
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """Create from dictionary."""
        # TODO: Parse dict back to ChatMessage
        pass


class SmartMemory:
    """
    Intelligent memory that combines buffer and summary.

    Keeps recent messages in full, summarizes older ones.
    """

    def __init__(
        self,
        recent_window: int = 10,
        summarizer: Callable[[str], str] = None,
        max_summary_length: int = 500,
    ):
        """Initialize smart memory.

        Args:
            recent_window: Number of recent messages to keep in full
            summarizer: Function to summarize text
            max_summary_length: Maximum summary length
        """
        # TODO: Store parameters
        # Initialize messages list, summary string, metadata dict
        pass

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to memory.

        Args:
            message: Message to add
        """
        # TODO: Add message to list
        # If messages exceed window * 2, trigger summarization
        pass

    def _summarize_old_messages(self) -> None:
        """Summarize and remove old messages."""
        # TODO: Take older messages (beyond window)
        # Summarize them and prepend to existing summary
        # Keep only recent window messages
        pass

    def get_context(self) -> dict:
        """Get memory context for prompts.

        Returns:
            Dict with 'summary', 'recent_messages', 'metadata'
        """
        # TODO: Return formatted context
        pass

    def get_formatted_history(self) -> str:
        """Get formatted conversation history.

        Returns:
            Formatted string with summary + recent messages
        """
        # TODO: Format as readable string
        pass

    def search_history(self, query: str) -> list[ChatMessage]:
        """Search message history.

        Args:
            query: Search query

        Returns:
            Relevant messages
        """
        # TODO: Simple substring search in message content
        pass

    def save_metadata(self, key: str, value: Any) -> None:
        """Save metadata (e.g., user preferences).

        Args:
            key: Metadata key
            value: Metadata value
        """
        # TODO: Store in metadata dict
        pass

    def clear(self) -> None:
        """Clear all memory."""
        # TODO: Clear messages, summary, and metadata
        pass


# =============================================================================
# PART 2: Tool System
# =============================================================================
@dataclass
class ToolParameter:
    """Describes a tool parameter."""

    name: str
    description: str
    type: str = "string"
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """Schema for a tool."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_function_schema(self) -> dict:
        """Convert to OpenAI function schema format.

        Returns:
            Dict in function calling format
        """
        # TODO: Return dict with name, description, parameters in JSON schema format
        pass


class Tool(ABC):
    """Base class for assistant tools."""

    name: str
    description: str
    parameters: list[ToolParameter] = []

    def get_schema(self) -> ToolSchema:
        """Get tool schema.

        Returns:
            ToolSchema for this tool
        """
        # TODO: Return ToolSchema with name, description, parameters
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool output
        """
        pass

    def __call__(self, **kwargs) -> str:
        """Make tool callable."""
        return self.execute(**kwargs)


class CalculatorTool(Tool):
    """Calculator for math operations."""

    name = "calculator"
    description = "Evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, tan, log."
    parameters = [
        ToolParameter(
            name="expression",
            description="Mathematical expression to evaluate",
            required=True,
        )
    ]

    def execute(self, expression: str) -> str:
        """Evaluate math expression safely.

        Args:
            expression: Math expression

        Returns:
            Result or error message
        """
        import math

        # TODO: Safely evaluate expression with math functions
        # Return result as string
        # Handle errors gracefully
        pass


class WebSearchTool(Tool):
    """Web search simulation."""

    name = "web_search"
    description = "Search the web for information."
    parameters = [
        ToolParameter(name="query", description="Search query", required=True),
        ToolParameter(
            name="num_results",
            description="Number of results",
            type="integer",
            required=False,
            default=3,
        ),
    ]

    def __init__(self, knowledge_base: dict[str, str] = None):
        """Initialize with knowledge base."""
        # TODO: Store knowledge base (or use defaults)
        pass

    def execute(self, query: str, num_results: int = 3) -> str:
        """Search knowledge base.

        Args:
            query: Search query
            num_results: Max results

        Returns:
            Search results
        """
        # TODO: Find matching entries, return formatted results
        pass


class DocumentRetrieverTool(Tool):
    """Retrieve relevant documents (RAG component)."""

    name = "retrieve_documents"
    description = "Retrieve relevant documents based on a query."
    parameters = [
        ToolParameter(
            name="query", description="Query to find relevant documents", required=True
        ),
        ToolParameter(
            name="top_k",
            description="Number of documents to retrieve",
            type="integer",
            required=False,
            default=3,
        ),
    ]

    def __init__(self, documents: list[dict] = None, embedder=None):
        """Initialize with documents.

        Args:
            documents: List of {"content": str, "metadata": dict}
            embedder: Function to create embeddings
        """
        # TODO: Store documents and embedder
        # Create simple index for retrieval
        pass

    def _simple_embed(self, text: str) -> dict:
        """Simple bag-of-words embedding.

        Args:
            text: Text to embed

        Returns:
            Word frequency dict
        """
        # TODO: Return word frequency dictionary
        pass

    def _similarity(self, query_emb: dict, doc_emb: dict) -> float:
        """Compute similarity between embeddings.

        Args:
            query_emb: Query embedding
            doc_emb: Document embedding

        Returns:
            Similarity score
        """
        # TODO: Compute cosine-like similarity
        pass

    def execute(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Formatted document excerpts
        """
        # TODO: Embed query, score documents, return top_k
        pass


class ToolManager:
    """Manages available tools."""

    def __init__(self):
        """Initialize tool manager."""
        # TODO: Initialize tools dict
        pass

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool to register
        """
        # TODO: Add to tools dict
        pass

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None
        """
        # TODO: Look up tool
        pass

    def list_tools(self) -> list[str]:
        """List available tools.

        Returns:
            List of tool names
        """
        # TODO: Return tool names
        pass

    def get_all_schemas(self) -> list[ToolSchema]:
        """Get schemas for all tools.

        Returns:
            List of ToolSchema objects
        """
        # TODO: Return schemas for all tools
        pass

    def execute_tool(self, name: str, **kwargs) -> str:
        """Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool output or error message
        """
        # TODO: Get tool and execute, handle errors
        pass


# =============================================================================
# PART 3: Response Generation
# =============================================================================
class ResponseGenerator:
    """Generates responses using LLM."""

    def __init__(self, llm: Callable[[str], str] = None, system_prompt: str = None):
        """Initialize generator.

        Args:
            llm: LLM function
            system_prompt: Default system prompt
        """
        # TODO: Store LLM and system prompt
        pass

    def _default_llm(self, prompt: str) -> str:
        """Default mock LLM.

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        # TODO: Return simple mock response based on prompt content
        pass

    def generate(
        self,
        user_input: str,
        context: str = "",
        tool_results: list[tuple[str, str]] = None,
    ) -> str:
        """Generate a response.

        Args:
            user_input: User's message
            context: Conversation context
            tool_results: List of (tool_name, result) tuples

        Returns:
            Generated response
        """
        # TODO: Build prompt and generate response
        pass

    def stream(self, user_input: str, context: str = "") -> Generator[str, None, None]:
        """Stream a response word by word.

        Args:
            user_input: User's message
            context: Context

        Yields:
            Response words/chunks
        """
        # TODO: Generate response and yield word by word
        pass


# =============================================================================
# PART 4: Tool Decision Engine
# =============================================================================
@dataclass
class ToolDecision:
    """Decision about which tool to use."""

    use_tool: bool
    tool_name: Optional[str] = None
    tool_args: dict = field(default_factory=dict)
    reasoning: str = ""


class ToolDecisionEngine:
    """Decides when and which tool to use."""

    def __init__(
        self, tool_manager: ToolManager, decision_llm: Callable[[str], str] = None
    ):
        """Initialize decision engine.

        Args:
            tool_manager: Available tools
            decision_llm: LLM for decisions
        """
        # TODO: Store tool manager and LLM
        pass

    def _build_decision_prompt(self, user_input: str) -> str:
        """Build prompt for tool decision.

        Args:
            user_input: User's query

        Returns:
            Decision prompt
        """
        # TODO: Build prompt that asks whether to use tools
        # Include available tool descriptions
        pass

    def decide(self, user_input: str, context: str = "") -> ToolDecision:
        """Decide whether to use a tool.

        Args:
            user_input: User's message
            context: Conversation context

        Returns:
            ToolDecision with tool info or no-tool decision
        """
        # TODO: Analyze input and decide
        # Use simple heuristics or LLM
        pass

    def _parse_decision(self, llm_output: str) -> ToolDecision:
        """Parse LLM output into decision.

        Args:
            llm_output: LLM response

        Returns:
            Parsed ToolDecision
        """
        # TODO: Parse response for tool choice
        pass


# =============================================================================
# PART 5: Error Handling
# =============================================================================
class FallbackHandler:
    """Handles errors with fallback strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        fallback_response: str = "I apologize, but I'm having trouble processing that request.",
    ):
        """Initialize fallback handler.

        Args:
            max_retries: Max retry attempts
            fallback_response: Default fallback message
        """
        # TODO: Store parameters
        pass

    def with_retry(self, func: Callable, *args, **kwargs) -> tuple[Any, bool]:
        """Execute with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, success)
        """
        # TODO: Try func up to max_retries times
        # Return result and success flag
        pass

    def with_fallback(
        self, primary: Callable, fallbacks: list[Callable], *args, **kwargs
    ) -> Any:
        """Execute with fallback chain.

        Args:
            primary: Primary function
            fallbacks: Fallback functions
            *args: Arguments
            **kwargs: Keyword arguments

        Returns:
            Result from first successful function
        """
        # TODO: Try primary, then fallbacks in order
        pass


# =============================================================================
# PART 6: Conversation Manager
# =============================================================================
class ConversationManager:
    """Manages conversation sessions."""

    def __init__(self):
        """Initialize conversation manager."""
        # TODO: Initialize sessions dict
        pass

    def create_session(self, session_id: str = None) -> str:
        """Create a new conversation session.

        Args:
            session_id: Optional session ID

        Returns:
            Session ID
        """
        import uuid

        # TODO: Create session with unique ID, initialize memory
        pass

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data.

        Args:
            session_id: Session ID

        Returns:
            Session data or None
        """
        # TODO: Return session from sessions dict
        pass

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add message to session.

        Args:
            session_id: Session ID
            role: Message role
            content: Message content

        Returns:
            True if successful
        """
        # TODO: Add message to session memory
        pass

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Get session history.

        Args:
            session_id: Session ID

        Returns:
            List of messages
        """
        # TODO: Return messages from session memory
        pass

    def end_session(self, session_id: str) -> bool:
        """End and clean up a session.

        Args:
            session_id: Session ID

        Returns:
            True if session was ended
        """
        # TODO: Remove session from sessions dict
        pass

    def export_session(self, session_id: str) -> Optional[str]:
        """Export session as JSON.

        Args:
            session_id: Session ID

        Returns:
            JSON string or None
        """
        # TODO: Convert session to JSON
        pass


# =============================================================================
# PART 7: Complete AI Assistant
# =============================================================================
class AIAssistant:
    """
    Complete AI assistant combining all components.

    Features:
    - Multi-turn conversation memory
    - Tool use (calculator, search, RAG)
    - Streaming responses
    - Error handling
    - Session management
    """

    def __init__(
        self,
        llm: Callable[[str], str] = None,
        system_prompt: str = None,
        verbose: bool = False,
    ):
        """Initialize AI assistant.

        Args:
            llm: LLM function
            system_prompt: System prompt
            verbose: Enable debug output
        """
        # TODO: Initialize all components:
        # - Memory (SmartMemory)
        # - Tool Manager with default tools
        # - Response Generator
        # - Tool Decision Engine
        # - Fallback Handler
        # - Conversation Manager
        pass

    def _setup_tools(self) -> None:
        """Set up default tools."""
        # TODO: Register calculator, search, and retriever tools
        pass

    def chat(
        self, message: str, session_id: str = None, stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Process a chat message.

        Args:
            message: User message
            session_id: Optional session ID
            stream: Whether to stream response

        Returns:
            Response string or generator
        """
        # TODO: Full chat pipeline:
        # 1. Get or create session
        # 2. Add user message to memory
        # 3. Get conversation context
        # 4. Decide if tool needed
        # 5. Execute tools if needed
        # 6. Generate response
        # 7. Add response to memory
        # 8. Return response (or stream)
        pass

    def _execute_tool_if_needed(
        self, message: str, context: str
    ) -> tuple[Optional[str], list[tuple[str, str]]]:
        """Execute tool if decision engine says so.

        Args:
            message: User message
            context: Conversation context

        Returns:
            Tuple of (tool_result, [(tool_name, result)])
        """
        # TODO: Use decision engine, execute tool if needed
        pass

    def add_document(self, content: str, metadata: dict = None) -> None:
        """Add a document to RAG retriever.

        Args:
            content: Document content
            metadata: Document metadata
        """
        # TODO: Add document to retriever tool
        pass

    def get_available_tools(self) -> list[str]:
        """Get list of available tools.

        Returns:
            Tool names
        """
        # TODO: Return tool names from manager
        pass

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        # TODO: Clear memory
        pass

    def export_conversation(self) -> str:
        """Export conversation as JSON.

        Returns:
            JSON string
        """
        # TODO: Export memory as JSON
        pass

    def interactive_chat(self) -> None:
        """Run interactive chat loop.

        Useful for testing.
        """
        print("AI Assistant ready. Type 'quit' to exit, 'clear' to reset.")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.clear_memory()
                    print("Memory cleared.")
                    continue

                response = self.chat(user_input)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


# =============================================================================
# Test the complete assistant
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 9 Project: AI Assistant with Tools and Memory")
    print("=" * 60)

    # Mock LLM for testing
    def mock_llm(prompt: str) -> str:
        """Simple mock LLM."""
        prompt_lower = prompt.lower()

        if "calculate" in prompt_lower or "math" in prompt_lower:
            if "2 + 2" in prompt:
                return "The result of 2 + 2 is 4."
            return "I can help with calculations. What would you like to compute?"

        if "search" in prompt_lower:
            return "I found some relevant information about your query."

        if "document" in prompt_lower:
            return "Based on the documents, here's what I found..."

        return f"I understand you're asking about: {prompt[:100]}..."

    # Create assistant
    assistant = AIAssistant(llm=mock_llm, verbose=True)

    print("\n--- Testing Chat ---")
    if assistant:
        # Test basic chat
        response = assistant.chat("Hello! How are you?")
        print(f"Response: {response}")

        # Test calculator
        response = assistant.chat("What is 2 + 2?")
        print(f"Response: {response}")

        # Test memory
        response = assistant.chat("What did I just ask you?")
        print(f"Response: {response}")

        # Show available tools
        print(f"\nAvailable tools: {assistant.get_available_tools()}")

    print("\n--- Running Interactive Mode ---")
    print("(Type a message to test, or 'quit' to exit)")

    # Uncomment to run interactive mode:
    # assistant.interactive_chat()

    print("\n✅ Project implementation complete!")

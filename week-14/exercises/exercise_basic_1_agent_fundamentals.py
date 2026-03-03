"""
Week 14 - Exercise 1: Agent Fundamentals (Basic)

Learn the core building blocks of AI agents:
- Message and state models
- Tool definitions and execution
- Agent context and memory
- Basic agent implementation

Run tests with: pytest tests/test_exercise_basic_1_agent_fundamentals.py -v
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime


# =============================================================================
# Part 1: Agent Message Model
# =============================================================================
class MessageRole(Enum):
    """Roles for agent messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class AgentMessage:
    """
    A message in agent conversation.

    Attributes:
        role: Who sent the message (user, assistant, system, tool)
        content: The message content
        timestamp: When the message was created
        metadata: Additional message metadata

    Example:
        >>> msg = AgentMessage(role=MessageRole.USER, content="Hello!")
        >>> msg.role
        <MessageRole.USER: 'user'>
        >>> msg.is_from_user()
        True
    """

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def is_from_user(self) -> bool:
        """Check if message is from user."""
        # TODO: Return True if role is USER
        pass

    def is_from_assistant(self) -> bool:
        """Check if message is from assistant."""
        # TODO: Return True if role is ASSISTANT
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        # TODO: Return dict with role (as string), content, and timestamp (as ISO string)
        pass


# =============================================================================
# Part 2: Tool Result Model
# =============================================================================
@dataclass
class ToolResult:
    """
    Result from executing a tool.

    Attributes:
        tool_name: Name of the tool that was executed
        success: Whether execution succeeded
        result: The result value (if successful)
        error: Error message (if failed)
        execution_time_ms: How long execution took

    Example:
        >>> result = ToolResult.success("calculator", 42)
        >>> result.success
        True
        >>> result.result
        42
    """

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    @classmethod
    def success(cls, tool_name: str, result: Any, time_ms: float = 0.0) -> "ToolResult":
        """Create a successful result."""
        # TODO: Create ToolResult with success=True and the given result
        pass

    @classmethod
    def failure(cls, tool_name: str, error: str, time_ms: float = 0.0) -> "ToolResult":
        """Create a failed result."""
        # TODO: Create ToolResult with success=False and the error message
        pass

    def to_message(self) -> AgentMessage:
        """Convert result to agent message."""
        # TODO: Create AgentMessage with TOOL role
        # Content should be the result if success, else the error
        pass


# =============================================================================
# Part 3: Agent Context
# =============================================================================
class AgentContext:
    """
    Manages conversation context for an agent.

    Tracks messages, provides context window management,
    and supports context summarization.

    Example:
        >>> ctx = AgentContext(max_messages=10)
        >>> ctx.add_user_message("Hello")
        >>> ctx.add_assistant_message("Hi there!")
        >>> len(ctx.messages)
        2
    """

    def __init__(self, max_messages: int = 50):
        """Initialize context with message limit."""
        # TODO: Initialize messages list and max_messages
        pass

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to context."""
        # TODO: Add message and trim if exceeds max_messages
        pass

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        # TODO: Create and add USER message
        pass

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        # TODO: Create and add ASSISTANT message
        pass

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        # TODO: Create and add SYSTEM message
        pass

    def get_recent(self, n: int) -> list[AgentMessage]:
        """Get the n most recent messages."""
        # TODO: Return last n messages
        pass

    def to_prompt_format(self) -> list[dict]:
        """Convert context to LLM prompt format."""
        # TODO: Return list of dicts with role and content keys
        pass

    def clear(self) -> None:
        """Clear all messages."""
        # TODO: Empty the messages list
        pass


# =============================================================================
# Part 4: Tool Definition
# =============================================================================
@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """
    Definition of an agent tool.

    Wraps a callable function with metadata for the LLM.

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> tool = Tool(
        ...     name="add",
        ...     description="Add two numbers",
        ...     function=add,
        ...     parameters=[
        ...         ToolParameter("a", "int", "First number"),
        ...         ToolParameter("b", "int", "Second number")
        ...     ]
        ... )
        >>> tool.execute({"a": 1, "b": 2})
        3
    """

    name: str
    description: str
    function: Callable
    parameters: list[ToolParameter] = field(default_factory=list)

    def execute(self, args: dict) -> Any:
        """Execute the tool with given arguments."""
        # TODO: Call self.function with **args and return result
        pass

    def to_schema(self) -> dict:
        """Convert to OpenAI function schema format."""
        # TODO: Return dict with name, description, and parameters schema
        # Parameters should be in JSON Schema format
        pass

    def validate_args(self, args: dict) -> tuple[bool, Optional[str]]:
        """Validate arguments against parameter definitions."""
        # TODO: Check required parameters are present
        # Return (True, None) if valid, (False, error_message) if not
        pass


# =============================================================================
# Part 5: Tool Registry
# =============================================================================
class ToolRegistry:
    """
    Registry for managing agent tools.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register
        ... def greet(name: str) -> str:
        ...     '''Greet someone.'''
        ...     return f"Hello, {name}!"
        >>> registry.get("greet").execute({"name": "World"})
        'Hello, World!'
    """

    def __init__(self):
        """Initialize empty registry."""
        # TODO: Initialize tools dict
        pass

    def register(self, func: Callable) -> Callable:
        """Decorator to register a function as a tool."""
        # TODO: Create Tool from function and add to registry
        # Use function name, docstring, and type hints
        # Return the original function unchanged
        pass

    def add(self, tool: Tool) -> None:
        """Add a tool directly."""
        # TODO: Add tool to registry by name
        pass

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        # TODO: Return tool or None
        pass

    def list_tools(self) -> list[str]:
        """List all tool names."""
        # TODO: Return list of registered tool names
        pass

    def to_schemas(self) -> list[dict]:
        """Get schemas for all tools."""
        # TODO: Return list of tool schemas
        pass


# =============================================================================
# Part 6: Agent State
# =============================================================================
class AgentStateType(Enum):
    """Possible agent states."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


class AgentState:
    """
    State machine for agent execution.

    Example:
        >>> state = AgentState()
        >>> state.current
        <AgentStateType.IDLE: 'idle'>
        >>> state.transition_to(AgentStateType.THINKING)
        True
        >>> state.current
        <AgentStateType.THINKING: 'thinking'>
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        AgentStateType.IDLE: [AgentStateType.THINKING, AgentStateType.ERROR],
        AgentStateType.THINKING: [
            AgentStateType.EXECUTING,
            AgentStateType.DONE,
            AgentStateType.ERROR,
        ],
        AgentStateType.EXECUTING: [
            AgentStateType.THINKING,
            AgentStateType.WAITING,
            AgentStateType.ERROR,
        ],
        AgentStateType.WAITING: [AgentStateType.THINKING, AgentStateType.ERROR],
        AgentStateType.DONE: [AgentStateType.IDLE],
        AgentStateType.ERROR: [AgentStateType.IDLE],
    }

    def __init__(self):
        """Initialize in IDLE state."""
        # TODO: Set current state to IDLE and initialize history
        pass

    @property
    def current(self) -> AgentStateType:
        """Get current state."""
        # TODO: Return current state
        pass

    def transition_to(self, new_state: AgentStateType) -> bool:
        """Transition to a new state."""
        # TODO: Check if transition is valid, update state if so
        # Add transition to history
        # Return True if successful, False if invalid
        pass

    def can_transition(self, new_state: AgentStateType) -> bool:
        """Check if transition is valid."""
        # TODO: Return whether transition from current to new_state is valid
        pass

    def reset(self) -> None:
        """Reset to IDLE state."""
        # TODO: Set state to IDLE and clear history
        pass

    def get_history(self) -> list[tuple[AgentStateType, datetime]]:
        """Get state transition history."""
        # TODO: Return history of state transitions
        pass


# =============================================================================
# Part 7: Simple Agent
# =============================================================================
class SimpleAgent:
    """
    A basic agent that can use tools to respond to queries.

    Example:
        >>> agent = SimpleAgent(name="assistant")
        >>> @agent.tool
        ... def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>> response = agent.run("Search for Python tutorials")
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 10,
    ):
        """Initialize agent."""
        # TODO: Initialize name, system_prompt, max_iterations
        # Create ToolRegistry, AgentContext, AgentState
        pass

    def tool(self, func: Callable) -> Callable:
        """Decorator to add a tool."""
        # TODO: Register function with tool registry
        pass

    def think(self, user_input: str) -> Optional[dict]:
        """
        Determine what action to take.

        Returns None if no tool call needed, or dict with:
        - tool: name of tool to call
        - args: arguments for tool
        """
        # TODO: Simple keyword-based tool selection
        # Check if any tool name appears in user_input
        # This is a mock - real implementation would use LLM
        pass

    def execute_tool(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool and return result."""
        # TODO: Get tool from registry and execute
        # Handle errors and return ToolResult
        pass

    def run(self, user_input: str) -> str:
        """Run agent on user input."""
        # TODO: Implement basic agent loop:
        # 1. Add user message to context
        # 2. Think about what to do
        # 3. If tool call needed, execute and add result
        # 4. Generate response (mock: return tool result or echo)
        # 5. Add assistant message and return
        pass


# =============================================================================
# Part 8: Agent Memory
# =============================================================================
@dataclass
class MemoryItem:
    """An item stored in agent memory."""

    content: str
    timestamp: datetime
    importance: float = 1.0
    metadata: dict = field(default_factory=dict)


class AgentMemory:
    """
    Memory system for agents with importance-based retrieval.

    Example:
        >>> memory = AgentMemory(capacity=100)
        >>> memory.store("User prefers Python", importance=0.8)
        >>> memory.store("User is learning ML", importance=0.9)
        >>> results = memory.retrieve("programming", k=1)
        >>> len(results)
        1
    """

    def __init__(self, capacity: int = 1000):
        """Initialize memory with capacity limit."""
        # TODO: Initialize items list and capacity
        pass

    def store(
        self, content: str, importance: float = 1.0, metadata: Optional[dict] = None
    ) -> None:
        """Store a memory item."""
        # TODO: Create MemoryItem and add to items
        # If over capacity, remove least important item
        pass

    def retrieve(self, query: str, k: int = 5) -> list[MemoryItem]:
        """Retrieve relevant memories."""
        # TODO: Simple keyword matching retrieval
        # Return top k items that contain query words
        pass

    def get_recent(self, n: int) -> list[MemoryItem]:
        """Get n most recent memories."""
        # TODO: Return last n items by timestamp
        pass

    def get_important(self, n: int, min_importance: float = 0.5) -> list[MemoryItem]:
        """Get most important memories."""
        # TODO: Filter by importance and return top n
        pass

    def clear(self) -> None:
        """Clear all memories."""
        # TODO: Empty items list
        pass

    def count(self) -> int:
        """Get memory count."""
        # TODO: Return number of items
        pass


# =============================================================================
# Part 9: Agent Logger
# =============================================================================
class LogLevel(Enum):
    """Log levels for agent actions."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LogEntry:
    """A log entry for agent actions."""

    level: LogLevel
    message: str
    timestamp: datetime
    agent_name: str
    metadata: dict = field(default_factory=dict)


class AgentLogger:
    """
    Logger for tracking agent actions and decisions.

    Example:
        >>> logger = AgentLogger("my-agent")
        >>> logger.info("Starting task")
        >>> logger.tool_call("search", {"query": "test"})
        >>> logger.error("Tool failed", error="Connection timeout")
    """

    def __init__(self, agent_name: str, min_level: LogLevel = LogLevel.INFO):
        """Initialize logger."""
        # TODO: Initialize agent_name, min_level, and entries list
        pass

    def log(self, level: LogLevel, message: str, **metadata) -> None:
        """Log a message."""
        # TODO: Create LogEntry and add to entries if >= min_level
        pass

    def debug(self, message: str, **metadata) -> None:
        """Log debug message."""
        # TODO: Call log with DEBUG level
        pass

    def info(self, message: str, **metadata) -> None:
        """Log info message."""
        # TODO: Call log with INFO level
        pass

    def warning(self, message: str, **metadata) -> None:
        """Log warning message."""
        # TODO: Call log with WARNING level
        pass

    def error(self, message: str, **metadata) -> None:
        """Log error message."""
        # TODO: Call log with ERROR level
        pass

    def tool_call(self, tool_name: str, args: dict) -> None:
        """Log a tool call."""
        # TODO: Log INFO with tool call details
        pass

    def tool_result(self, tool_name: str, result: ToolResult) -> None:
        """Log a tool result."""
        # TODO: Log INFO if success, WARNING if failure
        pass

    def get_entries(self, level: Optional[LogLevel] = None) -> list[LogEntry]:
        """Get log entries, optionally filtered by level."""
        # TODO: Return entries, filtered if level specified
        pass


# =============================================================================
# Part 10: Agent Executor
# =============================================================================
class AgentExecutor:
    """
    Executes agents with retries, timeouts, and error handling.

    Example:
        >>> executor = AgentExecutor(max_retries=3, timeout_seconds=30)
        >>> result = executor.run(agent, "What is 2+2?")
    """

    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
        retry_delay_seconds: float = 1.0,
    ):
        """Initialize executor."""
        # TODO: Initialize retry/timeout settings
        pass

    def run(
        self,
        agent: SimpleAgent,
        user_input: str,
        context: Optional[AgentContext] = None,
    ) -> str:
        """Run agent with retries."""
        # TODO: Execute agent.run with retry logic
        # On failure, wait retry_delay and try again
        # Return result or raise after max_retries
        pass

    def run_with_callback(
        self,
        agent: SimpleAgent,
        user_input: str,
        on_thinking: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ) -> str:
        """Run agent with callbacks for each step."""
        # TODO: Run agent and call callbacks at each stage:
        # - on_thinking: when agent starts thinking
        # - on_tool_call: when tool is called (tool_name, args)
        # - on_complete: when done (result)
        pass

    def run_batch(self, agent: SimpleAgent, inputs: list[str]) -> list[str]:
        """Run agent on multiple inputs."""
        # TODO: Run agent on each input and collect results
        pass


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create a simple agent
    agent = SimpleAgent(
        name="calculator-agent", system_prompt="You are a helpful calculator assistant."
    )

    # Add calculator tools
    @agent.tool
    def add(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @agent.tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    # List available tools
    print("Available tools:", agent._tools.list_tools())

    # Run agent
    response = agent.run("Can you add 5 and 3?")
    print(f"Response: {response}")

    # Check agent state
    print(f"Agent state: {agent._state.current}")

    # View context
    print(f"Messages in context: {len(agent._context.messages)}")

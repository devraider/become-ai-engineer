"""
Solution for Week 14 - Exercise 1: Agent Fundamentals

Complete implementations for core agent building blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import asyncio
import json
import logging
import time
import uuid


# =============================================================================
# Part 1: Agent Message - SOLUTION
# =============================================================================
class MessageRole(Enum):
    """Roles for messages in agent conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class AgentMessage:
    """
    Represents a message in an agent conversation.

    Solution implements:
    - Message role tracking
    - Metadata support
    - Serialization
    - Helper methods
    """

    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_from_user(self) -> bool:
        """Check if message is from user."""
        return self.role == MessageRole.USER

    def is_from_assistant(self) -> bool:
        """Check if message is from assistant."""
        return self.role == MessageRole.ASSISTANT

    def is_system(self) -> bool:
        """Check if message is a system message."""
        return self.role == MessageRole.SYSTEM

    def is_tool_result(self) -> bool:
        """Check if message is a tool result."""
        return self.role == MessageRole.TOOL

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now()
            ),
        )

    @classmethod
    def user(cls, content: str, **metadata) -> "AgentMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, metadata=metadata)

    @classmethod
    def assistant(cls, content: str, **metadata) -> "AgentMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, metadata=metadata)

    @classmethod
    def system(cls, content: str, **metadata) -> "AgentMessage":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, metadata=metadata)

    @classmethod
    def tool(cls, content: str, tool_name: str, **metadata) -> "AgentMessage":
        """Create a tool result message."""
        metadata["tool_name"] = tool_name
        return cls(role=MessageRole.TOOL, content=content, metadata=metadata)


# =============================================================================
# Part 2: Tool Result - SOLUTION
# =============================================================================
@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.

    Solution implements:
    - Success/failure tracking
    - Error handling
    - Execution metadata
    """

    success: bool
    output: Any
    error: Optional[str] = None
    tool_name: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls, output: Any, tool_name: str = "", execution_time: float = 0.0
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            success=True,
            output=output,
            tool_name=tool_name,
            execution_time=execution_time,
        )

    @classmethod
    def failure_result(
        cls, error: str, tool_name: str = "", partial_output: Any = None
    ) -> "ToolResult":
        """Create a failed tool result."""
        return cls(
            success=False, output=partial_output, error=error, tool_name=tool_name
        )

    def to_message(self) -> AgentMessage:
        """Convert result to a tool message."""
        if self.success:
            content = (
                json.dumps(self.output)
                if not isinstance(self.output, str)
                else self.output
            )
        else:
            content = f"Error: {self.error}"

        return AgentMessage.tool(
            content=content,
            tool_name=self.tool_name,
            success=self.success,
            execution_time=self.execution_time,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


# =============================================================================
# Part 3: Agent Context - SOLUTION
# =============================================================================
@dataclass
class AgentContext:
    """
    Maintains the context for an agent conversation.

    Solution implements:
    - Message history management
    - Context window limits
    - Variable storage
    - System prompt handling
    """

    messages: List[AgentMessage] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    max_messages: int = 100
    token_limit: int = 4096
    _created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the context."""
        self.messages.append(message)
        self._enforce_limits()

    def add_user_message(self, content: str) -> AgentMessage:
        """Add a user message."""
        msg = AgentMessage.user(content)
        self.add_message(msg)
        return msg

    def add_assistant_message(self, content: str) -> AgentMessage:
        """Add an assistant message."""
        msg = AgentMessage.assistant(content)
        self.add_message(msg)
        return msg

    def _enforce_limits(self) -> None:
        """Enforce message limits."""
        # Keep system messages + recent messages within limit
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in self.messages if m.role != MessageRole.SYSTEM]

        # Keep most recent messages
        if len(other_msgs) > self.max_messages:
            other_msgs = other_msgs[-self.max_messages :]

        self.messages = system_msgs + other_msgs

    def get_recent_messages(self, count: int = 10) -> List[AgentMessage]:
        """Get the most recent messages."""
        return self.messages[-count:]

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API calls."""
        result = []

        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})

        for msg in self.messages:
            result.append({"role": msg.role.value, "content": msg.content})

        return result

    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)

    def clear_messages(self) -> None:
        """Clear all messages except system messages."""
        self.messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "variables": self.variables,
            "system_prompt": self.system_prompt,
            "max_messages": self.max_messages,
            "created_at": self._created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContext":
        """Deserialize context from dictionary."""
        ctx = cls(
            messages=[AgentMessage.from_dict(m) for m in data.get("messages", [])],
            variables=data.get("variables", {}),
            system_prompt=data.get("system_prompt"),
            max_messages=data.get("max_messages", 100),
        )
        if "created_at" in data:
            ctx._created_at = datetime.fromisoformat(data["created_at"])
        return ctx


# =============================================================================
# Part 4: Tool Definition - SOLUTION
# =============================================================================
@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class Tool:
    """
    Defines a tool that an agent can use.

    Solution implements:
    - Parameter validation
    - Schema generation
    - Execution with timing
    """

    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for the tool."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def validate_args(self, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate arguments against parameter definitions."""
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"

            if param.name in args:
                value = args[param.name]

                # Type checking (basic)
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                elif param.type == "integer" and not isinstance(value, int):
                    return False, f"Parameter {param.name} must be an integer"
                elif param.type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be a number"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"

                # Enum checking
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of: {param.enum}"

        return True, None

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        # Validate arguments
        valid, error = self.validate_args(kwargs)
        if not valid:
            return ToolResult.failure_result(error, self.name)

        if not self.handler:
            return ToolResult.failure_result("No handler defined", self.name)

        start = time.time()
        try:
            result = self.handler(**kwargs)
            execution_time = time.time() - start
            return ToolResult.success_result(result, self.name, execution_time)
        except Exception as e:
            execution_time = time.time() - start
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time,
            )

    async def execute_async(self, **kwargs) -> ToolResult:
        """Execute the tool asynchronously."""
        valid, error = self.validate_args(kwargs)
        if not valid:
            return ToolResult.failure_result(error, self.name)

        if not self.handler:
            return ToolResult.failure_result("No handler defined", self.name)

        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)
            execution_time = time.time() - start
            return ToolResult.success_result(result, self.name, execution_time)
        except Exception as e:
            execution_time = time.time() - start
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time,
            )


# =============================================================================
# Part 5: Tool Registry - SOLUTION
# =============================================================================
class ToolRegistry:
    """
    Registry for managing available tools.

    Solution implements:
    - Tool registration
    - Lookup by name
    - Schema generation
    - Decorator support
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    def tool(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> Callable:
        """Decorator to register a function as a tool."""

        def decorator(func: Callable) -> Callable:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters or [],
                handler=func,
            )
            self.register(tool)
            return func

        return decorator

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult.failure_result(f"Tool not found: {name}", name)
        return tool.execute(**kwargs)

    async def execute_async(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool asynchronously by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult.failure_result(f"Tool not found: {name}", name)
        return await tool.execute_async(**kwargs)


# =============================================================================
# Part 6: Agent State - SOLUTION
# =============================================================================
class AgentStateType(Enum):
    """States an agent can be in."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentState:
    """
    Tracks the current state of an agent.

    Solution implements:
    - State transitions
    - Transition validation
    - State history
    """

    current: AgentStateType = AgentStateType.IDLE
    previous: Optional[AgentStateType] = None
    history: List[tuple[AgentStateType, datetime]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _transition_count: int = 0

    # Valid state transitions
    VALID_TRANSITIONS = {
        AgentStateType.IDLE: [AgentStateType.THINKING, AgentStateType.ERROR],
        AgentStateType.THINKING: [
            AgentStateType.EXECUTING,
            AgentStateType.WAITING,
            AgentStateType.ERROR,
            AgentStateType.COMPLETED,
        ],
        AgentStateType.EXECUTING: [
            AgentStateType.THINKING,
            AgentStateType.WAITING,
            AgentStateType.ERROR,
            AgentStateType.COMPLETED,
        ],
        AgentStateType.WAITING: [
            AgentStateType.THINKING,
            AgentStateType.EXECUTING,
            AgentStateType.ERROR,
        ],
        AgentStateType.ERROR: [AgentStateType.IDLE, AgentStateType.THINKING],
        AgentStateType.COMPLETED: [AgentStateType.IDLE],
    }

    def can_transition_to(self, new_state: AgentStateType) -> bool:
        """Check if transition to new state is valid."""
        return new_state in self.VALID_TRANSITIONS.get(self.current, [])

    def transition(self, new_state: AgentStateType, **metadata) -> bool:
        """Transition to a new state."""
        if not self.can_transition_to(new_state):
            return False

        self.previous = self.current
        self.history.append((self.current, datetime.now()))
        self.current = new_state
        self.metadata = metadata
        self._transition_count += 1

        return True

    def force_transition(self, new_state: AgentStateType, **metadata) -> None:
        """Force transition regardless of validity."""
        self.previous = self.current
        self.history.append((self.current, datetime.now()))
        self.current = new_state
        self.metadata = metadata
        self._transition_count += 1

    def reset(self) -> None:
        """Reset to idle state."""
        self.previous = self.current
        self.current = AgentStateType.IDLE
        self.metadata = {}

    def is_active(self) -> bool:
        """Check if agent is in an active state."""
        return self.current in [AgentStateType.THINKING, AgentStateType.EXECUTING]

    def is_terminal(self) -> bool:
        """Check if agent is in a terminal state."""
        return self.current in [AgentStateType.COMPLETED, AgentStateType.ERROR]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get state transition history."""
        return [
            {"state": state.value, "timestamp": ts.isoformat()}
            for state, ts in self.history
        ]


# =============================================================================
# Part 7: Simple Agent - SOLUTION
# =============================================================================
class SimpleAgent:
    """
    A simple agent with tool execution capability.

    Solution implements:
    - ReAct-style reasoning
    - Tool execution
    - Context management
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        max_iterations: int = 10,
    ):
        self.name = name
        self.context = AgentContext(system_prompt=system_prompt)
        self.state = AgentState()
        self.registry = ToolRegistry()
        self.max_iterations = max_iterations
        self._iteration = 0

    def tool(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        """Decorator to register a tool."""
        return self.registry.tool(name, description, parameters)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.registry.register(tool)

    async def think(self, user_input: str) -> str:
        """Process user input and generate response."""
        self.state.transition(AgentStateType.THINKING)
        self.context.add_user_message(user_input)
        self._iteration = 0

        while self._iteration < self.max_iterations:
            self._iteration += 1

            # Get tool schemas
            tools = self.registry.get_all_schemas()

            # In a real implementation, call LLM here
            # For now, simulate processing
            response = await self._process_step(tools)

            if "tool_call" in response:
                # Execute tool
                tool_name = response["tool_call"]["name"]
                tool_args = response["tool_call"]["arguments"]

                self.state.transition(AgentStateType.EXECUTING)
                result = await self.registry.execute_async(tool_name, **tool_args)

                # Add tool result to context
                self.context.add_message(result.to_message())
                self.state.transition(AgentStateType.THINKING)
            else:
                # Final response
                self.state.transition(AgentStateType.COMPLETED)
                self.context.add_assistant_message(response.get("content", ""))
                return response.get("content", "")

        self.state.transition(AgentStateType.COMPLETED)
        return "Max iterations reached"

    async def _process_step(self, tools: List[Dict]) -> Dict[str, Any]:
        """
        Process a single step.
        In production, this would call an LLM.
        """
        # Mock implementation - in reality call LLM
        messages = self.context.get_messages_for_api()
        last_message = messages[-1]["content"] if messages else ""

        # Simple mock logic
        if "search" in last_message.lower() and self.registry.get("search"):
            return {
                "tool_call": {"name": "search", "arguments": {"query": last_message}}
            }

        return {"content": f"I processed: {last_message}"}

    def run_sync(self, user_input: str) -> str:
        """Run agent synchronously."""
        return asyncio.run(self.think(user_input))

    def reset(self) -> None:
        """Reset agent state."""
        self.context.clear_messages()
        self.state.reset()
        self._iteration = 0


# =============================================================================
# Part 8: Agent Memory - SOLUTION
# =============================================================================
@dataclass
class MemoryItem:
    """A single memory item."""

    content: str
    type: str  # "fact", "observation", "action", "reflection"
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def decay(self, factor: float = 0.95) -> None:
        """Apply decay to importance."""
        self.importance *= factor


class AgentMemory:
    """
    Memory system for an agent.

    Solution implements:
    - Short-term and long-term memory
    - Importance-based retention
    - Memory retrieval
    - Consolidation
    """

    def __init__(self, max_short_term: int = 50, max_long_term: int = 1000):
        self.short_term: List[MemoryItem] = []
        self.long_term: List[MemoryItem] = []
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term

    def add(
        self, content: str, type: str = "observation", importance: float = 0.5
    ) -> MemoryItem:
        """Add a new memory."""
        item = MemoryItem(content=content, type=type, importance=importance)
        self.short_term.append(item)

        # Move to long-term if short-term is full
        if len(self.short_term) > self.max_short_term:
            self._consolidate()

        return item

    def _consolidate(self) -> None:
        """Consolidate short-term to long-term memory."""
        # Sort by importance
        self.short_term.sort(key=lambda x: x.importance, reverse=True)

        # Move important memories to long-term
        threshold = 0.3
        to_move = [m for m in self.short_term if m.importance >= threshold]
        to_discard = [m for m in self.short_term if m.importance < threshold]

        self.long_term.extend(to_move)

        # Keep only most recent low-importance items
        self.short_term = to_discard[-10:] if to_discard else []

        # Trim long-term if needed
        if len(self.long_term) > self.max_long_term:
            self.long_term.sort(key=lambda x: x.importance, reverse=True)
            self.long_term = self.long_term[: self.max_long_term]

    def recall(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Recall memories relevant to a query."""
        all_memories = self.short_term + self.long_term

        # Simple relevance scoring (in production, use embeddings)
        query_words = set(query.lower().split())

        def relevance(item: MemoryItem) -> float:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            return overlap * item.importance

        scored = [(m, relevance(m)) for m in all_memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:limit] if _ > 0]

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """Get most recent memories."""
        all_memories = self.short_term + self.long_term
        all_memories.sort(key=lambda x: x.timestamp, reverse=True)
        return all_memories[:limit]

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """Get most important memories."""
        all_memories = self.short_term + self.long_term
        all_memories.sort(key=lambda x: x.importance, reverse=True)
        return all_memories[:limit]

    def decay_all(self, factor: float = 0.95) -> None:
        """Apply decay to all memories."""
        for item in self.short_term + self.long_term:
            item.decay(factor)

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory."""
        return {
            "short_term": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.type,
                    "importance": m.importance,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.short_term
            ],
            "long_term": [
                {
                    "id": m.id,
                    "content": m.content,
                    "type": m.type,
                    "importance": m.importance,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.long_term
            ],
        }


# =============================================================================
# Part 9: Agent Logger - SOLUTION
# =============================================================================
class LogLevel(Enum):
    """Log levels for agent logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """A single log entry."""

    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = "agent"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "metadata": self.metadata,
        }

    def format(self) -> str:
        """Format log entry as string."""
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{self.level.value.upper()}] [{self.component}] {self.message}"


class AgentLogger:
    """
    Logging system for agents.

    Solution implements:
    - Multiple log levels
    - Structured logging
    - Log storage and retrieval
    - Integration with Python logging
    """

    def __init__(
        self,
        name: str = "agent",
        level: LogLevel = LogLevel.INFO,
        max_entries: int = 1000,
    ):
        self.name = name
        self.level = level
        self.max_entries = max_entries
        self._entries: List[LogEntry] = []
        self._python_logger = logging.getLogger(name)

        # Map to Python logging levels
        self._level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        level_order = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]
        return level_order.index(level) >= level_order.index(self.level)

    def _log(
        self, level: LogLevel, message: str, component: str = "agent", **metadata
    ) -> Optional[LogEntry]:
        """Internal logging method."""
        if not self._should_log(level):
            return None

        entry = LogEntry(
            level=level, message=message, component=component, metadata=metadata
        )

        self._entries.append(entry)

        # Trim if needed
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

        # Also log to Python logger
        self._python_logger.log(self._level_map[level], message, extra=metadata)

        return entry

    def debug(self, message: str, **kwargs) -> Optional[LogEntry]:
        """Log debug message."""
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> Optional[LogEntry]:
        """Log info message."""
        return self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> Optional[LogEntry]:
        """Log warning message."""
        return self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> Optional[LogEntry]:
        """Log error message."""
        return self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> Optional[LogEntry]:
        """Log critical message."""
        return self._log(LogLevel.CRITICAL, message, **kwargs)

    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Get log entries with optional filtering."""
        entries = self._entries

        if level:
            entries = [e for e in entries if e.level == level]

        if component:
            entries = [e for e in entries if e.component == component]

        return entries[-limit:]

    def clear(self) -> None:
        """Clear all log entries."""
        self._entries = []

    def to_dict(self) -> Dict[str, Any]:
        """Export logs as dictionary."""
        return {
            "name": self.name,
            "level": self.level.value,
            "entries": [e.to_dict() for e in self._entries],
        }


# =============================================================================
# Part 10: Agent Executor - SOLUTION
# =============================================================================
class AgentExecutor:
    """
    Executes agent tasks with error handling and retries.

    Solution implements:
    - Task execution with retries
    - Timeout handling
    - Error recovery
    - Execution metrics
    """

    def __init__(
        self,
        agent: SimpleAgent,
        max_retries: int = 3,
        timeout: float = 60.0,
        retry_delay: float = 1.0,
    ):
        self.agent = agent
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.logger = AgentLogger(f"{agent.name}_executor")
        self._executions: List[Dict[str, Any]] = []

    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute a task with retries and error handling."""
        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            attempts += 1

            try:
                self.logger.info(f"Executing task (attempt {attempts})", task=task)

                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        self.agent.think(task), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Task timed out after {self.timeout}s")

                execution_time = time.time() - start_time

                execution_record = {
                    "task": task,
                    "result": result,
                    "success": True,
                    "attempts": attempts,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                }

                self._executions.append(execution_record)
                self.logger.info(f"Task completed in {execution_time:.2f}s")

                return execution_record

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempts} failed: {e}")

                if attempts < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempts)
                    self.agent.reset()

        # All retries failed
        execution_time = time.time() - start_time
        execution_record = {
            "task": task,
            "result": None,
            "success": False,
            "error": last_error,
            "attempts": attempts,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
        }

        self._executions.append(execution_record)
        self.logger.error(f"Task failed after {attempts} attempts: {last_error}")

        return execution_record

    def execute_sync(self, task: str) -> Dict[str, Any]:
        """Execute task synchronously."""
        return asyncio.run(self.execute(task))

    async def execute_batch(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple tasks sequentially."""
        results = []
        for task in tasks:
            result = await self.execute(task)
            results.append(result)
        return results

    async def execute_parallel(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel (separate agents needed)."""
        # Note: This would require separate agent instances
        # For now, execute sequentially
        return await self.execute_batch(tasks)

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        if not self._executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "total_attempts": 0,
            }

        successes = sum(1 for e in self._executions if e["success"])
        total_time = sum(e["execution_time"] for e in self._executions)
        total_attempts = sum(e["attempts"] for e in self._executions)

        return {
            "total_executions": len(self._executions),
            "success_rate": successes / len(self._executions),
            "avg_execution_time": total_time / len(self._executions),
            "total_attempts": total_attempts,
            "avg_attempts": total_attempts / len(self._executions),
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self._executions[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._executions = []


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":

    async def main():
        # Create agent
        agent = SimpleAgent(
            name="assistant", system_prompt="You are a helpful assistant with tools."
        )

        # Add tools
        @agent.tool(
            name="calculator",
            description="Perform basic math calculations",
            parameters=[
                ToolParameter("expression", "string", "The math expression to evaluate")
            ],
        )
        def calculator(expression: str) -> float:
            return eval(expression)

        @agent.tool(
            name="search",
            description="Search for information",
            parameters=[ToolParameter("query", "string", "The search query")],
        )
        def search(query: str) -> str:
            return f"Search results for: {query}"

        # Create executor
        executor = AgentExecutor(agent, max_retries=2)

        # Execute task
        result = await executor.execute("What is 2 + 2?")
        print(f"Result: {result}")

        # Get metrics
        metrics = executor.get_metrics()
        print(f"Metrics: {metrics}")

    asyncio.run(main())

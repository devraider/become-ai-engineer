"""
Solution for Week 14 - Exercise 2: Google ADK Integration

Complete implementations for Google ADK integration patterns.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    AsyncIterator,
    get_origin,
    get_args,
)
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
from importlib.util import find_spec
import asyncio
import inspect
import json
import time
import uuid

# Check for optional dependencies
HAS_ADK = find_spec("google.genai") is not None

if HAS_ADK:
    from google import genai
    from google.genai import types


# =============================================================================
# Part 1: ADK Tool Wrapper - SOLUTION
# =============================================================================
@dataclass
class ToolSchema:
    """Schema definition for a tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ADKToolWrapper:
    """
    Wraps Python functions for use with Google ADK.

    Solution implements:
    - Function wrapping
    - Schema generation
    - Async support
    - Validation
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self._is_async = asyncio.iscoroutinefunction(func)
        self._schema = self._generate_schema()

    def _generate_schema(self) -> ToolSchema:
        """Generate schema from function signature."""
        sig = inspect.signature(self.func)
        parameters = {}
        required = []

        # Type mapping
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type
            if param.annotation != inspect.Parameter.empty:
                param_type = type_map.get(param.annotation, "string")
            else:
                param_type = "string"

            parameters[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            required=required,
        )

    @property
    def schema(self) -> ToolSchema:
        """Get tool schema."""
        return self._schema

    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self._is_async:
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)

    def execute_sync(self, **kwargs) -> Any:
        """Execute synchronously."""
        if self._is_async:
            return asyncio.run(self.func(**kwargs))
        return self.func(**kwargs)

    def to_adk_tool(self) -> Dict[str, Any]:
        """Convert to ADK tool format."""
        return {"type": "function", "function": self._schema.to_dict()}

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "ADKToolWrapper":
        """Create wrapper from a function."""
        return cls(func, name, description)


# =============================================================================
# Part 2: ADK Agent Config - SOLUTION
# =============================================================================
class ModelProvider(Enum):
    """Supported model providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ADKAgentConfig:
    """
    Configuration for an ADK agent.

    Solution implements:
    - Model configuration
    - Tool configuration
    - Generation parameters
    - Validation
    """

    name: str
    model: str = "gemini-2.0-flash"
    provider: ModelProvider = ModelProvider.GEMINI
    system_instruction: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: List[ADKToolWrapper] = field(default_factory=list)
    timeout: float = 60.0
    retry_count: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")

    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }

    def get_tools_config(self) -> List[Dict[str, Any]]:
        """Get tools configuration."""
        return [tool.to_adk_tool() for tool in self.tools]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "provider": self.provider.value,
            "system_instruction": self.system_instruction,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "tools": [t.name for t in self.tools],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ADKAgentConfig":
        """Create config from dictionary."""
        return cls(
            name=data["name"],
            model=data.get("model", "gemini-2.0-flash"),
            provider=ModelProvider(data.get("provider", "gemini")),
            system_instruction=data.get("system_instruction", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2048),
            timeout=data.get("timeout", 60.0),
            retry_count=data.get("retry_count", 3),
        )


# =============================================================================
# Part 3: ADK Agent - SOLUTION
# =============================================================================
class ADKAgent:
    """
    Agent implementation using Google ADK.

    Solution implements:
    - Model initialization
    - Tool execution
    - Conversation management
    - Error handling
    """

    def __init__(self, config: ADKAgentConfig):
        self.config = config
        self.id = str(uuid.uuid4())
        self._tools: Dict[str, ADKToolWrapper] = {
            tool.name: tool for tool in config.tools
        }
        self._conversation: List[Dict[str, Any]] = []
        self._client = None
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize the ADK client."""
        if self._initialized:
            return

        if HAS_ADK:
            self._client = genai.Client()
        else:
            # Mock client for testing
            self._client = MockGenAIClient()

        self._initialized = True

    async def chat(self, message: str) -> str:
        """Send a message and get a response."""
        self._initialize()

        # Add user message to conversation
        self._conversation.append({"role": "user", "content": message})

        # Prepare messages for API
        messages = self._build_messages()

        try:
            # Call model
            response = await self._call_model(messages)

            # Process response (handle tool calls)
            while self._has_tool_calls(response):
                tool_results = await self._execute_tool_calls(response)
                response = await self._call_model(messages + tool_results)

            # Extract final content
            content = self._extract_content(response)

            # Add to conversation
            self._conversation.append({"role": "assistant", "content": content})

            return content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._conversation.append({"role": "assistant", "content": error_msg})
            return error_msg

    def _build_messages(self) -> List[Dict[str, Any]]:
        """Build messages for API call."""
        messages = []

        if self.config.system_instruction:
            messages.append(
                {"role": "system", "content": self.config.system_instruction}
            )

        messages.extend(self._conversation)
        return messages

    async def _call_model(self, messages: List[Dict]) -> Any:
        """Call the model API."""
        if HAS_ADK:
            response = await self._client.aio.models.generate_content(
                model=self.config.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    tools=(
                        [
                            types.Tool(
                                function_declarations=[
                                    types.FunctionDeclaration(**t.schema.to_dict())
                                    for t in self._tools.values()
                                ]
                            )
                        ]
                        if self._tools
                        else None
                    ),
                ),
            )
            return response
        else:
            # Mock response
            return MockResponse(messages[-1]["content"])

    def _has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls."""
        if HAS_ADK:
            return hasattr(response, "candidates") and any(
                hasattr(part, "function_call")
                for part in response.candidates[0].content.parts
            )
        return getattr(response, "has_tool_call", False)

    async def _execute_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Execute tool calls from response."""
        results = []

        if HAS_ADK:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    fc = part.function_call
                    tool = self._tools.get(fc.name)
                    if tool:
                        result = await tool.execute(**dict(fc.args))
                        results.append(
                            {
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": fc.name,
                            }
                        )

        return results

    def _extract_content(self, response: Any) -> str:
        """Extract text content from response."""
        if HAS_ADK:
            return response.text
        return getattr(response, "content", str(response))

    def add_tool(self, tool: ADKToolWrapper) -> None:
        """Add a tool to the agent."""
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the agent."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation = []

    def get_conversation(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._conversation.copy()


# Mock classes for testing without ADK
class MockGenAIClient:
    """Mock GenAI client."""

    class aio:
        class models:
            @staticmethod
            async def generate_content(*args, **kwargs):
                return MockResponse("Mock response")


class MockResponse:
    """Mock API response."""

    def __init__(self, content: str):
        self.content = content
        self.text = content
        self.has_tool_call = False


# =============================================================================
# Part 4: Tool Schema Generator - SOLUTION
# =============================================================================
class ToolSchemaGenerator:
    """
    Generates JSON schemas for tools.

    Solution implements:
    - Automatic type inference
    - Docstring parsing
    - OpenAPI-style schemas
    """

    # Type mapping for JSON Schema
    TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    @classmethod
    def from_function(cls, func: Callable) -> Dict[str, Any]:
        """Generate schema from a function."""
        sig = inspect.signature(func)
        doc = cls._parse_docstring(func.__doc__ or "")

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            prop = cls._param_to_schema(name, param, doc.get("params", {}))
            properties[name] = prop

            if param.default == inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.get("description", func.__name__),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @classmethod
    def _param_to_schema(
        cls, name: str, param, param_docs: Dict[str, str]
    ) -> Dict[str, Any]:
        """Convert parameter to schema property."""
        schema = {"description": param_docs.get(name, f"Parameter {name}")}

        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation

            # Handle Optional types
            origin = get_origin(annotation)
            if origin is Union:
                args = get_args(annotation)
                # Get non-None type
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    annotation = non_none[0]

            schema["type"] = cls.TYPE_MAP.get(annotation, "string")

            # Handle List[X]
            if origin is list:
                schema["type"] = "array"
                args = get_args(annotation)
                if args:
                    schema["items"] = {"type": cls.TYPE_MAP.get(args[0], "string")}
        else:
            schema["type"] = "string"

        # Add default if exists
        if param.default != inspect.Parameter.empty:
            schema["default"] = param.default

        return schema

    @classmethod
    def _parse_docstring(cls, docstring: str) -> Dict[str, Any]:
        """Parse docstring for description and parameters."""
        if not docstring:
            return {}

        lines = docstring.strip().split("\n")
        result = {"description": "", "params": {}}

        current_section = "description"
        description_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("Args:") or line.startswith("Parameters:"):
                current_section = "params"
                continue
            elif line.startswith("Returns:"):
                current_section = "returns"
                continue

            if current_section == "description" and line:
                description_lines.append(line)
            elif current_section == "params":
                # Parse parameter docs
                if ":" in line:
                    param_name, param_desc = line.split(":", 1)
                    param_name = param_name.strip()
                    result["params"][param_name] = param_desc.strip()

        result["description"] = " ".join(description_lines)
        return result

    @classmethod
    def validate_schema(cls, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate a tool schema."""
        if "type" not in schema:
            return False, "Missing 'type' field"

        if schema["type"] != "function":
            return False, "Type must be 'function'"

        if "function" not in schema:
            return False, "Missing 'function' field"

        func = schema["function"]

        if "name" not in func:
            return False, "Missing function name"

        if "parameters" not in func:
            return False, "Missing parameters"

        return True, None


# =============================================================================
# Part 5: ADK Memory - SOLUTION
# =============================================================================
class MemoryType(Enum):
    """Types of memory entries."""

    CONVERSATION = "conversation"
    FACT = "fact"
    CONTEXT = "context"
    TOOL_RESULT = "tool_result"


@dataclass
class MemoryEntry:
    """A single memory entry."""

    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
        }


class ADKMemory:
    """
    Memory system for ADK agents.

    Solution implements:
    - Short-term and long-term memory
    - Semantic search (mock)
    - Memory consolidation
    - Persistence
    """

    def __init__(
        self,
        max_short_term: int = 100,
        max_long_term: int = 10000,
        consolidation_threshold: float = 0.7,
    ):
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.consolidation_threshold = consolidation_threshold

    def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        importance: float = 0.5,
        **metadata,
    ) -> MemoryEntry:
        """Add a memory entry."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )

        self.short_term.append(entry)

        # Consolidate if needed
        if len(self.short_term) > self.max_short_term:
            self._consolidate()

        return entry

    def _consolidate(self) -> None:
        """Consolidate short-term to long-term memory."""
        # Move important memories to long-term
        to_move = []
        to_keep = []

        for entry in self.short_term:
            if entry.importance >= self.consolidation_threshold:
                to_move.append(entry)
            else:
                to_keep.append(entry)

        self.long_term.extend(to_move)

        # Keep only recent short-term
        self.short_term = to_keep[-self.max_short_term // 2 :]

        # Trim long-term if needed
        if len(self.long_term) > self.max_long_term:
            self.long_term.sort(key=lambda x: x.importance, reverse=True)
            self.long_term = self.long_term[: self.max_long_term]

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memories by relevance."""
        all_memories = self.short_term + self.long_term

        # Simple keyword matching (in production, use embeddings)
        query_words = set(query.lower().split())

        def relevance(entry: MemoryEntry) -> float:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            return overlap * entry.importance

        scored = [(m, relevance(m)) for m in all_memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, score in scored[:limit] if score > 0]

    def get_context(self, limit: int = 20) -> str:
        """Get relevant context as string."""
        recent = self.get_recent(limit)
        return "\n".join([f"- {m.content}" for m in recent])

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent memories."""
        all_memories = self.short_term + self.long_term
        all_memories.sort(key=lambda x: x.timestamp, reverse=True)
        return all_memories[:limit]

    def get_by_type(self, memory_type: MemoryType) -> List[MemoryEntry]:
        """Get memories by type."""
        all_memories = self.short_term + self.long_term
        return [m for m in all_memories if m.memory_type == memory_type]

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term = []

    def to_dict(self) -> Dict[str, Any]:
        """Export memory to dictionary."""
        return {
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term": [m.to_dict() for m in self.long_term],
        }

    def save(self, path: str) -> None:
        """Save memory to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> None:
        """Load memory from file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.short_term = [
            MemoryEntry(
                id=m["id"],
                content=m["content"],
                memory_type=MemoryType(m["type"]),
                timestamp=datetime.fromisoformat(m["timestamp"]),
                importance=m["importance"],
                metadata=m.get("metadata", {}),
            )
            for m in data.get("short_term", [])
        ]

        self.long_term = [
            MemoryEntry(
                id=m["id"],
                content=m["content"],
                memory_type=MemoryType(m["type"]),
                timestamp=datetime.fromisoformat(m["timestamp"]),
                importance=m["importance"],
                metadata=m.get("metadata", {}),
            )
            for m in data.get("long_term", [])
        ]


# =============================================================================
# Part 6: ADK Runner - SOLUTION
# =============================================================================
class RunnerMode(Enum):
    """Runner execution modes."""

    SINGLE = "single"
    LOOP = "loop"
    STREAMING = "streaming"


@dataclass
class RunResult:
    """Result of a runner execution."""

    success: bool
    output: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ADKRunner:
    """
    Runner for executing ADK agents.

    Solution implements:
    - Single and loop execution modes
    - Tool call handling
    - Timeout management
    - Progress tracking
    """

    def __init__(
        self,
        agent: ADKAgent,
        mode: RunnerMode = RunnerMode.SINGLE,
        max_iterations: int = 10,
        timeout: float = 300.0,
    ):
        self.agent = agent
        self.mode = mode
        self.max_iterations = max_iterations
        self.timeout = timeout
        self._iteration = 0
        self._tool_calls: List[Dict[str, Any]] = []

    async def run(self, input_text: str) -> RunResult:
        """Run the agent with given input."""
        start_time = time.time()
        self._iteration = 0
        self._tool_calls = []

        try:
            if self.mode == RunnerMode.SINGLE:
                output = await self._run_single(input_text)
            elif self.mode == RunnerMode.LOOP:
                output = await self._run_loop(input_text)
            else:
                output = await self._run_single(input_text)

            execution_time = time.time() - start_time

            return RunResult(
                success=True,
                output=output,
                tool_calls=self._tool_calls,
                execution_time=execution_time,
                iterations=self._iteration,
            )

        except asyncio.TimeoutError:
            return RunResult(
                success=False,
                output="Execution timed out",
                execution_time=self.timeout,
                iterations=self._iteration,
            )
        except Exception as e:
            return RunResult(
                success=False,
                output=str(e),
                execution_time=time.time() - start_time,
                iterations=self._iteration,
            )

    async def _run_single(self, input_text: str) -> str:
        """Run single iteration."""
        self._iteration = 1

        async with asyncio.timeout(self.timeout):
            return await self.agent.chat(input_text)

    async def _run_loop(self, input_text: str) -> str:
        """Run in loop mode until completion."""
        output = ""
        current_input = input_text

        while self._iteration < self.max_iterations:
            self._iteration += 1

            async with asyncio.timeout(self.timeout):
                response = await self.agent.chat(current_input)

            output = response

            # Check for completion markers
            if self._is_complete(response):
                break

            # Continue with follow-up
            current_input = self._get_follow_up(response)

        return output

    def _is_complete(self, response: str) -> bool:
        """Check if response indicates completion."""
        completion_markers = ["completed", "finished", "done", "final answer", "[DONE]"]
        response_lower = response.lower()
        return any(marker in response_lower for marker in completion_markers)

    def _get_follow_up(self, response: str) -> str:
        """Generate follow-up input."""
        return "Please continue or provide the final answer."

    def run_sync(self, input_text: str) -> RunResult:
        """Run synchronously."""
        return asyncio.run(self.run(input_text))

    def reset(self) -> None:
        """Reset runner state."""
        self._iteration = 0
        self._tool_calls = []
        self.agent.clear_conversation()


# =============================================================================
# Part 7: Streaming Handler - SOLUTION
# =============================================================================
class StreamEvent(Enum):
    """Types of streaming events."""

    START = "start"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    END = "end"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A chunk in a stream."""

    event: StreamEvent
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class StreamingHandler:
    """
    Handles streaming responses from ADK.

    Solution implements:
    - Token streaming
    - Event callbacks
    - Buffer management
    - Progress tracking
    """

    def __init__(self):
        self._buffer: List[StreamChunk] = []
        self._callbacks: Dict[StreamEvent, List[Callable]] = {
            event: [] for event in StreamEvent
        }
        self._is_streaming = False
        self._total_tokens = 0

    def on(self, event: StreamEvent, callback: Callable) -> None:
        """Register a callback for an event."""
        self._callbacks[event].append(callback)

    def off(self, event: StreamEvent, callback: Callable) -> bool:
        """Remove a callback."""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            return True
        return False

    async def stream(
        self, agent: ADKAgent, input_text: str
    ) -> AsyncIterator[StreamChunk]:
        """Stream agent response."""
        self._is_streaming = True
        self._buffer = []
        self._total_tokens = 0

        # Emit start event
        start_chunk = StreamChunk(event=StreamEvent.START)
        self._emit(StreamEvent.START, start_chunk)
        yield start_chunk

        try:
            # In production, this would use the actual streaming API
            # For now, simulate streaming with word-by-word output
            response = await agent.chat(input_text)

            # Simulate token streaming
            words = response.split()
            for word in words:
                self._total_tokens += 1
                chunk = StreamChunk(event=StreamEvent.TOKEN, content=word + " ")
                self._buffer.append(chunk)
                self._emit(StreamEvent.TOKEN, chunk)
                yield chunk
                await asyncio.sleep(0.01)  # Simulate streaming delay

            # End event
            end_chunk = StreamChunk(
                event=StreamEvent.END, metadata={"total_tokens": self._total_tokens}
            )
            self._emit(StreamEvent.END, end_chunk)
            yield end_chunk

        except Exception as e:
            error_chunk = StreamChunk(event=StreamEvent.ERROR, content=str(e))
            self._emit(StreamEvent.ERROR, error_chunk)
            yield error_chunk

        finally:
            self._is_streaming = False

    def _emit(self, event: StreamEvent, chunk: StreamChunk) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks[event]:
            try:
                callback(chunk)
            except Exception:
                pass  # Don't let callback errors break streaming

    def get_full_response(self) -> str:
        """Get the full response from buffer."""
        return "".join(
            chunk.content for chunk in self._buffer if chunk.event == StreamEvent.TOKEN
        )

    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming

    @property
    def token_count(self) -> int:
        """Get total token count."""
        return self._total_tokens


# =============================================================================
# Part 8: ADK Toolkit - SOLUTION
# =============================================================================
class ADKToolkit:
    """
    Collection of pre-built tools for ADK agents.

    Solution implements:
    - Common tools (search, calculator, etc.)
    - Tool organization
    - Easy registration
    """

    def __init__(self):
        self._tools: Dict[str, ADKToolWrapper] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Calculator tool
        self._tools["calculator"] = ADKToolWrapper(
            func=self._calculator,
            name="calculator",
            description="Evaluate mathematical expressions",
        )

        # Current time tool
        self._tools["get_current_time"] = ADKToolWrapper(
            func=self._get_current_time,
            name="get_current_time",
            description="Get the current date and time",
        )

        # String tool
        self._tools["text_processor"] = ADKToolWrapper(
            func=self._text_processor,
            name="text_processor",
            description="Process text with various operations",
        )

    @staticmethod
    def _calculator(expression: str) -> float:
        """Evaluate a math expression."""
        # Safe evaluation
        allowed = set("0123456789+-*/().^ ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid characters in expression")

        # Replace ^ with **
        expression = expression.replace("^", "**")
        return eval(expression)

    @staticmethod
    def _get_current_time() -> str:
        """Get current time."""
        return datetime.now().isoformat()

    @staticmethod
    def _text_processor(text: str, operation: str = "length") -> Union[int, str]:
        """Process text with various operations."""
        operations = {
            "length": lambda t: len(t),
            "upper": lambda t: t.upper(),
            "lower": lambda t: t.lower(),
            "reverse": lambda t: t[::-1],
            "word_count": lambda t: len(t.split()),
        }

        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")

        return operations[operation](text)

    def get(self, name: str) -> Optional[ADKToolWrapper]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list(self) -> List[str]:
        """List all available tools."""
        return list(self._tools.keys())

    def register(self, tool: ADKToolWrapper) -> None:
        """Register a custom tool."""
        self._tools[tool.name] = tool

    def get_for_agent(self, *names: str) -> List[ADKToolWrapper]:
        """Get tools for agent registration."""
        if not names:
            return list(self._tools.values())
        return [self._tools[name] for name in names if name in self._tools]

    def create_subset(self, *names: str) -> "ADKToolkit":
        """Create a toolkit with a subset of tools."""
        subset = ADKToolkit.__new__(ADKToolkit)
        subset._tools = {
            name: self._tools[name] for name in names if name in self._tools
        }
        return subset


# =============================================================================
# Part 9: Agent Monitor - SOLUTION
# =============================================================================
@dataclass
class AgentMetrics:
    """Metrics for an agent."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    avg_response_time: float = 0.0
    last_activity: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "avg_response_time": self.avg_response_time,
            "last_activity": (
                self.last_activity.isoformat() if self.last_activity else None
            ),
        }

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class AgentMonitor:
    """
    Monitors ADK agent performance.

    Solution implements:
    - Metrics tracking
    - Performance analysis
    - Alerts
    - Reporting
    """

    def __init__(self):
        self._metrics: Dict[str, AgentMetrics] = {}
        self._response_times: Dict[str, List[float]] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._alert_thresholds = {"error_rate": 0.2, "response_time": 10.0}

    def track(self, agent_id: str, result: RunResult) -> None:
        """Track an agent execution."""
        if agent_id not in self._metrics:
            self._metrics[agent_id] = AgentMetrics()
            self._response_times[agent_id] = []

        metrics = self._metrics[agent_id]
        metrics.total_requests += 1
        metrics.last_activity = datetime.now()

        if result.success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            self._check_alert(agent_id, "error_rate")

        metrics.total_tool_calls += len(result.tool_calls)

        # Update response time
        self._response_times[agent_id].append(result.execution_time)
        metrics.avg_response_time = sum(self._response_times[agent_id]) / len(
            self._response_times[agent_id]
        )

        self._check_alert(agent_id, "response_time", result.execution_time)

    def _check_alert(self, agent_id: str, alert_type: str, value: float = None) -> None:
        """Check and create alerts if thresholds exceeded."""
        metrics = self._metrics.get(agent_id)
        if not metrics:
            return

        if alert_type == "error_rate":
            if metrics.total_requests >= 10:  # Need minimum samples
                error_rate = metrics.failed_requests / metrics.total_requests
                if error_rate > self._alert_thresholds["error_rate"]:
                    self._create_alert(agent_id, "High error rate", error_rate)

        elif alert_type == "response_time" and value:
            if value > self._alert_thresholds["response_time"]:
                self._create_alert(agent_id, "Slow response", value)

    def _create_alert(self, agent_id: str, message: str, value: float) -> None:
        """Create an alert."""
        alert = {
            "agent_id": agent_id,
            "message": message,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        self._alerts.append(alert)

    def get_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for an agent."""
        return self._metrics.get(agent_id)

    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get all agent metrics."""
        return self._metrics.copy()

    def get_alerts(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by agent."""
        if agent_id:
            return [a for a in self._alerts if a["agent_id"] == agent_id]
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts = []

    def set_threshold(self, alert_type: str, value: float) -> None:
        """Set an alert threshold."""
        self._alert_thresholds[alert_type] = value

    def generate_report(self) -> str:
        """Generate a monitoring report."""
        lines = [
            "# Agent Monitoring Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        for agent_id, metrics in self._metrics.items():
            lines.append(f"## Agent: {agent_id}")
            lines.append(f"- Total Requests: {metrics.total_requests}")
            lines.append(f"- Success Rate: {metrics.success_rate:.1%}")
            lines.append(f"- Avg Response Time: {metrics.avg_response_time:.2f}s")
            lines.append(f"- Tool Calls: {metrics.total_tool_calls}")
            lines.append("")

        if self._alerts:
            lines.append("## Alerts")
            for alert in self._alerts[-10:]:  # Last 10 alerts
                lines.append(
                    f"- [{alert['timestamp']}] {alert['agent_id']}: {alert['message']}"
                )

        return "\n".join(lines)


# =============================================================================
# Part 10: ADK Agent Factory - SOLUTION
# =============================================================================
class AgentTemplate(Enum):
    """Pre-defined agent templates."""

    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    CREATIVE = "creative"


class ADKAgentFactory:
    """
    Factory for creating pre-configured agents.

    Solution implements:
    - Template-based creation
    - Custom configuration
    - Tool assignment
    - Batch creation
    """

    TEMPLATES = {
        AgentTemplate.ASSISTANT: {
            "system_instruction": "You are a helpful assistant. Provide clear, accurate, and helpful responses.",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        AgentTemplate.RESEARCHER: {
            "system_instruction": "You are a research assistant. Analyze information carefully, cite sources, and provide comprehensive answers.",
            "temperature": 0.3,
            "max_tokens": 4096,
        },
        AgentTemplate.CODER: {
            "system_instruction": "You are a coding assistant. Write clean, efficient, and well-documented code. Explain your solutions.",
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        AgentTemplate.ANALYST: {
            "system_instruction": "You are a data analyst. Analyze data thoroughly, identify patterns, and provide insights with supporting evidence.",
            "temperature": 0.4,
            "max_tokens": 4096,
        },
        AgentTemplate.CREATIVE: {
            "system_instruction": "You are a creative assistant. Generate imaginative and original content while maintaining quality.",
            "temperature": 1.0,
            "max_tokens": 2048,
        },
    }

    def __init__(self, toolkit: Optional[ADKToolkit] = None):
        self.toolkit = toolkit or ADKToolkit()
        self._created_agents: Dict[str, ADKAgent] = {}

    def create(
        self,
        name: str,
        template: AgentTemplate = AgentTemplate.ASSISTANT,
        tools: Optional[List[str]] = None,
        **overrides,
    ) -> ADKAgent:
        """Create an agent from template."""
        template_config = self.TEMPLATES[template].copy()
        template_config.update(overrides)

        # Get tools
        if tools:
            tool_wrappers = self.toolkit.get_for_agent(*tools)
        else:
            tool_wrappers = []

        config = ADKAgentConfig(name=name, tools=tool_wrappers, **template_config)

        agent = ADKAgent(config)
        self._created_agents[name] = agent

        return agent

    def create_custom(
        self,
        name: str,
        system_instruction: str,
        model: str = "gemini-2.0-flash",
        tools: Optional[List[str]] = None,
        **kwargs,
    ) -> ADKAgent:
        """Create a custom agent."""
        tool_wrappers = []
        if tools:
            tool_wrappers = self.toolkit.get_for_agent(*tools)

        config = ADKAgentConfig(
            name=name,
            model=model,
            system_instruction=system_instruction,
            tools=tool_wrappers,
            **kwargs,
        )

        agent = ADKAgent(config)
        self._created_agents[name] = agent

        return agent

    def create_batch(self, configs: List[Dict[str, Any]]) -> List[ADKAgent]:
        """Create multiple agents."""
        agents = []
        for cfg in configs:
            name = cfg.pop("name")
            template = AgentTemplate(cfg.pop("template", "assistant"))
            tools = cfg.pop("tools", None)

            agent = self.create(name, template, tools, **cfg)
            agents.append(agent)

        return agents

    def get_agent(self, name: str) -> Optional[ADKAgent]:
        """Get a created agent by name."""
        return self._created_agents.get(name)

    def list_agents(self) -> List[str]:
        """List all created agent names."""
        return list(self._created_agents.keys())

    def destroy(self, name: str) -> bool:
        """Destroy an agent."""
        if name in self._created_agents:
            del self._created_agents[name]
            return True
        return False

    @classmethod
    def get_template_info(cls, template: AgentTemplate) -> Dict[str, Any]:
        """Get information about a template."""
        return cls.TEMPLATES.get(template, {}).copy()

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available templates."""
        return [t.value for t in AgentTemplate]


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":

    async def main():
        # Create toolkit
        toolkit = ADKToolkit()
        print("Available tools:", toolkit.list())

        # Create factory
        factory = ADKAgentFactory(toolkit)

        # Create agent from template
        assistant = factory.create(
            name="my-assistant",
            template=AgentTemplate.ASSISTANT,
            tools=["calculator", "get_current_time"],
        )

        # Create custom agent
        researcher = factory.create_custom(
            name="researcher",
            system_instruction="You are a research specialist.",
            tools=["text_processor"],
        )

        # Use agent
        response = await assistant.chat("What is 2 + 2?")
        print(f"Assistant: {response}")

        # Create runner
        runner = ADKRunner(assistant, mode=RunnerMode.SINGLE)
        result = await runner.run("What time is it?")
        print(f"Runner result: {result}")

        # Monitor
        monitor = AgentMonitor()
        monitor.track(assistant.id, result)
        print(f"Metrics: {monitor.get_metrics(assistant.id).to_dict()}")

    asyncio.run(main())

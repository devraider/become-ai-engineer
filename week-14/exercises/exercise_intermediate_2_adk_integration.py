"""
Week 14 - Exercise 2: Google ADK Integration (Intermediate)

Learn to integrate with Google's Agent Development Kit (ADK):
- Tool wrappers for ADK compatibility
- Agent configuration and creation
- Memory and state management
- Streaming and monitoring

Run tests with: pytest tests/test_exercise_intermediate_2_adk_integration.py -v
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, AsyncIterator, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
from datetime import datetime
import json
import inspect


# =============================================================================
# Part 1: ADK Tool Wrapper
# =============================================================================
@dataclass
class ToolSchema:
    """Schema for a tool in ADK format."""

    name: str
    description: str
    parameters: dict  # JSON Schema format
    returns: dict  # Return type schema


class ADKToolWrapper:
    """
    Wraps Python functions as ADK-compatible tools.

    Example:
        >>> wrapper = ADKToolWrapper()
        >>> @wrapper.tool
        ... def search(query: str) -> str:
        ...     '''Search the web.'''
        ...     return f"Results for {query}"
        >>> schema = wrapper.get_schema("search")
        >>> schema.name
        'search'
    """

    def __init__(self):
        """Initialize the wrapper."""
        # TODO: Initialize tools dict for storing wrapped functions
        # Initialize schemas dict for storing tool schemas
        pass

    def tool(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> Callable:
        """
        Decorator to wrap a function as an ADK tool.

        Can be used with or without arguments:
        @wrapper.tool
        def my_func(): ...

        @wrapper.tool(name="custom_name")
        def my_func(): ...
        """
        # TODO: Handle both @tool and @tool() syntax
        # Extract name from function if not provided
        # Extract description from docstring if not provided
        # Generate schema from type hints
        # Store function and schema
        pass

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get schema for a tool by name."""
        # TODO: Return schema from schemas dict
        pass

    def get_all_schemas(self) -> list[ToolSchema]:
        """Get all tool schemas."""
        # TODO: Return list of all schemas
        pass

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        # TODO: Get function from tools dict and call with kwargs
        pass

    async def execute_async(self, name: str, **kwargs) -> Any:
        """Execute a tool asynchronously."""
        # TODO: Handle async functions properly
        # For sync functions, run in executor
        pass

    def _generate_schema(self, func: Callable) -> dict:
        """Generate JSON Schema from function signature."""
        # TODO: Use inspect to get parameters and type hints
        # Convert Python types to JSON Schema types
        pass


# =============================================================================
# Part 2: ADK Agent Configuration
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

    Example:
        >>> config = ADKAgentConfig(
        ...     name="research-agent",
        ...     model="gemini-2.0-flash",
        ...     temperature=0.7
        ... )
        >>> config.validate()
        True
    """

    name: str
    model: str = "gemini-2.0-flash"
    provider: ModelProvider = ModelProvider.GEMINI

    # Model settings
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95

    # Agent settings
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    timeout_seconds: float = 60.0

    # Tool settings
    tool_choice: str = "auto"  # auto, none, required
    parallel_tool_calls: bool = True

    # Memory settings
    memory_enabled: bool = True
    max_context_messages: int = 20

    # Safety settings
    safety_threshold: str = "medium"

    def validate(self) -> bool:
        """Validate configuration."""
        # TODO: Check all parameters are valid
        # temperature between 0 and 2
        # max_tokens > 0
        # max_iterations > 0
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        # TODO: Return dict representation
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "ADKAgentConfig":
        """Create from dictionary."""
        # TODO: Create config from dict, handling enum conversion
        pass

    def with_model(self, model: str) -> "ADKAgentConfig":
        """Create copy with different model."""
        # TODO: Return new config with updated model
        pass


# =============================================================================
# Part 3: ADK Agent
# =============================================================================
class ADKAgent:
    """
    Agent using Google ADK patterns.

    Example:
        >>> config = ADKAgentConfig(name="assistant")
        >>> agent = ADKAgent(config)
        >>> agent.add_tool(search_function)
        >>> response = await agent.run("Search for Python tutorials")
    """

    def __init__(self, config: ADKAgentConfig):
        """Initialize agent with configuration."""
        # TODO: Store config, initialize tool wrapper, memory, state
        pass

    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Add a tool to the agent."""
        # TODO: Wrap function and add to tools
        pass

    def add_tools(self, tools: list[Callable]) -> None:
        """Add multiple tools."""
        # TODO: Add each tool
        pass

    async def run(self, user_input: str) -> str:
        """Run the agent on user input."""
        # TODO: Implement agent loop:
        # 1. Add user message to context
        # 2. Call LLM (mock for exercise)
        # 3. If tool call, execute and loop
        # 4. Return final response
        pass

    async def _call_llm(self, messages: list[dict]) -> dict:
        """Call the LLM (mock implementation)."""
        # TODO: Mock LLM call that returns response or tool call
        pass

    async def _execute_tool(self, tool_call: dict) -> dict:
        """Execute a tool call."""
        # TODO: Execute tool and return result
        pass

    def get_conversation(self) -> list[dict]:
        """Get conversation history."""
        # TODO: Return formatted conversation
        pass

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        # TODO: Clear context/memory
        pass


# =============================================================================
# Part 4: Tool Schema Generator
# =============================================================================
class ToolSchemaGenerator:
    """
    Generates tool schemas from Python functions.

    Example:
        >>> def greet(name: str, formal: bool = False) -> str:
        ...     '''Greet a person.'''
        ...     return f"Hello, {name}!"
        >>> schema = ToolSchemaGenerator.from_function(greet)
        >>> schema["name"]
        'greet'
    """

    # Python type to JSON Schema type mapping
    TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    @classmethod
    def from_function(cls, func: Callable) -> dict:
        """Generate schema from function."""
        # TODO: Extract function metadata and generate schema
        # Include name, description, parameters with types
        pass

    @classmethod
    def _get_parameter_type(cls, annotation: Any) -> dict:
        """Convert Python type annotation to JSON Schema."""
        # TODO: Handle basic types, Optional, List, Dict
        pass

    @classmethod
    def _get_description(cls, func: Callable) -> str:
        """Extract description from docstring."""
        # TODO: Parse docstring and return description
        pass

    @classmethod
    def from_class(cls, tool_class: type) -> list[dict]:
        """Generate schemas from a class's public methods."""
        # TODO: Get all public methods and generate schemas
        pass

    @classmethod
    def validate_against_schema(
        cls, args: dict, schema: dict
    ) -> tuple[bool, list[str]]:
        """Validate arguments against schema."""
        # TODO: Check required fields and types
        # Return (valid, list of errors)
        pass


# =============================================================================
# Part 5: ADK Memory
# =============================================================================
class MemoryType(Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"


@dataclass
class MemoryEntry:
    """An entry in agent memory."""

    content: str
    memory_type: MemoryType
    timestamp: datetime
    relevance_score: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ADKMemory:
    """
    Memory system compatible with ADK agents.

    Example:
        >>> memory = ADKMemory()
        >>> memory.add("User likes Python", MemoryType.LONG_TERM, tags=["preference"])
        >>> results = memory.search("programming", k=5)
    """

    def __init__(self, short_term_capacity: int = 20, long_term_capacity: int = 1000):
        """Initialize memory stores."""
        # TODO: Initialize separate stores for each memory type
        pass

    def add(
        self,
        content: str,
        memory_type: MemoryType,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add entry to memory, returns entry ID."""
        # TODO: Create MemoryEntry and add to appropriate store
        pass

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID."""
        # TODO: Search all stores for entry
        pass

    def search(
        self, query: str, k: int = 5, memory_types: Optional[list[MemoryType]] = None
    ) -> list[MemoryEntry]:
        """Search memory for relevant entries."""
        # TODO: Search specified memory types (or all)
        # Rank by relevance to query
        pass

    def get_context(self, max_entries: int = 10) -> list[MemoryEntry]:
        """Get context-relevant memories for current conversation."""
        # TODO: Combine short-term recent + relevant long-term
        pass

    def consolidate(self) -> None:
        """Move important short-term memories to long-term."""
        # TODO: Find high-relevance short-term entries
        # Move to long-term storage
        pass

    def prune(self, memory_type: MemoryType) -> int:
        """Remove old/low-relevance entries."""
        # TODO: Remove entries below threshold
        # Return count of removed entries
        pass

    def export(self) -> dict:
        """Export memory state."""
        # TODO: Return serializable dict of all memories
        pass

    def import_state(self, state: dict) -> None:
        """Import memory state."""
        # TODO: Load memories from dict
        pass


# =============================================================================
# Part 6: ADK Runner
# =============================================================================
class RunnerMode(Enum):
    """Execution modes for the runner."""

    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"


@dataclass
class RunResult:
    """Result from running an agent."""

    output: str
    tool_calls: list[dict]
    iterations: int
    total_time_ms: float
    token_usage: dict = field(default_factory=dict)


class ADKRunner:
    """
    Runner for executing ADK agents.

    Example:
        >>> runner = ADKRunner(mode=RunnerMode.ASYNC)
        >>> result = await runner.run(agent, "Hello!")
        >>> result.output
        'Hi there!'
    """

    def __init__(
        self,
        mode: RunnerMode = RunnerMode.SYNC,
        max_iterations: int = 10,
        timeout_seconds: float = 60.0,
    ):
        """Initialize runner."""
        # TODO: Store mode and execution settings
        pass

    def run(self, agent: ADKAgent, input_text: str) -> RunResult:
        """Run agent synchronously."""
        # TODO: Execute agent and collect results
        pass

    async def run_async(self, agent: ADKAgent, input_text: str) -> RunResult:
        """Run agent asynchronously."""
        # TODO: Async execution with proper await handling
        pass

    async def run_streaming(
        self, agent: ADKAgent, input_text: str
    ) -> AsyncIterator[str]:
        """Run agent with streaming output."""
        # TODO: Yield tokens as they're generated
        pass

    def run_batch(
        self, agent: ADKAgent, inputs: list[str], parallel: bool = True
    ) -> list[RunResult]:
        """Run agent on multiple inputs."""
        # TODO: Execute batch, optionally in parallel
        pass

    def _create_result(
        self, output: str, tool_calls: list, iterations: int, time_ms: float
    ) -> RunResult:
        """Create RunResult from execution data."""
        # TODO: Construct and return RunResult
        pass


# =============================================================================
# Part 7: Streaming Handler
# =============================================================================
class StreamEvent(Enum):
    """Types of streaming events."""

    TOKEN = "token"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    THINKING = "thinking"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A chunk from streaming output."""

    event: StreamEvent
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)


class StreamingHandler:
    """
    Handles streaming output from agents.

    Example:
        >>> handler = StreamingHandler()
        >>> handler.on_token(lambda t: print(t, end=""))
        >>> async for chunk in agent.stream("Hello"):
        ...     handler.process(chunk)
    """

    def __init__(self):
        """Initialize handler with empty callbacks."""
        # TODO: Initialize callback dicts for each event type
        pass

    def on_token(self, callback: Callable[[str], None]) -> None:
        """Register callback for token events."""
        # TODO: Store callback for TOKEN events
        pass

    def on_tool_call(self, callback: Callable[[dict], None]) -> None:
        """Register callback for tool call events."""
        # TODO: Store callback for tool call events
        pass

    def on_complete(self, callback: Callable[[str], None]) -> None:
        """Register callback for completion."""
        # TODO: Store callback for COMPLETE events
        pass

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for errors."""
        # TODO: Store callback for ERROR events
        pass

    def process(self, chunk: StreamChunk) -> None:
        """Process a streaming chunk."""
        # TODO: Call appropriate callback based on event type
        pass

    async def collect(self, stream: AsyncIterator[StreamChunk]) -> str:
        """Collect all chunks and return final output."""
        # TODO: Process all chunks and concatenate tokens
        pass


# =============================================================================
# Part 8: ADK Toolkit
# =============================================================================
class ADKToolkit:
    """
    Collection of useful tools for ADK agents.

    Example:
        >>> toolkit = ADKToolkit()
        >>> tools = toolkit.get_all()
        >>> agent.add_tools(tools)
    """

    def __init__(self):
        """Initialize toolkit."""
        # TODO: Initialize tool wrapper
        pass

    def get_all(self) -> list[Callable]:
        """Get all tools in the toolkit."""
        # TODO: Return list of all tool functions
        pass

    def get_by_category(self, category: str) -> list[Callable]:
        """Get tools by category."""
        # TODO: Filter tools by category tag
        pass

    # Built-in tools
    @staticmethod
    def current_datetime() -> str:
        """Get the current date and time."""
        # TODO: Return formatted datetime string
        pass

    @staticmethod
    def calculate(expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # TODO: Parse and evaluate expression
        # Only allow safe math operations
        pass

    @staticmethod
    def json_parse(json_string: str) -> dict:
        """Parse a JSON string."""
        # TODO: Parse JSON and return dict
        pass

    @staticmethod
    def json_format(data: dict, indent: int = 2) -> str:
        """Format data as JSON string."""
        # TODO: Convert to formatted JSON string
        pass

    @staticmethod
    def text_length(text: str) -> int:
        """Get the length of text."""
        # TODO: Return character count
        pass

    @staticmethod
    def text_search(text: str, pattern: str) -> list[str]:
        """Search for pattern in text."""
        # TODO: Find all matches of pattern
        pass


# =============================================================================
# Part 9: Agent Monitor
# =============================================================================
@dataclass
class AgentMetrics:
    """Metrics for agent execution."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_tool_calls: int = 0
    average_iterations: float = 0.0
    average_time_ms: float = 0.0
    total_tokens: int = 0


class AgentMonitor:
    """
    Monitor for tracking agent behavior and performance.

    Example:
        >>> monitor = AgentMonitor()
        >>> monitor.start_run("agent-1")
        >>> monitor.log_tool_call("agent-1", "search", {"query": "test"})
        >>> monitor.end_run("agent-1", success=True)
        >>> metrics = monitor.get_metrics("agent-1")
    """

    def __init__(self):
        """Initialize monitor."""
        # TODO: Initialize tracking dicts for agents
        pass

    def start_run(self, agent_id: str) -> str:
        """Start tracking a run, returns run ID."""
        # TODO: Create run record and return ID
        pass

    def end_run(
        self,
        run_id: str,
        success: bool,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a tracked run."""
        # TODO: Update run record with result
        pass

    def log_tool_call(
        self, run_id: str, tool_name: str, args: dict, result: Any = None
    ) -> None:
        """Log a tool call within a run."""
        # TODO: Add tool call to run record
        pass

    def log_iteration(self, run_id: str, iteration: int) -> None:
        """Log an iteration within a run."""
        # TODO: Update iteration count
        pass

    def get_metrics(self, agent_id: str) -> AgentMetrics:
        """Get metrics for an agent."""
        # TODO: Calculate and return metrics
        pass

    def get_run_history(self, agent_id: str, limit: int = 10) -> list[dict]:
        """Get recent run history."""
        # TODO: Return recent runs for agent
        pass

    def export_metrics(self) -> dict:
        """Export all metrics."""
        # TODO: Return dict of all agent metrics
        pass


# =============================================================================
# Part 10: ADK Agent Factory
# =============================================================================
class AgentTemplate(Enum):
    """Pre-defined agent templates."""

    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"


class ADKAgentFactory:
    """
    Factory for creating pre-configured ADK agents.

    Example:
        >>> factory = ADKAgentFactory()
        >>> agent = factory.create(AgentTemplate.RESEARCHER)
        >>> agent = factory.create_custom(
        ...     name="my-agent",
        ...     tools=[search, analyze],
        ...     system_prompt="You are a data analyst."
        ... )
    """

    # Template configurations
    TEMPLATES = {
        AgentTemplate.ASSISTANT: {
            "system_prompt": "You are a helpful AI assistant.",
            "tools": ["current_datetime", "calculate"],
            "temperature": 0.7,
        },
        AgentTemplate.RESEARCHER: {
            "system_prompt": "You are a thorough research assistant.",
            "tools": ["search", "summarize", "cite"],
            "temperature": 0.3,
        },
        AgentTemplate.CODER: {
            "system_prompt": "You are an expert programmer.",
            "tools": ["run_code", "analyze_code", "explain_code"],
            "temperature": 0.2,
        },
        AgentTemplate.ANALYST: {
            "system_prompt": "You are a data analyst.",
            "tools": ["calculate", "visualize", "analyze"],
            "temperature": 0.4,
        },
    }

    def __init__(self, default_model: str = "gemini-2.0-flash"):
        """Initialize factory."""
        # TODO: Store default model, initialize toolkit
        pass

    def create(
        self,
        template: AgentTemplate,
        name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ADKAgent:
        """Create agent from template."""
        # TODO: Get template config, create ADKAgentConfig, build agent
        pass

    def create_custom(
        self,
        name: str,
        tools: list[Callable],
        system_prompt: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> ADKAgent:
        """Create custom agent."""
        # TODO: Create custom config and agent with provided tools
        pass

    def register_template(self, template_name: str, config: dict) -> None:
        """Register a custom template."""
        # TODO: Add template to TEMPLATES
        pass

    def list_templates(self) -> list[str]:
        """List available templates."""
        # TODO: Return template names
        pass


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    import asyncio

    # Create agent using factory
    factory = ADKAgentFactory()

    # Create from template
    config = ADKAgentConfig(
        name="demo-agent",
        model="gemini-2.0-flash",
        system_prompt="You are a helpful assistant that can do math.",
    )

    agent = ADKAgent(config)

    # Add custom tools
    @agent.add_tool
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @agent.add_tool
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    # Set up monitoring
    monitor = AgentMonitor()

    # Run agent with monitoring
    async def run_with_monitoring():
        run_id = monitor.start_run("demo-agent")
        try:
            result = await agent.run("What is 5 + 3?")
            monitor.end_run(run_id, success=True, output=result)
            print(f"Result: {result}")
        except Exception as e:
            monitor.end_run(run_id, success=False, error=str(e))

    # Run
    asyncio.run(run_with_monitoring())

    # Get metrics
    metrics = monitor.get_metrics("demo-agent")
    print(f"Total runs: {metrics.total_runs}")

"""
Week 10 - Exercise 3: Agents & Persistence
==========================================
Build production-ready agents with tools and persistence.

Topics covered:
- ReAct agent pattern
- Tool integration
- Checkpointing for persistence
- Human-in-the-loop workflows
- Streaming execution
"""

from typing import TypedDict, Annotated, Sequence, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import uuid


# =============================================================================
# TASK 1: Define Agent State with Messages
# =============================================================================
@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "user", "assistant", "tool"
    content: str
    name: str | None = None  # Tool name for tool messages
    tool_calls: list[dict] | None = None  # For assistant tool calls
    tool_call_id: str | None = None  # For tool responses
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "timestamp": self.timestamp.isoformat(),
        }


def add_messages(existing: list, new: list) -> list:
    """Reducer that appends new messages to existing list."""
    return existing + new


class AgentState(TypedDict):
    """
    TODO: Define agent state:
    - messages: Annotated[list[Message], add_messages] - Conversation history
    - current_step: str - Current agent step
    - tool_outputs: dict - Results from tool calls
    - final_response: str - Final response to user
    """

    pass


# =============================================================================
# TASK 2: Implement Tool Base Class
# =============================================================================
class Tool(ABC):
    """
    TODO: Base class for agent tools.

    Properties:
    - name: str - Tool identifier
    - description: str - What the tool does
    - parameters: dict - JSON schema for parameters

    Methods:
    - execute(**kwargs) -> str: Run the tool
    - get_schema() -> dict: Return OpenAI-style function schema
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def get_schema(self) -> dict:
        """Return function calling schema."""
        # TODO: Return OpenAI-style function schema
        pass

    def __call__(self, **kwargs) -> str:
        return self.execute(**kwargs)


# =============================================================================
# TASK 3: Implement Calculator Tool
# =============================================================================
class CalculatorTool(Tool):
    """
    TODO: Calculator tool for mathematical expressions.

    Name: "calculator"
    Description: "Evaluate mathematical expressions"
    Parameters: {"expression": {"type": "string", "description": "Math expression"}}

    Execute: Safely evaluate the expression using eval with restricted globals
    """

    @property
    def name(self) -> str:
        pass

    @property
    def description(self) -> str:
        pass

    @property
    def parameters(self) -> dict:
        pass

    def execute(self, expression: str) -> str:
        """Safely evaluate mathematical expression."""
        pass


# =============================================================================
# TASK 4: Implement Search Tool
# =============================================================================
class SearchTool(Tool):
    """
    TODO: Search tool that simulates web search.

    Name: "search"
    Description: "Search for information on the web"
    Parameters: {"query": {"type": "string", "description": "Search query"}}

    For this exercise, use a mock knowledge base.
    """

    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level programming language known for readability.",
            "langgraph": "LangGraph is a library for building stateful AI agents.",
            "machine learning": "ML is a subset of AI that enables learning from data.",
            "langchain": "LangChain provides building blocks for LLM applications.",
        }

    @property
    def name(self) -> str:
        pass

    @property
    def description(self) -> str:
        pass

    @property
    def parameters(self) -> dict:
        pass

    def execute(self, query: str) -> str:
        """Search the knowledge base."""
        pass


# =============================================================================
# TASK 5: Implement Tool Registry
# =============================================================================
class ToolRegistry:
    """
    TODO: Registry for managing available tools.

    Methods:
    - register(tool: Tool) - Add tool to registry
    - get(name: str) -> Tool | None - Get tool by name
    - list_tools() -> list[str] - List all tool names
    - get_all_schemas() -> list[dict] - Get all tool schemas
    - execute(name: str, **kwargs) -> str - Execute a tool by name
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        pass

    def get(self, name: str) -> Tool | None:
        pass

    def list_tools(self) -> list[str]:
        pass

    def get_all_schemas(self) -> list[dict]:
        pass

    def execute(self, name: str, **kwargs) -> str:
        pass


# =============================================================================
# TASK 6: Implement ReAct Agent Logic
# =============================================================================
class ReActAgent:
    """
    TODO: Implement the ReAct (Reasoning + Acting) pattern.

    The agent follows this loop:
    1. Observe: Look at current state
    2. Think: Decide what to do (call tool or respond)
    3. Act: Execute tool or generate response
    4. Repeat until done

    For this exercise, implement decision logic without actual LLM calls.
    Use pattern matching to decide tool usage.
    """

    def __init__(self, tools: ToolRegistry, max_iterations: int = 5):
        self.tools = tools
        self.max_iterations = max_iterations

    def should_use_tool(
        self, messages: list[Message]
    ) -> tuple[bool, str | None, dict | None]:
        """
        TODO: Decide if a tool should be used.

        Analyze the last user message and determine:
        - If it needs calculation -> use calculator
        - If it needs information -> use search
        - Otherwise -> respond directly

        Returns: (use_tool: bool, tool_name: str | None, tool_args: dict | None)
        """
        pass

    def generate_response(self, messages: list[Message], tool_outputs: dict) -> str:
        """
        TODO: Generate final response based on conversation and tool outputs.

        If tool outputs exist, incorporate them into the response.
        Otherwise, generate a helpful response based on the query.
        """
        pass

    def run(self, user_message: str) -> dict:
        """
        TODO: Run the ReAct loop.

        Steps:
        1. Initialize state with user message
        2. Loop up to max_iterations:
           a. Check if tool should be used
           b. If yes, execute tool and add to outputs
           c. If no, generate final response and break
        3. Return final state
        """
        pass


# =============================================================================
# TASK 7: Implement Checkpointer
# =============================================================================
@dataclass
class Checkpoint:
    """A checkpoint of agent state at a point in time."""

    thread_id: str
    checkpoint_id: str
    state: dict
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: str | None = None


class MemoryCheckpointer:
    """
    TODO: In-memory checkpointer for conversation persistence.

    Methods:
    - put(thread_id: str, state: dict) -> str: Save checkpoint, return checkpoint_id
    - get(thread_id: str) -> Checkpoint | None: Get latest checkpoint for thread
    - get_history(thread_id: str) -> list[Checkpoint]: Get all checkpoints for thread
    - delete(thread_id: str) -> bool: Delete all checkpoints for thread
    """

    def __init__(self):
        self._checkpoints: dict[str, list[Checkpoint]] = {}

    def put(self, thread_id: str, state: dict) -> str:
        """Save a checkpoint and return its ID."""
        pass

    def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint for a thread."""
        pass

    def get_history(self, thread_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a thread."""
        pass

    def delete(self, thread_id: str) -> bool:
        """Delete all checkpoints for a thread."""
        pass


# =============================================================================
# TASK 8: Implement Persistent Agent
# =============================================================================
class PersistentAgent:
    """
    TODO: Agent with conversation persistence.

    Uses checkpointer to:
    - Save state after each interaction
    - Resume conversations from previous state
    - Track conversation history across sessions
    """

    def __init__(self, checkpointer: MemoryCheckpointer = None):
        self.tools = ToolRegistry()
        self.tools.register(CalculatorTool())
        self.tools.register(SearchTool())
        self.checkpointer = checkpointer or MemoryCheckpointer()
        self.react_agent = ReActAgent(self.tools)

    def chat(self, message: str, thread_id: str) -> dict:
        """
        TODO: Chat with persistence.

        Steps:
        1. Load existing state from checkpoint (if exists)
        2. Add new user message to state
        3. Run agent logic
        4. Save new checkpoint
        5. Return response
        """
        pass

    def get_conversation(self, thread_id: str) -> list[Message]:
        """Get conversation history for a thread."""
        pass

    def clear_conversation(self, thread_id: str) -> bool:
        """Clear conversation history for a thread."""
        pass


# =============================================================================
# TASK 9: Implement Human-in-the-Loop
# =============================================================================
class HumanInTheLoopState(TypedDict):
    """State for human approval workflow."""

    request: str
    analysis: str
    approval_required: bool
    approved: bool | None
    result: str


class HumanInTheLoopAgent:
    """
    TODO: Agent that pauses for human approval on sensitive actions.

    Workflow:
    1. Analyze request
    2. If sensitive action detected, pause for approval
    3. Human approves or rejects
    4. Continue or abort based on decision
    """

    SENSITIVE_KEYWORDS = ["delete", "remove", "modify", "update", "send"]

    def __init__(self):
        self.pending_approvals: dict[str, HumanInTheLoopState] = {}

    def analyze_request(self, request: str) -> dict:
        """
        TODO: Analyze if request requires approval.

        Check for sensitive keywords and set approval_required.
        """
        pass

    def submit_request(self, request_id: str, request: str) -> dict:
        """
        TODO: Submit a request for processing.

        If approval required, return status "pending_approval"
        Otherwise, execute and return result.
        """
        pass

    def approve(self, request_id: str) -> dict:
        """
        TODO: Approve a pending request.
        """
        pass

    def reject(self, request_id: str) -> dict:
        """
        TODO: Reject a pending request.
        """
        pass

    def execute_request(self, state: HumanInTheLoopState) -> dict:
        """
        TODO: Execute the request after approval.
        """
        pass


# =============================================================================
# TASK 10: Implement Streaming Agent
# =============================================================================
class StreamingAgent:
    """
    TODO: Agent that supports streaming responses.

    Yields events during execution:
    - "thinking": Agent is processing
    - "tool_call": Agent is calling a tool
    - "tool_result": Tool returned result
    - "response": Final response tokens
    """

    def __init__(self):
        self.tools = ToolRegistry()
        self.tools.register(CalculatorTool())
        self.tools.register(SearchTool())

    def stream(self, message: str):
        """
        TODO: Stream agent execution.

        Yields events as dict:
        {"event": "thinking", "data": "Analyzing query..."}
        {"event": "tool_call", "data": {"tool": "calculator", "args": {...}}}
        {"event": "tool_result", "data": "42"}
        {"event": "response", "data": "The answer is 42"}

        Use yield to produce each event.
        """
        pass

    async def astream(self, message: str):
        """
        TODO: Async streaming version.

        Same as stream but async.
        """
        pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 10 - Exercise 3: Agents & Persistence")
    print("=" * 60)

    # Test Task 3: Calculator Tool
    print("\n--- Task 3: Calculator Tool ---")
    # calc = CalculatorTool()
    # print(f"2 + 3 * 4 = {calc.execute(expression='2 + 3 * 4')}")

    # Test Task 4: Search Tool
    print("\n--- Task 4: Search Tool ---")
    # search = SearchTool()
    # print(f"Search 'python': {search.execute(query='python')}")

    # Test Task 5: Tool Registry
    print("\n--- Task 5: Tool Registry ---")
    # registry = ToolRegistry()
    # registry.register(CalculatorTool())
    # registry.register(SearchTool())
    # print(f"Tools: {registry.list_tools()}")

    # Test Task 6: ReAct Agent
    print("\n--- Task 6: ReAct Agent ---")
    # registry = ToolRegistry()
    # registry.register(CalculatorTool())
    # registry.register(SearchTool())
    # agent = ReActAgent(registry)
    # result = agent.run("What is 15 * 7?")
    # print(f"Agent result: {result}")

    # Test Task 8: Persistent Agent
    print("\n--- Task 8: Persistent Agent ---")
    # agent = PersistentAgent()
    # result1 = agent.chat("Hi, my name is Alice", "thread-1")
    # result2 = agent.chat("What's my name?", "thread-1")
    # print(f"First: {result1}")
    # print(f"Second: {result2}")

    # Test Task 10: Streaming Agent
    print("\n--- Task 10: Streaming Agent ---")
    # stream_agent = StreamingAgent()
    # for event in stream_agent.stream("Calculate 10 + 20"):
    #     print(f"Event: {event}")

    print("\n✅ Uncomment tests as you implement each task!")

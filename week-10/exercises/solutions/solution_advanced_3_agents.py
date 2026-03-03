"""
Solutions for Week 10 - Exercise 3: Agents & Persistence
========================================================
"""

from typing import TypedDict, Annotated, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import uuid
import math
import re


# =============================================================================
# TASK 1: Define Agent State with Messages
# =============================================================================
@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
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
    """Reducer that appends new messages."""
    return existing + new


class AgentState(TypedDict):
    """Agent state with message history."""

    messages: Annotated[list, add_messages]
    current_step: str
    tool_outputs: dict
    final_response: str


# =============================================================================
# TASK 2: Implement Tool Base Class
# =============================================================================
class Tool(ABC):
    """Base class for agent tools."""

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
        """Return OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, **kwargs) -> str:
        return self.execute(**kwargs)


# =============================================================================
# TASK 3: Implement Calculator Tool
# =============================================================================
class CalculatorTool(Tool):
    """Calculator tool for mathematical expressions."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str) -> str:
        """Safely evaluate mathematical expression."""
        try:
            allowed = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "pi": math.pi,
                "e": math.e,
                "abs": abs,
                "pow": pow,
                "round": round,
            }
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# TASK 4: Implement Search Tool
# =============================================================================
class SearchTool(Tool):
    """Search tool that simulates web search."""

    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level programming language known for readability.",
            "langgraph": "LangGraph is a library for building stateful AI agents.",
            "machine learning": "ML is a subset of AI that enables learning from data.",
            "langchain": "LangChain provides building blocks for LLM applications.",
        }

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information on the web"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }

    def execute(self, query: str) -> str:
        """Search the knowledge base."""
        query_lower = query.lower()
        results = []

        for key, value in self.knowledge_base.items():
            if key in query_lower or any(word in key for word in query_lower.split()):
                results.append(value)

        if results:
            return " | ".join(results)
        return "No results found for your query."


# =============================================================================
# TASK 5: Implement Tool Registry
# =============================================================================
class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict]:
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if not tool:
            return f"Tool '{name}' not found"
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {name}: {e}"


# =============================================================================
# TASK 6: Implement ReAct Agent Logic
# =============================================================================
class ReActAgent:
    """ReAct (Reasoning + Acting) agent."""

    def __init__(self, tools: ToolRegistry, max_iterations: int = 5):
        self.tools = tools
        self.max_iterations = max_iterations

    def should_use_tool(
        self, messages: list[Message]
    ) -> tuple[bool, str | None, dict | None]:
        """Decide if a tool should be used."""
        if not messages:
            return False, None, None

        last_msg = messages[-1]
        if last_msg.role != "user":
            return False, None, None

        content = last_msg.content.lower()

        # Check for calculator need
        math_keywords = [
            "calculate",
            "compute",
            "what is",
            "how much",
            "+",
            "-",
            "*",
            "/",
            "=",
        ]
        if any(kw in content for kw in math_keywords) or any(
            c.isdigit() for c in content
        ):
            # Extract expression
            expr_match = re.search(r"[\d\s\+\-\*/\(\)\.]+", content)
            if expr_match:
                expr = expr_match.group().strip()
                if expr and any(c.isdigit() for c in expr):
                    return True, "calculator", {"expression": expr}

        # Check for search need
        search_keywords = [
            "what is",
            "who is",
            "tell me about",
            "search",
            "find",
            "information",
        ]
        if any(kw in content for kw in search_keywords):
            return True, "search", {"query": content}

        return False, None, None

    def generate_response(self, messages: list[Message], tool_outputs: dict) -> str:
        """Generate final response."""
        if tool_outputs:
            tool_results = []
            for tool_name, result in tool_outputs.items():
                tool_results.append(f"{tool_name}: {result}")
            return f"Based on my research: {'; '.join(tool_results)}"

        # Simple response for non-tool queries
        if messages:
            last_msg = messages[-1].content
            if "hello" in last_msg.lower() or "hi" in last_msg.lower():
                return "Hello! How can I help you today?"
            return "I understand your question. How can I assist you further?"

        return "I'm ready to help!"

    def run(self, user_message: str) -> dict:
        """Run the ReAct loop."""
        messages = [Message(role="user", content=user_message)]
        tool_outputs = {}

        for _ in range(self.max_iterations):
            use_tool, tool_name, tool_args = self.should_use_tool(messages)

            if use_tool and tool_name:
                result = self.tools.execute(tool_name, **tool_args)
                tool_outputs[tool_name] = result
                messages.append(Message(role="tool", content=result, name=tool_name))
            else:
                break

        response = self.generate_response(messages, tool_outputs)
        messages.append(Message(role="assistant", content=response))

        return {
            "messages": messages,
            "tool_outputs": tool_outputs,
            "response": response,
        }


# =============================================================================
# TASK 7: Implement Checkpointer
# =============================================================================
@dataclass
class Checkpoint:
    """A checkpoint of agent state."""

    thread_id: str
    checkpoint_id: str
    state: dict
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: str | None = None


class MemoryCheckpointer:
    """In-memory checkpointer for conversation persistence."""

    def __init__(self):
        self._checkpoints: dict[str, list[Checkpoint]] = {}

    def put(self, thread_id: str, state: dict) -> str:
        """Save a checkpoint."""
        checkpoint_id = str(uuid.uuid4())

        parent_id = None
        if thread_id in self._checkpoints and self._checkpoints[thread_id]:
            parent_id = self._checkpoints[thread_id][-1].checkpoint_id

        checkpoint = Checkpoint(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            state=state,
            parent_id=parent_id,
        )

        if thread_id not in self._checkpoints:
            self._checkpoints[thread_id] = []
        self._checkpoints[thread_id].append(checkpoint)

        return checkpoint_id

    def get(self, thread_id: str) -> Checkpoint | None:
        """Get the latest checkpoint."""
        if thread_id in self._checkpoints and self._checkpoints[thread_id]:
            return self._checkpoints[thread_id][-1]
        return None

    def get_history(self, thread_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a thread."""
        return self._checkpoints.get(thread_id, [])

    def delete(self, thread_id: str) -> bool:
        """Delete all checkpoints for a thread."""
        if thread_id in self._checkpoints:
            del self._checkpoints[thread_id]
            return True
        return False


# =============================================================================
# TASK 8: Implement Persistent Agent
# =============================================================================
class PersistentAgent:
    """Agent with conversation persistence."""

    def __init__(self, checkpointer: MemoryCheckpointer = None):
        self.tools = ToolRegistry()
        self.tools.register(CalculatorTool())
        self.tools.register(SearchTool())
        self.checkpointer = checkpointer or MemoryCheckpointer()
        self.react_agent = ReActAgent(self.tools)

    def chat(self, message: str, thread_id: str) -> dict:
        """Chat with persistence."""
        # Load existing state
        checkpoint = self.checkpointer.get(thread_id)
        if checkpoint:
            messages = [
                Message(**m) if isinstance(m, dict) else m
                for m in checkpoint.state.get("messages", [])
            ]
        else:
            messages = []

        # Add user message
        messages.append(Message(role="user", content=message))

        # Run agent
        result = self.react_agent.run(message)
        response = result["response"]

        # Add assistant message
        messages.append(Message(role="assistant", content=response))

        # Save checkpoint
        state = {
            "messages": [m.to_dict() for m in messages],
            "tool_outputs": result.get("tool_outputs", {}),
        }
        self.checkpointer.put(thread_id, state)

        return {"response": response, "messages": messages}

    def get_conversation(self, thread_id: str) -> list[Message]:
        """Get conversation history."""
        checkpoint = self.checkpointer.get(thread_id)
        if checkpoint:
            return [
                Message(**m) if isinstance(m, dict) else m
                for m in checkpoint.state.get("messages", [])
            ]
        return []

    def clear_conversation(self, thread_id: str) -> bool:
        """Clear conversation history."""
        return self.checkpointer.delete(thread_id)


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
    """Agent that pauses for human approval."""

    SENSITIVE_KEYWORDS = ["delete", "remove", "modify", "update", "send", "execute"]

    def __init__(self):
        self.pending_approvals: dict[str, HumanInTheLoopState] = {}

    def analyze_request(self, request: str) -> dict:
        """Analyze if request requires approval."""
        request_lower = request.lower()
        is_sensitive = any(kw in request_lower for kw in self.SENSITIVE_KEYWORDS)

        return {
            "analysis": f"Request analyzed: {'sensitive' if is_sensitive else 'safe'}",
            "approval_required": is_sensitive,
        }

    def submit_request(self, request_id: str, request: str) -> dict:
        """Submit a request for processing."""
        analysis = self.analyze_request(request)

        state: HumanInTheLoopState = {
            "request": request,
            "analysis": analysis["analysis"],
            "approval_required": analysis["approval_required"],
            "approved": None,
            "result": "",
        }

        if analysis["approval_required"]:
            self.pending_approvals[request_id] = state
            return {"status": "pending_approval", "analysis": analysis["analysis"]}

        # Execute immediately if not sensitive
        return self.execute_request(state)

    def approve(self, request_id: str) -> dict:
        """Approve a pending request."""
        if request_id not in self.pending_approvals:
            return {"error": "Request not found"}

        state = self.pending_approvals.pop(request_id)
        state["approved"] = True
        return self.execute_request(state)

    def reject(self, request_id: str) -> dict:
        """Reject a pending request."""
        if request_id not in self.pending_approvals:
            return {"error": "Request not found"}

        state = self.pending_approvals.pop(request_id)
        state["approved"] = False
        state["result"] = "Request rejected by user"
        return {"status": "rejected", "result": state["result"]}

    def execute_request(self, state: HumanInTheLoopState) -> dict:
        """Execute the request."""
        state["result"] = f"Executed: {state['request']}"
        return {
            "status": "completed",
            "result": state["result"],
            "approved": state["approved"],
        }


# =============================================================================
# TASK 10: Implement Streaming Agent
# =============================================================================
class StreamingAgent:
    """Agent that supports streaming responses."""

    def __init__(self):
        self.tools = ToolRegistry()
        self.tools.register(CalculatorTool())
        self.tools.register(SearchTool())

    def stream(self, message: str):
        """Stream agent execution."""
        yield {"event": "thinking", "data": "Analyzing query..."}

        # Check for tool needs
        message_lower = message.lower()

        if any(c.isdigit() for c in message) or "calculate" in message_lower:
            yield {
                "event": "tool_call",
                "data": {"tool": "calculator", "args": {"expression": message}},
            }
            result = self.tools.execute("calculator", expression=message)
            yield {"event": "tool_result", "data": result}
            yield {"event": "response", "data": f"The result is: {result}"}

        elif "what is" in message_lower or "search" in message_lower:
            yield {
                "event": "tool_call",
                "data": {"tool": "search", "args": {"query": message}},
            }
            result = self.tools.execute("search", query=message)
            yield {"event": "tool_result", "data": result}
            yield {"event": "response", "data": f"Here's what I found: {result}"}

        else:
            yield {
                "event": "response",
                "data": "I'm here to help! Please ask me a question.",
            }

    async def astream(self, message: str):
        """Async streaming version."""
        for event in self.stream(message):
            yield event


if __name__ == "__main__":
    print("=" * 60)
    print("Week 10 - Solution 3: Agents & Persistence")
    print("=" * 60)

    # Test Calculator
    print("\n--- Calculator Tool ---")
    calc = CalculatorTool()
    print(f"2 + 3 * 4 = {calc.execute(expression='2 + 3 * 4')}")

    # Test Search
    print("\n--- Search Tool ---")
    search = SearchTool()
    print(f"Search 'python': {search.execute(query='python')}")

    # Test Registry
    print("\n--- Tool Registry ---")
    registry = ToolRegistry()
    registry.register(calc)
    registry.register(search)
    print(f"Tools: {registry.list_tools()}")

    # Test Persistent Agent
    print("\n--- Persistent Agent ---")
    agent = PersistentAgent()
    result1 = agent.chat("Hi, I'm Alice", "thread-1")
    print(f"Response 1: {result1['response']}")

    # Test Streaming
    print("\n--- Streaming Agent ---")
    stream_agent = StreamingAgent()
    for event in stream_agent.stream("What is Python?"):
        print(f"Event: {event}")

    print("\n✅ All solutions implemented!")

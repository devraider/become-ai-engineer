"""
Solutions for Week 9 - Project: AI Assistant with Tools and Memory
==================================================================
"""

from typing import Any, Callable, Generator, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import uuid


# =============================================================================
# PART 1: Message Types and Memory
# =============================================================================
@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str
    content: str
    name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            timestamp=datetime.fromisoformat(
                data.get("timestamp", datetime.now().isoformat())
            ),
            metadata=data.get("metadata", {}),
        )


class SmartMemory:
    """Intelligent memory that combines buffer and summary."""

    def __init__(
        self,
        recent_window: int = 10,
        summarizer: Callable[[str], str] = None,
        max_summary_length: int = 500,
    ):
        self.recent_window = recent_window
        self.summarizer = summarizer or (
            lambda t: (
                t[:max_summary_length] + "..." if len(t) > max_summary_length else t
            )
        )
        self.max_summary_length = max_summary_length
        self.messages: list[ChatMessage] = []
        self.summary: str = ""
        self.metadata: dict = {}

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.recent_window * 2:
            self._summarize_old_messages()

    def _summarize_old_messages(self) -> None:
        old_messages = self.messages[: -self.recent_window]
        text = "\n".join(f"{m.role}: {m.content}" for m in old_messages)
        new_summary = self.summarizer(text)
        self.summary = f"{self.summary}\n{new_summary}" if self.summary else new_summary
        self.messages = self.messages[-self.recent_window :]

    def get_context(self) -> dict:
        return {
            "summary": self.summary,
            "recent_messages": list(self.messages),
            "metadata": dict(self.metadata),
        }

    def get_formatted_history(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[Summary]: {self.summary}")
        for m in self.messages:
            parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)

    def search_history(self, query: str) -> list[ChatMessage]:
        query_lower = query.lower()
        return [m for m in self.messages if query_lower in m.content.lower()]

    def save_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""
        self.metadata.clear()


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
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class Tool(ABC):
    """Base class for assistant tools."""

    name: str
    description: str
    parameters: list[ToolParameter] = []

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name, description=self.description, parameters=self.parameters
        )

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def __call__(self, **kwargs) -> str:
        return self.execute(**kwargs)


class CalculatorTool(Tool):
    """Calculator for math operations."""

    name = "calculator"
    description = "Evaluate mathematical expressions."
    parameters = [
        ToolParameter(
            name="expression", description="Mathematical expression to evaluate"
        )
    ]

    def execute(self, expression: str) -> str:
        import math

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
            }
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class WebSearchTool(Tool):
    """Web search simulation."""

    name = "web_search"
    description = "Search the web for information."
    parameters = [
        ToolParameter(name="query", description="Search query"),
        ToolParameter(
            name="num_results",
            description="Number of results",
            type="integer",
            required=False,
            default=3,
        ),
    ]

    def __init__(self, knowledge_base: dict[str, str] = None):
        self.knowledge_base = knowledge_base or {
            "python": "Python is a versatile programming language.",
            "langchain": "LangChain is a framework for LLM applications.",
            "ai": "AI refers to artificial intelligence systems.",
        }

    def execute(self, query: str, num_results: int = 3) -> str:
        results = []
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if query_lower in key.lower() or key.lower() in query_lower:
                results.append(f"- {value}")
            if len(results) >= num_results:
                break
        return "\n".join(results) if results else "No results found."


class DocumentRetrieverTool(Tool):
    """Retrieve relevant documents."""

    name = "retrieve_documents"
    description = "Retrieve relevant documents based on a query."
    parameters = [
        ToolParameter(name="query", description="Query to find relevant documents"),
        ToolParameter(
            name="top_k",
            description="Number of documents",
            type="integer",
            required=False,
            default=3,
        ),
    ]

    def __init__(self, documents: list[dict] = None, embedder=None):
        self.documents = documents or []
        self.embedder = embedder or self._simple_embed
        self.embeddings = (
            [self.embedder(d["content"]) for d in self.documents]
            if self.documents
            else []
        )

    def _simple_embed(self, text: str) -> dict:
        words = text.lower().split()
        return {w: words.count(w) for w in set(words)}

    def _similarity(self, query_emb: dict, doc_emb: dict) -> float:
        common = set(query_emb.keys()) & set(doc_emb.keys())
        if not common:
            return 0.0
        dot = sum(query_emb.get(k, 0) * doc_emb.get(k, 0) for k in common)
        norm1 = sum(v**2 for v in query_emb.values()) ** 0.5
        norm2 = sum(v**2 for v in doc_emb.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def add_document(self, content: str, metadata: dict = None) -> None:
        self.documents.append({"content": content, "metadata": metadata or {}})
        self.embeddings.append(self.embedder(content))

    def execute(self, query: str, top_k: int = 3) -> str:
        if not self.documents:
            return "No documents available."
        query_emb = self.embedder(query)
        scores = [
            (i, self._similarity(query_emb, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [
            self.documents[i]["content"][:200] + "..." for i, _ in scores[:top_k]
        ]
        return "\n\n".join(f"[Doc {i+1}]: {r}" for i, r in enumerate(results))


class ToolManager:
    """Manages available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def get_all_schemas(self) -> list[ToolSchema]:
        return [t.get_schema() for t in self.tools.values()]

    def execute_tool(self, name: str, **kwargs) -> str:
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {name}: {e}"


# =============================================================================
# PART 3: Response Generation
# =============================================================================
class ResponseGenerator:
    """Generates responses using LLM."""

    def __init__(self, llm: Callable[[str], str] = None, system_prompt: str = None):
        self.llm = llm or self._default_llm
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def _default_llm(self, prompt: str) -> str:
        if "calculate" in prompt.lower():
            return "I can help with calculations."
        if "search" in prompt.lower():
            return "Here's what I found from searching."
        return "I'm happy to help with your question."

    def generate(
        self,
        user_input: str,
        context: str = "",
        tool_results: list[tuple[str, str]] = None,
    ) -> str:
        parts = [self.system_prompt]
        if context:
            parts.append(f"Context:\n{context}")
        if tool_results:
            tool_text = "\n".join(
                f"[{name}]: {result}" for name, result in tool_results
            )
            parts.append(f"Tool Results:\n{tool_text}")
        parts.append(f"User: {user_input}")
        prompt = "\n\n".join(parts)
        return self.llm(prompt)

    def stream(self, user_input: str, context: str = "") -> Generator[str, None, None]:
        response = self.generate(user_input, context)
        for word in response.split():
            yield word + " "


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
        self.tool_manager = tool_manager
        self.decision_llm = decision_llm

    def _build_decision_prompt(self, user_input: str) -> str:
        tools = self.tool_manager.list_tools()
        return f"Tools: {tools}\n\nQuery: {user_input}\n\nShould I use a tool?"

    def decide(self, user_input: str, context: str = "") -> ToolDecision:
        input_lower = user_input.lower()

        # Simple heuristics
        if any(
            w in input_lower
            for w in ["calculate", "compute", "math", "+", "-", "*", "/"]
        ):
            return ToolDecision(
                use_tool=True,
                tool_name="calculator",
                tool_args={"expression": user_input},
                reasoning="Math detected",
            )

        if any(w in input_lower for w in ["search", "find", "look up", "what is"]):
            return ToolDecision(
                use_tool=True,
                tool_name="web_search",
                tool_args={"query": user_input},
                reasoning="Search query detected",
            )

        if any(w in input_lower for w in ["document", "retrieve", "context"]):
            return ToolDecision(
                use_tool=True,
                tool_name="retrieve_documents",
                tool_args={"query": user_input},
                reasoning="Document retrieval needed",
            )

        return ToolDecision(use_tool=False, reasoning="No tool needed")


# =============================================================================
# PART 5: Error Handling
# =============================================================================
class FallbackHandler:
    """Handles errors with fallback strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        fallback_response: str = "I apologize, but I'm having trouble.",
    ):
        self.max_retries = max_retries
        self.fallback_response = fallback_response

    def with_retry(self, func: Callable, *args, **kwargs) -> tuple[Any, bool]:
        last_error = None
        for _ in range(self.max_retries):
            try:
                return func(*args, **kwargs), True
            except Exception as e:
                last_error = e
        return self.fallback_response, False

    def with_fallback(
        self, primary: Callable, fallbacks: list[Callable], *args, **kwargs
    ) -> Any:
        try:
            return primary(*args, **kwargs)
        except Exception:
            pass
        for fallback in fallbacks:
            try:
                return fallback(*args, **kwargs)
            except Exception:
                pass
        return self.fallback_response


# =============================================================================
# PART 6: Conversation Manager
# =============================================================================
class ConversationManager:
    """Manages conversation sessions."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}

    def create_session(self, session_id: str = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        self.sessions[session_id] = {
            "memory": SmartMemory(),
            "created": datetime.now(),
            "metadata": {},
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        session["memory"].add_message(ChatMessage(role=role, content=content))
        return True

    def get_history(self, session_id: str) -> list[ChatMessage]:
        session = self.get_session(session_id)
        return session["memory"].messages if session else []

    def end_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def export_session(self, session_id: str) -> Optional[str]:
        session = self.get_session(session_id)
        if not session:
            return None
        data = {
            "session_id": session_id,
            "created": session["created"].isoformat(),
            "messages": [m.to_dict() for m in session["memory"].messages],
            "summary": session["memory"].summary,
        }
        return json.dumps(data, indent=2)


# =============================================================================
# PART 7: Complete AI Assistant
# =============================================================================
class AIAssistant:
    """Complete AI assistant combining all components."""

    def __init__(
        self,
        llm: Callable[[str], str] = None,
        system_prompt: str = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.memory = SmartMemory()
        self.tool_manager = ToolManager()
        self._setup_tools()
        self.response_generator = ResponseGenerator(
            llm=llm, system_prompt=system_prompt
        )
        self.decision_engine = ToolDecisionEngine(self.tool_manager)
        self.fallback_handler = FallbackHandler()
        self.conversation_manager = ConversationManager()
        self.default_session = self.conversation_manager.create_session()

    def _setup_tools(self) -> None:
        self.tool_manager.register(CalculatorTool())
        self.tool_manager.register(WebSearchTool())
        self.retriever = DocumentRetrieverTool()
        self.tool_manager.register(self.retriever)

    def chat(
        self, message: str, session_id: str = None, stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        session_id = session_id or self.default_session

        if not self.conversation_manager.get_session(session_id):
            self.conversation_manager.create_session(session_id)

        self.conversation_manager.add_message(session_id, "user", message)

        context = self.memory.get_formatted_history()

        tool_results = self._execute_tool_if_needed(message, context)

        response = self.response_generator.generate(
            message, context, tool_results[1] if tool_results[0] else None
        )

        self.conversation_manager.add_message(session_id, "assistant", response)
        self.memory.add_message(ChatMessage(role="user", content=message))
        self.memory.add_message(ChatMessage(role="assistant", content=response))

        if stream:
            return (word + " " for word in response.split())
        return response

    def _execute_tool_if_needed(
        self, message: str, context: str
    ) -> tuple[Optional[str], list[tuple[str, str]]]:
        decision = self.decision_engine.decide(message, context)

        if not decision.use_tool:
            return None, []

        if self.verbose:
            print(f"Using tool: {decision.tool_name} with {decision.tool_args}")

        result = self.tool_manager.execute_tool(
            decision.tool_name, **decision.tool_args
        )
        return result, [(decision.tool_name, result)]

    def add_document(self, content: str, metadata: dict = None) -> None:
        self.retriever.add_document(content, metadata)

    def get_available_tools(self) -> list[str]:
        return self.tool_manager.list_tools()

    def clear_memory(self) -> None:
        self.memory.clear()

    def export_conversation(self) -> str:
        return json.dumps(
            {
                "messages": [m.to_dict() for m in self.memory.messages],
                "summary": self.memory.summary,
            },
            indent=2,
        )

    def interactive_chat(self) -> None:
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


if __name__ == "__main__":
    print("=" * 60)
    print("Week 9 Project: AI Assistant with Tools and Memory")
    print("=" * 60)

    def mock_llm(prompt: str) -> str:
        if "calculate" in prompt.lower():
            return "I can help with calculations."
        return "I understand your question and I'm happy to help."

    assistant = AIAssistant(llm=mock_llm, verbose=True)

    print("\n--- Testing Chat ---")
    response = assistant.chat("Hello!")
    print(f"Response: {response}")

    response = assistant.chat("Calculate 2 + 2")
    print(f"Response: {response}")

    print(f"\nAvailable tools: {assistant.get_available_tools()}")

    print("\n✅ Project implementation complete!")

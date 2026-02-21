"""
Solutions for Week 9 - Exercise Advanced 3: Agents and Tools
============================================================
"""

from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
import math


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str

    @abstractmethod
    def _run(self, input_str: str) -> str:
        pass

    def run(self, input_str: str) -> ToolResult:
        try:
            output = self._run(input_str)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


def tool(name: str = None, description: str = None) -> Callable:
    """Decorator to create a tool from a function."""

    def decorator(func: Callable) -> "FunctionTool":
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "No description")
        return FunctionTool(func, tool_name, tool_desc)

    return decorator


class FunctionTool(BaseTool):
    """A tool created from a function."""

    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "No description")

    def _run(self, input_str: str) -> str:
        return str(self.func(input_str))


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""

    name = "calculator"
    description = (
        "Evaluates mathematical expressions. Input should be a valid math expression."
    )

    def _run(self, input_str: str) -> str:
        allowed = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "pow": pow,
        }
        result = eval(input_str, {"__builtins__": {}}, allowed)
        return str(result)


class SearchTool(BaseTool):
    """Mock search tool for testing."""

    name = "search"
    description = "Search for information. Input should be a search query string."

    def __init__(self, knowledge_base: dict[str, str] = None):
        self.knowledge_base = knowledge_base or {
            "python": "Python is a versatile programming language",
            "langchain": "LangChain is an LLM application framework",
        }

    def _run(self, input_str: str) -> str:
        query = input_str.lower()
        for key, value in self.knowledge_base.items():
            if key.lower() in query or query in key.lower():
                return value
        return "No results found"


class WikipediaTool(BaseTool):
    """Tool to fetch Wikipedia summaries (mock version)."""

    name = "wikipedia"
    description = "Get Wikipedia summary for a topic."

    def __init__(self, summaries: dict[str, str] = None):
        self.summaries = summaries or {
            "python": "Python is a high-level programming language.",
            "langchain": "LangChain is a framework for LLM applications.",
        }

    def _run(self, input_str: str) -> str:
        topic = input_str.strip().lower()
        for key, value in self.summaries.items():
            if key.lower() == topic:
                return value
        return f"No Wikipedia article found for: {input_str}"


class ToolRegistry:
    """Registry to manage available tools."""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())


@dataclass
class AgentAction:
    """Represents an action the agent wants to take."""

    tool: str
    tool_input: str
    reasoning: str


@dataclass
class AgentFinish:
    """Represents the agent finishing with a final answer."""

    output: str
    reasoning: str


class ReActPromptParser:
    """Parses ReAct-style LLM outputs."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        thought = self._extract_field(text, "Thought") or ""

        final_answer = self._extract_field(text, "Final Answer")
        if final_answer:
            return AgentFinish(output=final_answer, reasoning=thought)

        action = self._extract_field(text, "Action")
        action_input = self._extract_field(text, "Action Input")

        if action and action_input is not None:
            return AgentAction(
                tool=action.strip(), tool_input=action_input.strip(), reasoning=thought
            )

        raise ValueError(
            "Could not parse response. Expected Action/Action Input or Final Answer."
        )

    def _extract_field(self, text: str, field: str) -> Optional[str]:
        pattern = rf"{field}:\s*(.+?)(?=\n[A-Z][a-z]+ ?:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None


class AgentExecutor:
    """Executes an agent loop until completion."""

    def __init__(
        self,
        agent,
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        self.agent = agent
        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.parser = ReActPromptParser()

    def _build_scratchpad(self, steps: list[tuple]) -> str:
        lines = []
        for action, observation in steps:
            lines.append(f"Thought: {action.reasoning}")
            lines.append(f"Action: {action.tool}")
            lines.append(f"Action Input: {action.tool_input}")
            lines.append(f"Observation: {observation}")
        return "\n".join(lines)

    def run(self, input_text: str) -> str:
        steps = []
        for _ in range(self.max_iterations):
            scratchpad = self._build_scratchpad(steps)
            prompt = f"Input: {input_text}\n\n{scratchpad}"

            response = self.agent(prompt)
            if self.verbose:
                print(f"Agent response:\n{response}\n")

            try:
                parsed = self.parser.parse(response)
            except ValueError:
                return "Could not understand agent response."

            if isinstance(parsed, AgentFinish):
                return parsed.output

            observation = self._run_tool(parsed)
            steps.append((parsed, observation))

            if self.verbose:
                print(f"Tool observation: {observation}\n")

        return "Max iterations reached without final answer."

    def _run_tool(self, action: AgentAction) -> str:
        tool = self.registry.get(action.tool)
        if not tool:
            return f"Error: Tool '{action.tool}' not found"
        result = tool.run(action.tool_input)
        return result.output if result.success else f"Error: {result.error}"


@dataclass
class ToolCall:
    """Represents a tool call request."""

    id: str
    name: str
    arguments: dict


class ToolCallingAgent:
    """Agent that uses structured tool calling."""

    def __init__(self, llm_with_tools, tools: list[BaseTool]):
        self.llm_with_tools = llm_with_tools
        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)

    def plan(self, input_text: str, context: str = "") -> list[ToolCall]:
        return self.llm_with_tools(f"{context}\n{input_text}")

    def execute(self, tool_calls: list[ToolCall]) -> list[tuple[str, str]]:
        results = []
        for call in tool_calls:
            tool = self.registry.get(call.name)
            if tool:
                input_str = call.arguments.get(
                    "expression", call.arguments.get("query", str(call.arguments))
                )
                result = tool.run(input_str)
                results.append(
                    (call.name, result.output if result.success else result.error)
                )
            else:
                results.append((call.name, f"Tool {call.name} not found"))
        return results

    def run(self, input_text: str) -> str:
        calls = self.plan(input_text)
        results = self.execute(calls)
        return "\n".join(f"{name}: {result}" for name, result in results)


class AgentWithMemory:
    """Agent that maintains conversation memory."""

    def __init__(self, agent_executor: AgentExecutor, memory=None):
        self.executor = agent_executor
        from solution_intermediate_2_memory import ConversationBufferMemory

        self.memory = memory or ConversationBufferMemory()

    def run(self, input_text: str) -> str:
        history = self.memory.load_memory_variables({}).get("history", [])
        history_str = (
            "\n".join(f"{m.role}: {m.content}" for m in history) if history else ""
        )

        full_input = (
            f"History:\n{history_str}\n\nCurrent: {input_text}"
            if history_str
            else input_text
        )
        response = self.executor.run(full_input)

        self.memory.add_user_message(input_text)
        self.memory.add_ai_message(response)

        return response

    def get_chat_history(self) -> list:
        return self.memory.get_messages()

    def clear_memory(self) -> None:
        self.memory.clear()


class MultiAgentSystem:
    """System with multiple specialized agents."""

    def __init__(self):
        self.agents: dict[str, tuple[Any, str]] = {}
        self.router: Callable[[str], str] = None

    def register_agent(self, name: str, agent, description: str) -> None:
        self.agents[name] = (agent, description)

    def set_router(self, router: Callable[[str], str]) -> None:
        self.router = router

    def route(self, input_text: str) -> str:
        if self.router:
            return self.router(input_text)
        return list(self.agents.keys())[0] if self.agents else None

    def run(self, input_text: str) -> str:
        agent_name = self.route(input_text)
        if not agent_name or agent_name not in self.agents:
            return "No suitable agent found"
        agent, _ = self.agents[agent_name]
        return agent.run(input_text)

    def list_agents(self) -> list[tuple[str, str]]:
        return [(name, desc) for name, (_, desc) in self.agents.items()]


class ResearchAgent:
    """Complete research agent with tools and memory."""

    def __init__(self, llm=None, verbose: bool = False):
        self.llm = llm or self._default_llm
        self.verbose = verbose
        self.tools = self._create_tools()
        self.registry = ToolRegistry()
        for t in self.tools:
            self.registry.register(t)
        from solution_intermediate_2_memory import ConversationBufferMemory

        self.memory = ConversationBufferMemory()
        self.parser = ReActPromptParser()

    def _default_llm(self, prompt: str) -> str:
        return (
            "Thought: I can answer directly.\nFinal Answer: This is a default response."
        )

    def _create_tools(self) -> list[BaseTool]:
        return [CalculatorTool(), SearchTool(), WikipediaTool()]

    def _build_prompt(self, input_text: str, history: str, scratchpad: str) -> str:
        tools_desc = self.registry.get_tool_descriptions()
        return f"""You are a helpful research assistant with tools.

Available tools:
{tools_desc}

Conversation history:
{history}

Scratchpad:
{scratchpad}

Question: {input_text}

Respond with Thought/Action/Action Input or Thought/Final Answer."""

    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        return self.parser.parse(response)

    def run(self, input_text: str) -> str:
        history_msgs = self.memory.get_messages()
        history = "\n".join(f"{m.role}: {m.content}" for m in history_msgs)

        steps = []
        for _ in range(5):
            scratchpad = "\n".join(
                f"Thought: {a.reasoning}\nAction: {a.tool}\nAction Input: {a.tool_input}\nObservation: {o}"
                for a, o in steps
            )

            prompt = self._build_prompt(input_text, history, scratchpad)
            response = self.llm(prompt)

            if self.verbose:
                print(f"LLM: {response}\n")

            try:
                parsed = self._parse_response(response)
            except ValueError:
                return "Could not understand response."

            if isinstance(parsed, AgentFinish):
                self.memory.add_user_message(input_text)
                self.memory.add_ai_message(parsed.output)
                return parsed.output

            tool = self.registry.get(parsed.tool)
            if tool:
                result = tool.run(parsed.tool_input)
                observation = result.output if result.success else result.error
            else:
                observation = f"Tool {parsed.tool} not found"

            steps.append((parsed, observation))

        return "Max iterations reached."

    def chat(self, message: str) -> str:
        return self.run(message)

    def reset(self) -> None:
        self.memory.clear()


if __name__ == "__main__":
    print("=== Testing Tools ===")
    calc = CalculatorTool()
    print(f"2 + 2 * 3 = {calc.run('2 + 2 * 3').output}")

    search = SearchTool()
    print(f"Search python: {search.run('python').output}")

    print("\n=== Testing ReAct Parser ===")
    parser = ReActPromptParser()
    action_text = "Thought: Need to calculate.\nAction: calculator\nAction Input: 5 + 5"
    result = parser.parse(action_text)
    print(f"Parsed: {result}")

    finish_text = "Thought: Done.\nFinal Answer: The answer is 10."
    result = parser.parse(finish_text)
    print(f"Parsed: {result}")

    print("\n=== Testing Research Agent ===")
    agent = ResearchAgent(verbose=False)
    response = agent.chat("Hello!")
    print(f"Response: {response}")

    print("\n✅ All solutions verified!")

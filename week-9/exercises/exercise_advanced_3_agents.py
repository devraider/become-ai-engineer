"""
Week 9 - Exercise Advanced 3: Agents and Tools
==============================================

Learn to build LangChain agents that use tools autonomously.

Topics covered:
- Tool creation and registration
- ReAct agent pattern
- Tool calling agents
- Agent executors
- Multi-tool agents
- Error handling in agents
"""

from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re


# =============================================================================
# TASK 1: Implement Base Tool Class
# =============================================================================
@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str
    error: Optional[str] = None


class BaseTool(ABC):
    """
    Abstract base class for tools.

    Tools are functions that agents can call.
    """

    name: str
    description: str

    @abstractmethod
    def _run(self, input_str: str) -> str:
        """Execute the tool.

        Args:
            input_str: Input string for the tool

        Returns:
            Tool output as string
        """
        pass

    def run(self, input_str: str) -> ToolResult:
        """Safe wrapper around _run with error handling.

        Args:
            input_str: Input for the tool

        Returns:
            ToolResult with success status and output/error
        """
        # TODO: Try to run _run, catch exceptions
        # Return ToolResult with appropriate values
        pass


# =============================================================================
# TASK 2: Implement Tool Decorator
# =============================================================================
def tool(name: str = None, description: str = None) -> Callable:
    """
    Decorator to create a tool from a function.

    Usage:
        @tool(name="calculator", description="Does math")
        def calculate(expression: str) -> str:
            return str(eval(expression))

    Args:
        name: Tool name (default: function name)
        description: Tool description (default: docstring)

    Returns:
        Decorated function wrapped as a Tool
    """

    def decorator(func: Callable) -> "FunctionTool":
        # TODO: Create and return a FunctionTool
        # Use func name and docstring as defaults
        pass

    return decorator


class FunctionTool(BaseTool):
    """A tool created from a function."""

    def __init__(self, func: Callable, name: str = None, description: str = None):
        """Initialize from function.

        Args:
            func: The function to wrap
            name: Tool name
            description: Tool description
        """
        # TODO: Store function, set name and description
        pass

    def _run(self, input_str: str) -> str:
        """Run the wrapped function.

        Args:
            input_str: Input to pass to function

        Returns:
            Function output as string
        """
        # TODO: Call function and return str result
        pass


# =============================================================================
# TASK 3: Implement Common Tools
# =============================================================================
class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""

    name = "calculator"
    description = "Evaluates mathematical expressions. Input should be a valid math expression like '2 + 2' or '(3 * 4) / 2'."

    def _run(self, input_str: str) -> str:
        """Safely evaluate math expression.

        Args:
            input_str: Math expression

        Returns:
            Result as string
        """
        import math

        # TODO: Use eval with restricted builtins (only math functions)
        # Return result as string
        pass


class SearchTool(BaseTool):
    """Mock search tool for testing."""

    name = "search"
    description = "Search for information. Input should be a search query string."

    def __init__(self, knowledge_base: dict[str, str] = None):
        """Initialize with optional knowledge base.

        Args:
            knowledge_base: Dict mapping queries to answers
        """
        # TODO: Store knowledge base (or use default)
        pass

    def _run(self, input_str: str) -> str:
        """Search the knowledge base.

        Args:
            input_str: Search query

        Returns:
            Best matching result or "Not found"
        """
        # TODO: Find best match in knowledge base
        # Use simple substring matching
        pass


class WikipediaTool(BaseTool):
    """Tool to fetch Wikipedia summaries (mock version)."""

    name = "wikipedia"
    description = "Get Wikipedia summary for a topic. Input should be a topic name."

    def __init__(self, summaries: dict[str, str] = None):
        """Initialize with mock summaries.

        Args:
            summaries: Dict mapping topics to summaries
        """
        # TODO: Store summaries dict
        pass

    def _run(self, input_str: str) -> str:
        """Get summary for topic.

        Args:
            input_str: Topic name

        Returns:
            Wikipedia summary or "No article found"
        """
        # TODO: Look up topic (case-insensitive)
        # Return summary or not found message
        pass


# =============================================================================
# TASK 4: Implement Tool Registry
# =============================================================================
class ToolRegistry:
    """
    Registry to manage available tools.

    Allows adding, removing, and looking up tools.
    """

    def __init__(self):
        """Initialize empty registry."""
        # TODO: Initialize tools dict
        pass

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If tool with same name exists
        """
        # TODO: Add tool to registry by name
        # Raise error if name already exists
        pass

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name to remove

        Returns:
            True if removed, False if not found
        """
        # TODO: Remove tool from registry
        pass

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None if not found
        """
        # TODO: Return tool from registry
        pass

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        # TODO: Return list of tool names
        pass

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools.

        Returns:
            Formatted string describing all tools
        """
        # TODO: Format each tool as "- name: description"
        # Join with newlines
        pass


# =============================================================================
# TASK 5: Implement ReAct Agent Logic
# =============================================================================
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
    """
    Parses ReAct-style LLM outputs.

    ReAct format:
    Thought: I need to search for information
    Action: search
    Action Input: Python programming

    Or to finish:
    Thought: I now have the answer
    Final Answer: Python is a programming language...
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse LLM output into action or finish.

        Args:
            text: LLM output text

        Returns:
            AgentAction or AgentFinish

        Raises:
            ValueError: If format is invalid
        """
        # TODO: Parse the text for Thought, Action, Action Input
        # Or Thought, Final Answer
        # Return appropriate dataclass
        pass

    def _extract_field(self, text: str, field: str) -> Optional[str]:
        """Extract a field value from text.

        Args:
            text: Text to search
            field: Field name (e.g., "Thought")

        Returns:
            Field value or None
        """
        # TODO: Use regex to find "Field: value"
        pass


# =============================================================================
# TASK 6: Implement Agent Executor
# =============================================================================
class AgentExecutor:
    """
    Executes an agent loop until completion.

    Handles the think-act-observe cycle.
    """

    def __init__(
        self,
        agent,  # Function that takes input and returns text
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize executor.

        Args:
            agent: Function that generates agent responses
            tools: List of available tools
            max_iterations: Maximum steps before stopping
            verbose: Print intermediate steps
        """
        # TODO: Store agent, create tool registry, store settings
        pass

    def _build_scratchpad(self, steps: list[tuple]) -> str:
        """Build scratchpad from intermediate steps.

        Args:
            steps: List of (action, observation) tuples

        Returns:
            Formatted scratchpad string
        """
        # TODO: Format each step as:
        # Thought: ...
        # Action: ...
        # Action Input: ...
        # Observation: ...
        pass

    def run(self, input_text: str) -> str:
        """Run the agent until completion.

        Args:
            input_text: User input

        Returns:
            Final answer
        """
        # TODO: Implement the agent loop:
        # 1. Build prompt with input and scratchpad
        # 2. Call agent to get response
        # 3. Parse response
        # 4. If AgentFinish, return output
        # 5. If AgentAction, run tool, add to steps
        # 6. Repeat until finish or max_iterations
        pass

    def _run_tool(self, action: AgentAction) -> str:
        """Run a tool from an agent action.

        Args:
            action: The action to execute

        Returns:
            Tool observation string
        """
        # TODO: Get tool from registry and run it
        # Handle tool not found error
        pass


# =============================================================================
# TASK 7: Implement Tool-Calling Agent
# =============================================================================
@dataclass
class ToolCall:
    """Represents a tool call request."""

    id: str
    name: str
    arguments: dict


class ToolCallingAgent:
    """
    Agent that uses structured tool calling.

    More reliable than text parsing for tool use.
    """

    def __init__(
        self, llm_with_tools, tools: list[BaseTool]  # Function that returns ToolCalls
    ):
        """Initialize agent.

        Args:
            llm_with_tools: Function that takes input, returns ToolCall list
            tools: Available tools
        """
        # TODO: Store LLM and create tool registry
        pass

    def plan(self, input_text: str, context: str = "") -> list[ToolCall]:
        """Plan which tools to call.

        Args:
            input_text: User input
            context: Previous context

        Returns:
            List of tool calls to make
        """
        # TODO: Call LLM to get tool calls
        pass

    def execute(self, tool_calls: list[ToolCall]) -> list[tuple[str, str]]:
        """Execute tool calls.

        Args:
            tool_calls: List of ToolCall objects

        Returns:
            List of (tool_name, result) tuples
        """
        # TODO: Execute each tool call and collect results
        pass

    def run(self, input_text: str) -> str:
        """Run complete agent cycle.

        Args:
            input_text: User input

        Returns:
            Final response
        """
        # TODO: Plan -> Execute -> Generate final response
        pass


# =============================================================================
# TASK 8: Implement Agent with Memory
# =============================================================================
class AgentWithMemory:
    """
    Agent that maintains conversation memory.

    Can reference past interactions.
    """

    def __init__(self, agent_executor: AgentExecutor, memory=None):  # Memory instance
        """Initialize agent with memory.

        Args:
            agent_executor: The agent executor to use
            memory: Memory instance for conversation history
        """
        # TODO: Store executor and memory
        # Import and create default memory if None
        pass

    def run(self, input_text: str) -> str:
        """Run agent with memory context.

        Args:
            input_text: User input

        Returns:
            Agent response
        """
        # TODO: Get history from memory
        # Build input with history context
        # Run agent
        # Save to memory
        # Return response
        pass

    def get_chat_history(self) -> list:
        """Get conversation history.

        Returns:
            List of messages
        """
        # TODO: Return messages from memory
        pass

    def clear_memory(self) -> None:
        """Clear conversation history."""
        # TODO: Clear memory
        pass


# =============================================================================
# TASK 9: Implement Multi-Agent System
# =============================================================================
class MultiAgentSystem:
    """
    System with multiple specialized agents.

    Routes requests to appropriate agents.
    """

    def __init__(self):
        """Initialize multi-agent system."""
        # TODO: Initialize agents dict and router
        pass

    def register_agent(self, name: str, agent: AgentExecutor, description: str) -> None:
        """Register an agent.

        Args:
            name: Agent name
            agent: Agent executor
            description: What this agent does
        """
        # TODO: Store agent with name and description
        pass

    def set_router(self, router: Callable[[str], str]) -> None:
        """Set the routing function.

        Args:
            router: Function that takes input and returns agent name
        """
        # TODO: Store router function
        pass

    def route(self, input_text: str) -> str:
        """Determine which agent to use.

        Args:
            input_text: User input

        Returns:
            Agent name to use
        """
        # TODO: Call router or use default logic
        pass

    def run(self, input_text: str) -> str:
        """Route and run appropriate agent.

        Args:
            input_text: User input

        Returns:
            Agent response
        """
        # TODO: Route to agent, run, return response
        pass

    def list_agents(self) -> list[tuple[str, str]]:
        """List all agents.

        Returns:
            List of (name, description) tuples
        """
        # TODO: Return agent names and descriptions
        pass


# =============================================================================
# TASK 10: Build Complete Research Agent
# =============================================================================
class ResearchAgent:
    """
    A complete research agent that can:
    - Search for information
    - Calculate values
    - Look up Wikipedia
    - Remember conversation context

    This combines all the components.
    """

    def __init__(
        self,
        llm=None,  # Function that takes prompt and returns text
        verbose: bool = False,
    ):
        """Initialize research agent.

        Args:
            llm: LLM function (default: mock)
            verbose: Print debug info
        """
        # TODO: Set up LLM (use mock if None)
        # Create tools: calculator, search, wikipedia
        # Create tool registry
        # Set up memory
        # Store verbose flag
        pass

    def _create_tools(self) -> list[BaseTool]:
        """Create the agent's tools.

        Returns:
            List of tool instances
        """
        # TODO: Create and return calculator, search, wikipedia tools
        pass

    def _build_prompt(self, input_text: str, history: str, scratchpad: str) -> str:
        """Build the complete agent prompt.

        Args:
            input_text: User input
            history: Conversation history
            scratchpad: Intermediate steps

        Returns:
            Complete prompt
        """
        # TODO: Build ReAct-style prompt with:
        # - System instructions
        # - Available tools
        # - History
        # - Current input
        # - Scratchpad
        pass

    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        """Parse LLM response.

        Args:
            response: LLM output

        Returns:
            Action or Finish
        """
        # TODO: Use ReActPromptParser
        pass

    def run(self, input_text: str) -> str:
        """Run the research agent.

        Args:
            input_text: User question

        Returns:
            Final answer
        """
        # TODO: Implement full agent loop:
        # 1. Get history from memory
        # 2. Build prompt with input
        # 3. Loop: call LLM, parse, execute tools
        # 4. Save to memory
        # 5. Return final answer
        pass

    def chat(self, message: str) -> str:
        """Convenience method for chat-style interaction.

        Args:
            message: User message

        Returns:
            Agent response
        """
        # TODO: Just call run()
        pass

    def reset(self) -> None:
        """Reset agent state (clear memory)."""
        # TODO: Clear memory
        pass


# =============================================================================
# Test your implementations
# =============================================================================
if __name__ == "__main__":
    # Test Task 3: Common Tools
    print("=== Testing Tools ===")
    calc = CalculatorTool()
    result = calc.run("2 + 2 * 3")
    print(f"Calculator: 2 + 2 * 3 = {result.output}")

    search = SearchTool(
        {
            "python": "Python is a programming language",
            "langchain": "LangChain is an LLM framework",
        }
    )
    result = search.run("what is python")
    print(f"Search: {result.output}")

    # Test Task 4: Tool Registry
    print("\n=== Testing Tool Registry ===")
    registry = ToolRegistry()
    registry.register(calc)
    registry.register(search)
    print(f"Registered tools: {registry.list_tools()}")
    print(registry.get_tool_descriptions())

    # Test Task 5: ReAct Parser
    print("\n=== Testing ReAct Parser ===")
    parser = ReActPromptParser()

    action_text = """
Thought: I need to search for information about Python.
Action: search
Action Input: python programming
"""
    result = parser.parse(action_text)
    if isinstance(result, AgentAction):
        print(f"Parsed action: {result.tool}({result.tool_input})")

    finish_text = """
Thought: I now know the answer.
Final Answer: Python is a versatile programming language.
"""
    result = parser.parse(finish_text)
    if isinstance(result, AgentFinish):
        print(f"Parsed finish: {result.output[:50]}...")

    # Test Task 10: Research Agent
    print("\n=== Testing Research Agent ===")

    def mock_llm(prompt):
        if "calculator" in prompt.lower() and "2 + 2" in prompt:
            return """
Thought: I need to calculate 2 + 2.
Action: calculator
Action Input: 2 + 2
"""
        return """
Thought: I can answer this directly.
Final Answer: This is a test response.
"""

    agent = ResearchAgent(llm=mock_llm, verbose=True)
    if agent:
        response = agent.run("What is 2 + 2?")
        print(f"Agent response: {response}")

    print("\n✅ Advanced exercises completed!")

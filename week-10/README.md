# Week 10 - LangGraph: Stateful AI Workflows

> Build production-ready agentic applications with graph-based state machines

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph Conceptual Guide](https://langchain-ai.github.io/langgraph/concepts/)
- [LangGraph Examples Repository](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/)

## Installation

```bash
# Core LangGraph
uv add langgraph langgraph-checkpoint

# LangChain integration (from Week 9)
uv add langchain langchain-core langchain-community

# LLM Providers
uv add langchain-google-genai langchain-openai

# For persistence
uv add langgraph-checkpoint-sqlite langgraph-checkpoint-postgres

# Optional: Visualization
uv add pygraphviz matplotlib
```

## Why LangGraph?

While LangChain provides building blocks, **LangGraph** adds:

- **Stateful Execution**: Maintain state across multiple steps
- **Cycles & Loops**: Support for iterative agent patterns
- **Human-in-the-Loop**: Built-in breakpoints and approval flows
- **Persistence**: Save and resume conversations
- **Streaming**: First-class streaming support
- **Debugging**: Visual graph inspection

**LangGraph vs LangChain**:

- LangChain = Building blocks (chains, prompts, tools)
- LangGraph = Orchestration layer (state machines, workflows)

---

## Concepts

### 🔹 Graph Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      LangGraph Components                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   State     │  ◀── TypedDict defining all graph data           │
│   └─────────────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │    Node     │───▶│    Node     │───▶│    Node     │            │
│   │  (function) │    │  (function) │    │  (function) │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│          │                  │                  │                    │
│          └──────────────────┴──────────────────┘                    │
│                             │                                       │
│                             ▼                                       │
│                      ┌─────────────┐                                │
│                      │    Edges    │  ◀── Conditional routing       │
│                      └─────────────┘                                │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Key Concepts**:

- **State**: TypedDict that flows through the graph
- **Nodes**: Functions that transform state
- **Edges**: Define transitions between nodes
- **Conditional Edges**: Dynamic routing based on state
- **Checkpointer**: Persistence layer for state

---

### 🔹 Basic State Definition

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# State with message history (uses reducer for appending)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: str
    context: dict

# Simple state without reducers
class SimpleState(TypedDict):
    input: str
    output: str
    intermediate_steps: list
```

**State Reducers**:

```python
from operator import add

class CounterState(TypedDict):
    # Each update ADDS to the count (reducer pattern)
    count: Annotated[int, add]
    # Each update REPLACES the value (default behavior)
    status: str
```

---

### 🔹 Building Your First Graph

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    processed: str
    output: str

def process_input(state: State) -> dict:
    """First node: process input"""
    return {"processed": state["input"].upper()}

def generate_output(state: State) -> dict:
    """Second node: generate output"""
    return {"output": f"Result: {state['processed']}"}

# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("process", process_input)
builder.add_node("generate", generate_output)

# Add edges (define flow)
builder.add_edge(START, "process")
builder.add_edge("process", "generate")
builder.add_edge("generate", END)

# Compile the graph
graph = builder.compile()

# Run the graph
result = graph.invoke({"input": "hello world"})
print(result)  # {'input': 'hello world', 'processed': 'HELLO WORLD', 'output': 'Result: HELLO WORLD'}
```

---

### 🔹 Conditional Routing

```python
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    query: str
    category: str
    response: str

def classify_query(state: RouterState) -> dict:
    """Classify the query type"""
    query = state["query"].lower()
    if "weather" in query:
        return {"category": "weather"}
    elif "calculate" in query or any(c in query for c in "+-*/"):
        return {"category": "math"}
    else:
        return {"category": "general"}

def handle_weather(state: RouterState) -> dict:
    return {"response": "Checking weather data..."}

def handle_math(state: RouterState) -> dict:
    return {"response": "Computing calculation..."}

def handle_general(state: RouterState) -> dict:
    return {"response": "Processing general query..."}

def route_query(state: RouterState) -> str:
    """Conditional edge function - returns next node name"""
    return state["category"]

# Build graph with conditional routing
builder = StateGraph(RouterState)

builder.add_node("classify", classify_query)
builder.add_node("weather", handle_weather)
builder.add_node("math", handle_math)
builder.add_node("general", handle_general)

builder.add_edge(START, "classify")

# Conditional edge based on classification
builder.add_conditional_edges(
    "classify",
    route_query,
    {
        "weather": "weather",
        "math": "math",
        "general": "general"
    }
)

# All handlers go to END
builder.add_edge("weather", END)
builder.add_edge("math", END)
builder.add_edge("general", END)

graph = builder.compile()
```

---

### 🔹 ReAct Agent Pattern

The ReAct (Reasoning + Acting) pattern is fundamental for tool-using agents:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Check if agent should continue or stop"""
    last_message = state["messages"][-1]
    # If LLM made tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end
    return "end"

def call_model(state: AgentState) -> dict:
    """Call the LLM"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define tools
@tool
def search(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate math expression"""
    return str(eval(expression))

tools = [search, calculator]
llm_with_tools = llm.bind_tools(tools)

# Build ReAct graph
builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
builder.add_edge("tools", "agent")  # Loop back after tool execution

graph = builder.compile()
```

---

### 🔹 Persistence with Checkpointers

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory checkpointer (for development)
memory_checkpointer = MemorySaver()

# SQLite checkpointer (for persistence)
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Compile graph with checkpointer
graph = builder.compile(checkpointer=memory_checkpointer)

# Run with thread_id for conversation persistence
config = {"configurable": {"thread_id": "user-123"}}

# First message
result1 = graph.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Alice")]},
    config
)

# Continue same conversation
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config
)
# Agent remembers: "Your name is Alice"

# Get conversation history
history = graph.get_state(config)
```

---

### 🔹 Human-in-the-Loop

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    request: str
    approved: bool | None
    result: str

def process_request(state: ApprovalState) -> dict:
    return {"request": f"Processed: {state['request']}"}

def execute_action(state: ApprovalState) -> dict:
    if state.get("approved"):
        return {"result": "Action executed successfully!"}
    return {"result": "Action was rejected."}

def check_approval(state: ApprovalState) -> str:
    if state.get("approved") is None:
        return "wait"  # Interrupt here
    return "execute"

builder = StateGraph(ApprovalState)
builder.add_node("process", process_request)
builder.add_node("execute", execute_action)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    check_approval,
    {"wait": END, "execute": "execute"}  # END = interrupt point
)
builder.add_edge("execute", END)

# Compile with interrupt_before for human approval
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"]  # Pause before execution
)

config = {"configurable": {"thread_id": "approval-1"}}

# Start the workflow
result = graph.invoke({"request": "Delete all files"}, config)
# Workflow pauses at "execute" node

# Human reviews and approves
graph.update_state(config, {"approved": True})

# Resume execution
final_result = graph.invoke(None, config)
```

---

### 🔹 Streaming

```python
# Stream all events
for event in graph.stream({"messages": [HumanMessage("Hello")]}):
    print(event)

# Stream specific modes
for event in graph.stream(
    {"messages": [HumanMessage("Hello")]},
    stream_mode="values"  # or "updates", "debug"
):
    print(event)

# Async streaming
async for event in graph.astream({"messages": [HumanMessage("Hello")]}):
    print(event)

# Stream tokens from LLM
async for event in graph.astream_events(
    {"messages": [HumanMessage("Hello")]},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

---

### 🔹 Subgraphs (Nested Graphs)

```python
# Define a subgraph for research
class ResearchState(TypedDict):
    topic: str
    findings: list[str]

def search_web(state: ResearchState) -> dict:
    return {"findings": [f"Web result for {state['topic']}"]}

def search_papers(state: ResearchState) -> dict:
    return {"findings": state["findings"] + [f"Paper about {state['topic']}"]}

research_builder = StateGraph(ResearchState)
research_builder.add_node("web", search_web)
research_builder.add_node("papers", search_papers)
research_builder.add_edge(START, "web")
research_builder.add_edge("web", "papers")
research_builder.add_edge("papers", END)

research_graph = research_builder.compile()

# Main graph uses subgraph as a node
class MainState(TypedDict):
    query: str
    research: ResearchState
    final_answer: str

def prepare_research(state: MainState) -> dict:
    return {"research": {"topic": state["query"], "findings": []}}

def run_research(state: MainState) -> dict:
    result = research_graph.invoke(state["research"])
    return {"research": result}

def synthesize(state: MainState) -> dict:
    findings = state["research"]["findings"]
    return {"final_answer": f"Based on: {findings}"}

main_builder = StateGraph(MainState)
main_builder.add_node("prepare", prepare_research)
main_builder.add_node("research", run_research)
main_builder.add_node("synthesize", synthesize)

main_builder.add_edge(START, "prepare")
main_builder.add_edge("prepare", "research")
main_builder.add_edge("research", "synthesize")
main_builder.add_edge("synthesize", END)

main_graph = main_builder.compile()
```

---

### 🔹 Error Handling

```python
from langgraph.errors import GraphRecursionError

class RobustState(TypedDict):
    input: str
    error: str | None
    retries: int
    output: str

def risky_operation(state: RobustState) -> dict:
    try:
        # Simulated operation that might fail
        if state["retries"] < 2:
            raise ValueError("Temporary failure")
        return {"output": "Success!", "error": None}
    except Exception as e:
        return {"error": str(e), "retries": state["retries"] + 1}

def should_retry(state: RobustState) -> str:
    if state.get("error") and state["retries"] < 3:
        return "retry"
    elif state.get("error"):
        return "failed"
    return "success"

builder = StateGraph(RobustState)
builder.add_node("operation", risky_operation)
builder.add_node("handle_failure", lambda s: {"output": "Failed after retries"})

builder.add_edge(START, "operation")
builder.add_conditional_edges(
    "operation",
    should_retry,
    {
        "retry": "operation",  # Loop back
        "failed": "handle_failure",
        "success": END
    }
)
builder.add_edge("handle_failure", END)

# Set recursion limit to prevent infinite loops
graph = builder.compile()
result = graph.invoke(
    {"input": "test", "retries": 0},
    {"recursion_limit": 10}
)
```

---

### 🔹 Prebuilt Components

LangGraph provides prebuilt agents:

```python
from langgraph.prebuilt import create_react_agent

# Quick ReAct agent creation
tools = [search_tool, calculator_tool]
agent = create_react_agent(llm, tools)

# With system message
agent = create_react_agent(
    llm,
    tools,
    state_modifier="You are a helpful research assistant."
)

# Run the agent
result = agent.invoke({
    "messages": [HumanMessage("What's the weather in Paris?")]
})
```

---

## Exercises

### Exercise 1 - Basic Graph Building

**File**: `exercises/exercise_basic_1_graphs.py`

Build fundamental graph structures:

- Create state definitions with TypedDict
- Build sequential node pipelines
- Handle basic state transformations
- Compile and run simple graphs

### Exercise 2 - Conditional Routing & Cycles

**File**: `exercises/exercise_intermediate_2_routing.py`

Implement advanced flow control:

- Create conditional edges with routing functions
- Build graphs with loops and cycles
- Implement retry patterns
- Handle multiple routing paths

### Exercise 3 - Agents & Persistence

**File**: `exercises/exercise_advanced_3_agents.py`

Build production-ready agents:

- Implement ReAct agent pattern
- Add tool calling capabilities
- Configure checkpointers for persistence
- Implement human-in-the-loop workflows

---

## Weekly Project

**File**: `exercises/project_pipeline.py`

### Research Assistant with Multi-Step Reasoning

Build a comprehensive research assistant that:

1. **Query Analysis**: Parse and classify user queries
2. **Research Planning**: Create multi-step research plan
3. **Parallel Search**: Execute searches in parallel
4. **Source Evaluation**: Rate source credibility
5. **Synthesis**: Combine findings into coherent response
6. **Citation**: Track and format sources
7. **Human Review**: Allow user to approve final output

The assistant should:

- Use LangGraph for orchestration
- Implement checkpointing for long research tasks
- Support streaming responses
- Handle errors gracefully with retries
- Allow human intervention at key decision points

---

## Interview Questions

1. **What is the difference between LangChain and LangGraph?**
   - LangChain: Building blocks (chains, prompts, tools)
   - LangGraph: Orchestration with state machines, cycles, persistence

2. **Explain the role of State in LangGraph.**
   - TypedDict that defines all data flowing through graph
   - Passed to each node, updated by return values
   - Can use reducers (like `add_messages`) for accumulation

3. **What are conditional edges and when would you use them?**
   - Edges that route to different nodes based on state
   - Used for decision points, classification, error handling
   - Enable dynamic workflows based on runtime conditions

4. **How does persistence work in LangGraph?**
   - Checkpointers save state at each step
   - Thread IDs identify conversations
   - Enables resume, human-in-the-loop, and debugging

5. **Describe the ReAct agent pattern.**
   - Reasoning + Acting: Think, Act, Observe loop
   - Agent decides to call tools or respond
   - Tool results feed back into reasoning
   - Continues until task complete

6. **How do you implement human-in-the-loop in LangGraph?**
   - Use `interrupt_before` or `interrupt_after` in compile
   - Graph pauses at specified nodes
   - Human reviews/modifies state
   - Resume with `graph.invoke(None, config)`

7. **What are state reducers and why are they useful?**
   - Functions that combine state updates
   - `add_messages`: Appends to message list
   - `add` operator: Accumulates values
   - Useful for chat history, counters, lists

8. **How would you handle errors in a LangGraph workflow?**
   - Conditional edges for error routing
   - Retry loops with counter in state
   - Fallback nodes for graceful degradation
   - `recursion_limit` to prevent infinite loops

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Define state schemas with TypedDict and reducers
- [ ] Build sequential and parallel graph workflows
- [ ] Implement conditional routing and cycles
- [ ] Create ReAct agents with tool calling
- [ ] Configure persistence with checkpointers
- [ ] Add human-in-the-loop breakpoints
- [ ] Stream graph execution events
- [ ] Handle errors with retry patterns
- [ ] Use prebuilt LangGraph components
- [ ] Debug graphs with visualization

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 11](../week-11/README.md)**

# Week 14 - Agent Systems with Google ADK

## 📚 References

- [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)
- [ADK GitHub Repository](https://github.com/google/adk-python)
- [ADK Quickstart Guide](https://google.github.io/adk-docs/get-started/quickstart/)
- [LangGraph Agents](https://langchain-ai.github.io/langgraph/)
- [ReAct Pattern Paper](https://arxiv.org/abs/2210.03629)
- [Tool Calling Best Practices](https://platform.openai.com/docs/guides/function-calling)

## 🛠️ Installation

```bash
# Google ADK for agent development
uv add google-adk

# For LLM integration
uv add google-generativeai
uv add openai
uv add anthropic

# For agent tools and utilities
uv add httpx
uv add pydantic

# For testing
uv add pytest pytest-asyncio
```

## 🎯 Learning Objectives

By the end of this week, you will:

- Understand agent architectures and design patterns
- Build agents using Google ADK
- Implement tools and function calling
- Create multi-agent systems with coordination
- Handle agent state, memory, and context
- Build production-ready agent pipelines

---

## 📖 Concepts

### 1. Agent Fundamentals

Agents are AI systems that can:

- **Reason** about tasks and break them into steps
- **Act** by calling tools and external APIs
- **Observe** results and adjust their approach
- **Learn** from feedback and improve over time

#### The ReAct Pattern

```
Thought → Action → Observation → Thought → ...
```

```python
# ReAct loop pseudocode
while not task_complete:
    thought = llm.think(context)      # Reasoning
    action = select_action(thought)    # Decision
    result = execute_action(action)    # Execution
    context.update(result)             # Learning
```

#### Agent vs Chatbot

| Feature  | Chatbot | Agent     |
| -------- | ------- | --------- |
| Tools    | No      | Yes       |
| Planning | Limited | Yes       |
| Memory   | Session | Long-term |
| Autonomy | Low     | High      |

### 2. Google ADK Architecture

Google ADK provides a structured framework for building agents:

```python
from google.adk import Agent, Tool

# Define a tool
@Tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Create an agent
agent = Agent(
    name="research-agent",
    model="gemini-2.0-flash",
    tools=[search_web],
    instructions="You are a helpful research assistant."
)

# Run the agent
response = agent.run("Find information about Python")
```

#### ADK Components

1. **Agent**: The main orchestrator
2. **Tools**: Functions the agent can call
3. **Memory**: Context and conversation history
4. **Runners**: Execution environments

### 3. Tool Design

Tools are the building blocks of agent capabilities:

```python
from google.adk import Tool
from pydantic import BaseModel, Field

# Tool with typed parameters
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum results to return")

@Tool
def search(params: SearchParams) -> list[dict]:
    """Search for documents matching the query."""
    # Implementation
    return results

# Tool with error handling
@Tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

#### Tool Categories

| Category          | Examples                 |
| ----------------- | ------------------------ |
| **Information**   | Search, lookup, retrieve |
| **Action**        | Send email, create file  |
| **Computation**   | Calculate, analyze       |
| **Communication** | API calls, notifications |

### 4. Agent Memory and State

```python
# Short-term memory (conversation context)
class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

# Long-term memory (vector store)
class SemanticMemory:
    def __init__(self, vector_store):
        self.store = vector_store

    def remember(self, content: str, metadata: dict):
        embedding = embed(content)
        self.store.add(content, embedding, metadata)

    def recall(self, query: str, k: int = 5) -> list[str]:
        results = self.store.search(query, k)
        return [r.content for r in results]
```

### 5. Multi-Agent Systems

```python
# Coordinator pattern
class AgentTeam:
    def __init__(self):
        self.researcher = Agent(name="researcher", ...)
        self.writer = Agent(name="writer", ...)
        self.reviewer = Agent(name="reviewer", ...)

    async def process(self, task: str) -> str:
        # Research phase
        research = await self.researcher.run(task)

        # Writing phase
        draft = await self.writer.run(f"Write about: {research}")

        # Review phase
        final = await self.reviewer.run(f"Review: {draft}")

        return final
```

#### Multi-Agent Patterns

1. **Sequential**: Agents work in order
2. **Parallel**: Agents work simultaneously
3. **Hierarchical**: Manager delegates to workers
4. **Debate**: Agents discuss and refine

---

## 🏋️ Exercises

### Exercise 1: Agent Fundamentals (Basic)

**File:** `exercises/exercise_basic_1_agent_fundamentals.py`

Build foundational agent components:

- `AgentMessage` - Message model with role and content
- `ToolResult` - Result from tool execution
- `AgentContext` - Conversation context management
- `Tool` - Tool definition and execution
- `ToolRegistry` - Register and manage tools
- `AgentState` - Agent state machine
- `SimpleAgent` - Basic agent with tool calling
- `AgentMemory` - Short-term memory management
- `AgentLogger` - Agent action logging
- `AgentExecutor` - Run agents with retries

### Exercise 2: Google ADK Integration (Intermediate)

**File:** `exercises/exercise_intermediate_2_adk_integration.py`

Integrate with Google ADK:

- `ADKToolWrapper` - Wrap functions as ADK tools
- `ADKAgentConfig` - Configuration for ADK agents
- `ADKAgent` - Agent using Google ADK
- `ToolSchemaGenerator` - Generate tool schemas
- `ADKMemory` - Memory integration with ADK
- `ADKRunner` - Runner for ADK agents
- `StreamingHandler` - Handle streaming responses
- `ADKToolkit` - Collection of useful tools
- `AgentMonitor` - Monitor agent behavior
- `ADKAgentFactory` - Factory for creating agents

### Exercise 3: Multi-Agent Systems (Advanced)

**File:** `exercises/exercise_advanced_3_multi_agent.py`

Build sophisticated multi-agent systems:

- `AgentRole` - Define agent roles and capabilities
- `AgentMessage` - Inter-agent communication
- `MessageBroker` - Route messages between agents
- `AgentCoordinator` - Coordinate agent activities
- `WorkflowStep` - Define workflow steps
- `WorkflowEngine` - Execute multi-step workflows
- `AgentPool` - Pool of reusable agents
- `ConflictResolver` - Handle agent disagreements
- `AgentSupervisor` - Monitor and manage agents
- `MultiAgentSystem` - Complete multi-agent orchestration

---

## 🚀 Weekly Project: Research Assistant Agent System

**File:** `exercises/project_pipeline.py`

Build a complete research assistant that can:

1. **Accept Research Queries**
   - Parse user questions
   - Identify research topics
   - Plan research approach

2. **Search and Gather Information**
   - Web search integration
   - Document retrieval
   - API calls for data

3. **Process and Analyze**
   - Summarize findings
   - Extract key points
   - Identify gaps

4. **Generate Reports**
   - Structured output
   - Citations and references
   - Multiple formats

### Project Components:

```
project_pipeline.py
├── Models
│   ├── ResearchQuery
│   ├── SearchResult
│   ├── ResearchFinding
│   └── ResearchReport
├── Tools
│   ├── WebSearchTool
│   ├── DocumentReaderTool
│   ├── SummarizerTool
│   └── CitationTool
├── Agents
│   ├── PlannerAgent
│   ├── ResearcherAgent
│   ├── AnalyzerAgent
│   └── WriterAgent
├── Pipeline
│   ├── ResearchPipeline
│   ├── QualityChecker
│   └── OutputFormatter
└── System
    ├── ResearchAssistant
    └── ResearchConfig
```

---

## 🎤 Interview Questions

1. **What is the difference between an agent and a traditional chatbot?**
   - Agents can use tools, plan multi-step tasks, and act autonomously
   - Chatbots typically respond based on input without external actions

2. **Explain the ReAct pattern for agents.**
   - Reasoning + Acting: Think, Act, Observe, Repeat
   - Interleaves reasoning (chain-of-thought) with tool use
   - Enables complex problem-solving through iterative refinement

3. **What are the key considerations when designing agent tools?**
   - Clear, descriptive function signatures
   - Proper error handling and validation
   - Idempotency for safe retries
   - Rate limiting and resource management

4. **How do you handle agent memory and context?**
   - Short-term: Recent conversation history
   - Long-term: Vector store for semantic retrieval
   - Working memory: Current task state

5. **What patterns exist for multi-agent coordination?**
   - Sequential (pipeline)
   - Parallel (concurrent execution)
   - Hierarchical (manager with workers)
   - Debate/consensus (multiple perspectives)

6. **How do you prevent infinite loops in agents?**
   - Maximum iteration limits
   - Token budget constraints
   - Timeout mechanisms
   - Loop detection in action history

7. **What is function calling and how does it work?**
   - LLM generates structured tool calls
   - System executes tools and returns results
   - LLM continues with tool outputs

8. **How do you evaluate agent performance?**
   - Task completion rate
   - Tool efficiency (calls per task)
   - Response quality and accuracy
   - Latency and cost metrics

---

## ✅ Takeaway Checklist

After completing this week, you should be able to:

- [ ] Explain agent architectures and the ReAct pattern
- [ ] Design and implement effective agent tools
- [ ] Build agents using Google ADK
- [ ] Implement agent memory (short and long-term)
- [ ] Create multi-agent systems with coordination
- [ ] Handle errors and edge cases in agents
- [ ] Monitor and debug agent behavior
- [ ] Build production-ready agent pipelines

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[← Back to Week 13](../week-13/README.md)**

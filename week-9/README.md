# Week 9 - LangChain: Building AI Applications

> From building blocks to production-ready AI applications

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [LangSmith (Tracing & Debugging)](https://docs.smith.langchain.com/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangGraph for Agents](https://langchain-ai.github.io/langgraph/)

## Installation

```bash
# Core LangChain
pip install langchain langchain-core langchain-community

# LLM Providers
pip install langchain-google-genai  # For Gemini
pip install langchain-openai        # For OpenAI

# Vector stores and tools
pip install chromadb faiss-cpu
pip install wikipedia duckduckgo-search

# Optional: LangSmith for tracing
pip install langsmith
```

## Why LangChain?

LangChain provides **composable building blocks** for AI applications:

- **Chains**: Sequence operations together
- **Prompts**: Reusable prompt templates
- **Memory**: Conversation history management
- **Tools**: Connect to external services
- **Agents**: Autonomous decision-making
- **Retrieval**: Built-in RAG support

---

## Concepts

### 🔹 LangChain Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      LangChain Components                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│   │ Models  │  │ Prompts │  │ Chains  │  │ Memory  │              │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘              │
│        │            │            │            │                    │
│        ▼            ▼            ▼            ▼                    │
│   ┌─────────────────────────────────────────────────────┐         │
│   │            LangChain Expression Language (LCEL)      │         │
│   └─────────────────────────────────────────────────────┘         │
│        │            │            │            │                    │
│        ▼            ▼            ▼            ▼                    │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│   │ Agents  │  │  Tools  │  │Retrieval│  │ Output  │              │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

### 🔹 LangChain Expression Language (LCEL)

LCEL uses the pipe operator `|` to compose components:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Create chain using LCEL
chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({"question": "What is LangChain?"})
print(response)
```

**Key Concepts**:

- `|` operator: Pipes output from one component to the next
- `invoke()`: Run the chain with a single input
- `batch()`: Run with multiple inputs
- `stream()`: Stream the output
- `ainvoke()`: Async execution

📝 **Exercise: Basic 1** - Implement LCEL chains in [exercises/exercise_basic_1_chains.py](exercises/exercise_basic_1_chains.py)

---

### 🔹 Prompt Templates

LangChain provides flexible prompt management:

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)

# Simple template
simple = ChatPromptTemplate.from_template(
    "Translate '{text}' to {language}"
)

# Multi-message template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who speaks in {style}."),
    ("human", "{input}")
])

# Render the prompt
messages = chat_prompt.format_messages(
    role="pirate",
    style="pirate speak",
    input="Tell me about Python"
)

# Few-shot template
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "5*3", "output": "15"}
]

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ]),
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a calculator."),
    few_shot,
    ("human", "{input}")
])
```

**Partial Templates**:

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Partial with value
template = PromptTemplate.from_template(
    "Today is {date}. {question}"
)
partial = template.partial(date=datetime.now().strftime("%Y-%m-%d"))

# Partial with function
def get_date():
    return datetime.now().strftime("%Y-%m-%d")

partial = template.partial(date=get_date)
```

---

### 🔹 Output Parsers

Parse LLM output into structured formats:

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from pydantic import BaseModel, Field

# String parser (default)
str_parser = StrOutputParser()

# JSON parser
json_parser = JsonOutputParser()

# List parser
list_parser = CommaSeparatedListOutputParser()

# Pydantic parser for structured output
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's job")

pydantic_parser = PydanticOutputParser(pydantic_object=Person)

# Format instructions for the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract information. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=pydantic_parser.get_format_instructions())

# Chain with parser
chain = prompt | model | pydantic_parser
person = chain.invoke({"text": "John is 30 years old and works as a doctor"})
print(person.name)  # "John"
```

---

### 🔹 Memory: Conversation History

Manage conversation context:

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_core.messages import HumanMessage, AIMessage

# Buffer memory (stores all messages)
buffer_memory = ConversationBufferMemory(return_messages=True)
buffer_memory.save_context(
    {"input": "Hi there!"},
    {"output": "Hello! How can I help you?"}
)

# Window memory (keeps last k exchanges)
window_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Summary memory (summarizes old conversations)
summary_memory = ConversationSummaryMemory(llm=model)

# Access conversation history
history = buffer_memory.load_memory_variables({})
print(history["history"])
```

**Using Memory in Chains**:

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Prompt with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Chain that includes memory
def get_history(input_dict):
    return memory.load_memory_variables({})["history"]

chain = (
    RunnablePassthrough.assign(history=get_history)
    | prompt
    | model
    | StrOutputParser()
)
```

📝 **Exercise: Intermediate 2** - Implement memory systems in [exercises/exercise_intermediate_2_memory.py](exercises/exercise_intermediate_2_memory.py)

---

### 🔹 Tools: Connecting to External Services

LangChain tools let your LLM interact with the world:

```python
from langchain_core.tools import tool, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Optional

# Simple tool using decorator
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression. Returns the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Built-in search tool
search = DuckDuckGoSearchRun()

# Custom tool class
class WeatherTool(Tool):
    name = "weather"
    description = "Get weather for a location"

    def _run(self, location: str) -> str:
        # In practice, call a weather API
        return f"Weather in {location}: Sunny, 72°F"

# Use tools
result = calculate.invoke("2 + 2 * 3")
print(result)  # "8"
```

**Tool with Pydantic Schema**:

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum results")

@tool(args_schema=SearchInput)
def search_documents(query: str, max_results: int = 5) -> str:
    """Search documents based on a query."""
    # Implementation here
    return f"Found {max_results} results for '{query}'"
```

---

### 🔹 Agents: Autonomous Decision Making

Agents decide which tools to use and how to use them:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Define tools
tools = [calculate, search]

# Agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.
    Use tools when needed to answer questions accurately."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_tool_calling_agent(model, tools, prompt)

# Agent executor runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See the agent's reasoning
    max_iterations=5
)

# Run the agent
response = agent_executor.invoke({
    "input": "What is 25 * 48? Then search for information about that number."
})
print(response["output"])
```

**Agent Types**:

| Type             | Description                      | Best For                        |
| ---------------- | -------------------------------- | ------------------------------- |
| Tool Calling     | Uses model's native tool calling | Modern models with tool support |
| ReAct            | Reason + Act pattern             | General purpose                 |
| OpenAI Functions | Uses OpenAI function calling     | OpenAI models                   |
| Structured Chat  | Structured output format         | Complex tool inputs             |

---

### 🔹 Retrieval: RAG with LangChain

LangChain simplifies RAG implementation:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
Answer based on the context:

Context: {context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# Query
answer = rag_chain.invoke("What is the main topic?")
```

📝 **Exercise: Advanced 3** - Implement agents and RAG in [exercises/exercise_advanced_3_agents.py](exercises/exercise_advanced_3_agents.py)

---

### 🔹 Callbacks and Tracing

Monitor and debug your chains:

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class LoggingHandler(BaseCallbackHandler):
    """Custom callback handler for logging."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        print(f"LLM finished: {response.generations[0][0].text[:100]}...")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with: {inputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {serialized.get('name')} with {input_str}")

# Use callbacks
handler = LoggingHandler()
response = chain.invoke(
    {"question": "Hello"},
    config={"callbacks": [handler]}
)
```

**LangSmith for Production**:

```python
import os

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All chain runs are now traced automatically
```

---

### 🔹 Error Handling and Fallbacks

Build robust applications:

```python
from langchain_core.runnables import RunnableLambda

# Fallback chain
main_chain = prompt | model_a | StrOutputParser()
backup_chain = prompt | model_b | StrOutputParser()

chain_with_fallback = main_chain.with_fallbacks([backup_chain])

# Retry logic
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

# Custom error handling
def handle_error(error):
    return f"An error occurred: {str(error)}"

safe_chain = chain.with_fallbacks(
    [RunnableLambda(lambda x: handle_error(x))]
)
```

---

## 🛠️ Weekly Project

Build an **AI Assistant with Tools and Memory** that can:

1. Maintain conversation history
2. Use multiple tools (search, calculate, etc.)
3. Retrieve information from documents
4. Stream responses
5. Handle errors gracefully

See [exercises/project_pipeline.py](exercises/project_pipeline.py) for the complete project.

---

## Interview Questions

1. **What is LCEL and why is it useful?**
   - LCEL (LangChain Expression Language) provides a declarative way to compose chains using the pipe operator. It enables streaming, async, batch processing, and easy debugging.

2. **How does memory work in LangChain?**
   - Memory stores conversation history and injects it into prompts. Types include buffer (all messages), window (last k), and summary (condensed history).

3. **Explain the difference between chains and agents.**
   - Chains follow a fixed sequence of operations. Agents make dynamic decisions about which tools to use based on the input and intermediate results.

4. **How would you implement RAG with LangChain?**
   - Use a text splitter to chunk documents, embeddings model for vectors, vector store for storage, and a retrieval chain that combines retrieval with generation.

5. **What are LangChain tools and how do you create custom ones?**
   - Tools are functions that agents can call. Create custom tools using the `@tool` decorator with a docstring describing when to use it.

6. **How do you handle streaming responses?**
   - Use `chain.stream()` to get an iterator of response chunks. The chain must support streaming (most LLMs do).

7. **What strategies exist for managing long conversations?**
   - Window memory (keep last k), summary memory (summarize old messages), or custom logic to select relevant history.

8. **How do you debug LangChain applications?**
   - Use verbose mode, callbacks for logging, LangSmith for tracing, or add intermediate logging steps to chains.

9. **When would you use LangGraph over basic LangChain?**
   - LangGraph is better for complex agent workflows with cycles, conditional branching, and state management.

10. **How do you optimize LangChain applications for production?**
    - Use caching, batch operations, async execution, model fallbacks, retry logic, and proper error handling.

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Build chains using LCEL (pipe operator)
- [ ] Create and use prompt templates
- [ ] Parse LLM outputs into structured formats
- [ ] Implement conversation memory
- [ ] Create custom tools
- [ ] Build agents with tool use
- [ ] Implement RAG with LangChain
- [ ] Use callbacks for logging and monitoring
- [ ] Handle errors with fallbacks and retries
- [ ] Understand when to use LangGraph

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 10](../week-10/README.md)**

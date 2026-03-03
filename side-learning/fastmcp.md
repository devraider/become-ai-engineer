# FastMCP - High-Level MCP Framework

## Overview

**FastMCP** is a high-level Python framework that simplifies building MCP (Model Context Protocol) servers. Think of it as "FastAPI for MCP" - it provides decorators and abstractions that let you focus on your logic rather than protocol details.

## Installation

```bash
uv add fastmcp
```

## Why FastMCP?

### Without FastMCP (Low-level MCP SDK)

```python
from mcp.server import Server
from mcp.types import Tool, CallToolResult, TextContent
import json

server = Server("calculator")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "add":
        result = arguments["a"] + arguments["b"]
        return CallToolResult(
            content=[TextContent(type="text", text=str(result))]
        )
    raise ValueError(f"Unknown tool: {name}")
```

### With FastMCP

```python
from fastmcp import FastMCP

mcp = FastMCP("calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b
```

FastMCP automatically:

- Generates JSON schema from type hints
- Extracts description from docstring
- Handles serialization/deserialization
- Manages error responses

---

## Core Features

### 1. Creating a Server

```python
from fastmcp import FastMCP

# Basic server
mcp = FastMCP("my-server")

# With configuration
mcp = FastMCP(
    name="my-server",
    version="1.0.0",
    description="My awesome MCP server"
)
```

### 2. Defining Tools

Tools are functions the AI can call:

```python
from fastmcp import FastMCP

mcp = FastMCP("tools-demo")

# Simple tool - sync function
@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

# Async tool
@mcp.tool()
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

# Tool with optional parameters
@mcp.tool()
def calculate(
    a: float,
    b: float,
    operation: str = "add"
) -> float:
    """
    Perform arithmetic operation.

    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide'
    """
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf")
    }
    return ops.get(operation, ops["add"])

# Tool with complex types
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserInput:
    name: str
    age: int
    email: Optional[str] = None

@mcp.tool()
def process_user(user: UserInput) -> str:
    """Process user data."""
    return f"Processed: {user.name}, age {user.age}"
```

### 3. Defining Resources

Resources are data the AI can read:

```python
from fastmcp import FastMCP

mcp = FastMCP("resources-demo")

# Static resource
@mcp.resource("config://app")
def get_app_config() -> str:
    """Application configuration."""
    return '{"version": "1.0", "debug": true}'

# Dynamic resource with parameter
@mcp.resource("user://{user_id}")
def get_user(user_id: str) -> str:
    """Get user by ID."""
    return f'{{"id": "{user_id}", "name": "User {user_id}"}}'

# File resource
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """Read a file."""
    with open(path, "r") as f:
        return f.read()

# Resource with MIME type
@mcp.resource("data://image/{name}", mime_type="image/png")
def get_image(name: str) -> bytes:
    """Get image data."""
    with open(f"images/{name}.png", "rb") as f:
        return f.read()
```

### 4. Defining Prompts

Prompts are conversation templates:

```python
from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP("prompts-demo")

# Simple prompt
@mcp.prompt()
def code_review() -> list[Message]:
    """Template for code review requests."""
    return [
        Message(
            role="user",
            content="Please review the following code for bugs, style issues, and potential improvements."
        )
    ]

# Prompt with parameters
@mcp.prompt()
def analyze_error(error_message: str, stack_trace: str) -> list[Message]:
    """Analyze an error with context."""
    return [
        Message(
            role="user",
            content=f"""Please analyze this error:

Error: {error_message}

Stack Trace:
{stack_trace}

Explain what went wrong and how to fix it."""
        )
    ]

# Multi-turn prompt
@mcp.prompt()
def interview_prep(role: str, company: str) -> list[Message]:
    """Prepare for a technical interview."""
    return [
        Message(
            role="user",
            content=f"I'm preparing for a {role} interview at {company}."
        ),
        Message(
            role="assistant",
            content="I'll help you prepare. Let's start with common questions for this role."
        ),
        Message(
            role="user",
            content="What technical questions should I expect?"
        )
    ]
```

---

## Advanced Features

### 5. Context Object

Access request context in your handlers:

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("context-demo")

@mcp.tool()
async def process_with_context(data: str, ctx: Context) -> str:
    """Process data with context access."""

    # Log messages
    await ctx.debug("Starting processing...")
    await ctx.info("Processing data")
    await ctx.warning("This might take a while")

    # Report progress (if client supports it)
    await ctx.report_progress(0, 100)

    # Do work...
    for i in range(10):
        await ctx.report_progress(i * 10, 100)

    await ctx.info("Processing complete")
    return f"Processed: {data}"
```

### 6. Dependencies

Inject dependencies into your handlers:

```python
from fastmcp import FastMCP
from dataclasses import dataclass
import sqlite3

mcp = FastMCP("deps-demo")

# Define a dependency
@dataclass
class Database:
    path: str

    def query(self, sql: str) -> list:
        conn = sqlite3.connect(self.path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results

# Dependency provider
def get_database() -> Database:
    return Database("./data.db")

# Use dependency in tool
@mcp.tool()
def query_users(
    limit: int = 10,
    db: Database = mcp.depends(get_database)
) -> str:
    """Query users from database."""
    results = db.query(f"SELECT * FROM users LIMIT {limit}")
    return str(results)

# Multiple dependencies
@dataclass
class CacheClient:
    def get(self, key: str) -> str | None:
        return None  # Simplified

def get_cache() -> CacheClient:
    return CacheClient()

@mcp.tool()
def get_user_cached(
    user_id: str,
    db: Database = mcp.depends(get_database),
    cache: CacheClient = mcp.depends(get_cache)
) -> str:
    """Get user with caching."""
    # Check cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached

    # Query database
    results = db.query(f"SELECT * FROM users WHERE id = '{user_id}'")
    return str(results[0]) if results else "Not found"
```

### 7. Lifespan Management

Handle startup/shutdown:

```python
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Global resource that needs cleanup
db_connection = None

@asynccontextmanager
async def lifespan(mcp: FastMCP):
    """Manage server lifespan."""
    global db_connection

    # Startup
    print("Starting server...")
    db_connection = await create_db_connection()

    yield  # Server runs here

    # Shutdown
    print("Shutting down...")
    await db_connection.close()

mcp = FastMCP("lifespan-demo", lifespan=lifespan)
```

### 8. Error Handling

```python
from fastmcp import FastMCP

mcp = FastMCP("errors-demo")

class UserNotFoundError(Exception):
    """User not found."""
    pass

@mcp.tool()
def get_user(user_id: str) -> str:
    """Get user by ID."""
    users = {"1": "Alice", "2": "Bob"}

    if user_id not in users:
        # Errors are automatically converted to error responses
        raise UserNotFoundError(f"User {user_id} not found")

    return users[user_id]

@mcp.tool()
def safe_divide(a: float, b: float) -> float:
    """Divide two numbers safely."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### 9. Custom Serialization

```python
from fastmcp import FastMCP
from dataclasses import dataclass
from datetime import datetime
import json

mcp = FastMCP("serialization-demo")

@dataclass
class Event:
    name: str
    timestamp: datetime
    data: dict

@mcp.tool()
def create_event(name: str, data: dict) -> str:
    """Create an event with timestamp."""
    event = Event(
        name=name,
        timestamp=datetime.now(),
        data=data
    )

    # Custom serialization
    return json.dumps({
        "name": event.name,
        "timestamp": event.timestamp.isoformat(),
        "data": event.data
    })
```

---

## Running FastMCP Servers

### Development Mode

```python
from fastmcp import FastMCP

mcp = FastMCP("my-server")

# ... define tools, resources, prompts ...

if __name__ == "__main__":
    # Run with stdio transport
    mcp.run()
```

### As a Module

```bash
python -m my_server
```

### With Environment Variables

```python
import os
from fastmcp import FastMCP

mcp = FastMCP("config-server")

@mcp.tool()
def get_api_key() -> str:
    """Get the configured API key."""
    key = os.getenv("API_KEY")
    if not key:
        raise ValueError("API_KEY not configured")
    return f"Key: {key[:4]}****"
```

---

## Complete Example: Todo Server

```python
from fastmcp import FastMCP, Context
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

mcp = FastMCP("todo-server", version="1.0.0")

# Data model
@dataclass
class Todo:
    id: int
    title: str
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None

# In-memory storage
todos: dict[int, Todo] = {}
next_id = 1

# Tools
@mcp.tool()
def add_todo(
    title: str,
    due_date: Optional[str] = None
) -> str:
    """Add a new todo item."""
    global next_id

    todo = Todo(
        id=next_id,
        title=title,
        due_date=datetime.fromisoformat(due_date) if due_date else None
    )
    todos[next_id] = todo
    next_id += 1

    return json.dumps({
        "id": todo.id,
        "title": todo.title,
        "message": "Todo created successfully"
    })

@mcp.tool()
def complete_todo(todo_id: int) -> str:
    """Mark a todo as completed."""
    if todo_id not in todos:
        raise ValueError(f"Todo {todo_id} not found")

    todos[todo_id].completed = True
    return f"Todo {todo_id} marked as completed"

@mcp.tool()
def delete_todo(todo_id: int) -> str:
    """Delete a todo item."""
    if todo_id not in todos:
        raise ValueError(f"Todo {todo_id} not found")

    del todos[todo_id]
    return f"Todo {todo_id} deleted"

@mcp.tool()
def list_todos(
    show_completed: bool = True,
    limit: int = 10
) -> str:
    """List all todo items."""
    items = list(todos.values())

    if not show_completed:
        items = [t for t in items if not t.completed]

    items = items[:limit]

    return json.dumps([
        {
            "id": t.id,
            "title": t.title,
            "completed": t.completed,
            "due_date": t.due_date.isoformat() if t.due_date else None
        }
        for t in items
    ], indent=2)

# Resources
@mcp.resource("todos://all")
def get_all_todos() -> str:
    """Get all todos as a resource."""
    return list_todos(show_completed=True, limit=100)

@mcp.resource("todos://{todo_id}")
def get_todo(todo_id: str) -> str:
    """Get a specific todo."""
    tid = int(todo_id)
    if tid not in todos:
        return '{"error": "Not found"}'

    t = todos[tid]
    return json.dumps({
        "id": t.id,
        "title": t.title,
        "completed": t.completed,
        "created_at": t.created_at.isoformat(),
        "due_date": t.due_date.isoformat() if t.due_date else None
    })

# Prompts
@mcp.prompt()
def daily_planning() -> list:
    """Template for daily planning."""
    from fastmcp.prompts import Message
    return [
        Message(
            role="user",
            content="""Please help me plan my day. Here are my todos:

${todos://all}

Suggest a priority order and time estimates for each task."""
        )
    ]

if __name__ == "__main__":
    mcp.run()
```

---

## Testing FastMCP Servers

```python
import pytest
from fastmcp import FastMCP

# Your server
mcp = FastMCP("test-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Tests
@pytest.mark.asyncio
async def test_add_tool():
    """Test the add tool directly."""
    # Access the underlying function
    result = add(2, 3)
    assert result == 5

@pytest.mark.asyncio
async def test_tool_schema():
    """Test that tool schema is generated correctly."""
    tools = mcp._tools
    assert "add" in tools

    schema = tools["add"].inputSchema
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"
```

---

## Best Practices

1. **Use Type Hints**: FastMCP relies on type hints for schema generation
2. **Write Docstrings**: They become tool/resource descriptions
3. **Handle Errors**: Raise exceptions with clear messages
4. **Use Async When Needed**: For I/O operations, use async
5. **Inject Dependencies**: Makes testing easier
6. **Group Related Tools**: Use multiple FastMCP instances if needed

---

## Common Patterns

### API Wrapper

```python
from fastmcp import FastMCP
import httpx

mcp = FastMCP("github-server")

@mcp.tool()
async def get_repo_info(owner: str, repo: str) -> str:
    """Get GitHub repository information."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}"
        )
        return response.text
```

### Database CRUD

```python
from fastmcp import FastMCP
import sqlite3

mcp = FastMCP("db-server")
DB_PATH = "app.db"

@mcp.tool()
def create_record(table: str, data: dict) -> str:
    """Create a new record."""
    # ... implementation

@mcp.tool()
def read_records(table: str, where: str = None) -> str:
    """Read records from table."""
    # ... implementation

@mcp.tool()
def update_record(table: str, id: int, data: dict) -> str:
    """Update a record."""
    # ... implementation

@mcp.tool()
def delete_record(table: str, id: int) -> str:
    """Delete a record."""
    # ... implementation
```

### File Operations

```python
from fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("file-server")
ROOT = Path("/safe/directory")

@mcp.tool()
def read_file(path: str) -> str:
    """Read file contents."""
    # Validate path is within root
    full_path = (ROOT / path).resolve()
    if not str(full_path).startswith(str(ROOT)):
        raise ValueError("Access denied")
    return full_path.read_text()
```

---

## References

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/) (similar patterns)

# Week 12 - Model Context Protocol (MCP)

## References

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP](https://github.com/jlowin/fastmcp) - High-level MCP framework
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)

## Installation

```bash
# Core MCP packages
uv add mcp fastmcp

# Additional utilities
uv add httpx pydantic aiofiles

# For testing
uv add pytest pytest-asyncio
```

## What is MCP?

The **Model Context Protocol (MCP)** is an open standard for connecting AI assistants to external data sources, tools, and services. Think of it as a universal API that allows LLMs to:

- 📂 **Access Resources**: Files, databases, APIs
- 🔧 **Use Tools**: Execute functions, run commands
- 💬 **Follow Prompts**: Pre-defined conversation templates

### Why MCP Matters

Before MCP, every AI integration required custom code. With MCP:

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────────┐
│   AI Assistant  │────▶│  MCP Client │────▶│   MCP Server     │
│  (Claude, etc.) │◀────│             │◀────│ (Your Service)   │
└─────────────────┘     └─────────────┘     └──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Standard       │
                    │  Protocol       │
                    │  (JSON-RPC)     │
                    └─────────────────┘
```

## Core Concepts

### 1. MCP Architecture

```
MCP Server (Provider)           MCP Client (Consumer)
├── Resources                   ├── Connects to servers
│   └── Data the AI can access  ├── Lists available tools
├── Tools                       ├── Calls tools
│   └── Functions AI can call   └── Reads resources
└── Prompts
    └── Conversation templates
```

### 2. Resources

Resources are data that the AI can read:

```python
from mcp.server import Server
from mcp.types import Resource, TextResourceContents

server = Server("my-server")

@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file:///config.json",
            name="Configuration",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "file:///config.json":
        return TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text='{"setting": "value"}'
        )
```

### 3. Tools

Tools are functions the AI can call:

```python
from mcp.server import Server
from mcp.types import Tool

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
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "add":
        return {"result": arguments["a"] + arguments["b"]}
```

### 4. Prompts

Prompts are conversation templates:

```python
from mcp.server import Server
from mcp.types import Prompt, PromptMessage, TextContent

server = Server("prompts")

@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code-review",
            description="Review code for issues"
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict = None):
    if name == "code-review":
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Please review this code for bugs and improvements."
                )
            )
        ]
```

---

## FastMCP: Simplified MCP Development

**FastMCP** is a high-level framework that makes building MCP servers easy.
See [side-learning/fastmcp.md](../side-learning/fastmcp.md) for detailed FastMCP guide.

### Basic FastMCP Server

```python
from fastmcp import FastMCP

# Create server
mcp = FastMCP("My Server")

# Define a tool with decorator
@mcp.tool()
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

# Define a resource
@mcp.resource("config://app")
def get_config() -> str:
    """Get application configuration."""
    return '{"version": "1.0", "debug": true}'

# Run the server
if __name__ == "__main__":
    mcp.run()
```

### FastMCP with Context

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("Context Server")

@mcp.tool()
async def process_file(path: str, ctx: Context) -> str:
    """Process a file with progress reporting."""
    # Log progress
    await ctx.info(f"Processing {path}...")

    # Report progress
    await ctx.report_progress(50, 100)

    # Complete
    await ctx.info("Done!")
    return f"Processed {path}"
```

### FastMCP with Dependencies

```python
from fastmcp import FastMCP
from dataclasses import dataclass

@dataclass
class DatabaseConnection:
    host: str
    port: int

    async def query(self, sql: str) -> list:
        # Simulated query
        return [{"id": 1, "name": "test"}]

mcp = FastMCP("Database Server")

# Dependency injection
def get_db() -> DatabaseConnection:
    return DatabaseConnection(host="localhost", port=5432)

@mcp.tool()
async def query_users(db: DatabaseConnection = mcp.depends(get_db)) -> str:
    """Query all users from database."""
    results = await db.query("SELECT * FROM users")
    return str(results)
```

---

## Building MCP Servers

### Exercise 1 - Basic MCP Server: [exercises/exercise_basic_1_mcp_server.py](exercises/exercise_basic_1_mcp_server.py)

### Complete Server Example

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, Tool, TextResourceContents,
    CallToolResult, TextContent
)
import json

# Create server
server = Server("demo-server")

# In-memory data store
data_store = {
    "users": [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"}
    ]
}

# === RESOURCES ===

@server.list_resources()
async def list_resources():
    """List available resources."""
    return [
        Resource(
            uri="data://users",
            name="Users Database",
            description="List of all users",
            mimeType="application/json"
        ),
        Resource(
            uri="data://config",
            name="Configuration",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    """Read a resource by URI."""
    if uri == "data://users":
        return TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text=json.dumps(data_store["users"], indent=2)
        )
    elif uri == "data://config":
        return TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text='{"server": "demo", "version": "1.0"}'
        )
    raise ValueError(f"Unknown resource: {uri}")

# === TOOLS ===

@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="get_user",
            description="Get a user by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user's ID"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="add_user",
            description="Add a new user",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "default": "user"}
                },
                "required": ["name"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute a tool."""
    if name == "get_user":
        user_id = arguments["user_id"]
        for user in data_store["users"]:
            if user["id"] == user_id:
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(user))]
                )
        return CallToolResult(
            content=[TextContent(type="text", text="User not found")],
            isError=True
        )

    elif name == "add_user":
        new_id = max(u["id"] for u in data_store["users"]) + 1
        new_user = {
            "id": new_id,
            "name": arguments["name"],
            "role": arguments.get("role", "user")
        }
        data_store["users"].append(new_user)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(new_user))]
        )

    raise ValueError(f"Unknown tool: {name}")

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## MCP Client Implementation

### Exercise 2 - Intermediate MCP Client: [exercises/exercise_intermediate_2_mcp_client.py](exercises/exercise_intermediate_2_mcp_client.py)

### Connecting to an MCP Server

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
import asyncio

async def connect_to_server():
    """Connect to an MCP server."""
    # Start server process
    server_params = {
        "command": "python",
        "args": ["my_server.py"]
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # List resources
            resources = await session.list_resources()
            print("Available resources:")
            for r in resources.resources:
                print(f"  - {r.name}: {r.uri}")

            # List tools
            tools = await session.list_tools()
            print("\nAvailable tools:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            # Call a tool
            result = await session.call_tool("get_user", {"user_id": 1})
            print(f"\nTool result: {result}")

            # Read a resource
            content = await session.read_resource("data://users")
            print(f"\nResource content: {content}")

asyncio.run(connect_to_server())
```

### Client with Error Handling

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager

class MCPClient:
    """Robust MCP client with error handling."""

    def __init__(self, server_command: str, server_args: list[str] = None):
        self.server_command = server_command
        self.server_args = server_args or []
        self.session: ClientSession | None = None

    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server."""
        server_params = {
            "command": self.server_command,
            "args": self.server_args
        }

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    yield self
        finally:
            self.session = None

    async def call_tool_safe(self, name: str, arguments: dict) -> dict:
        """Call a tool with error handling."""
        if not self.session:
            raise RuntimeError("Not connected")

        try:
            result = await self.session.call_tool(name, arguments)
            if result.isError:
                return {"success": False, "error": result.content[0].text}
            return {"success": True, "data": result.content[0].text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_capabilities(self) -> dict:
        """Get server capabilities."""
        if not self.session:
            raise RuntimeError("Not connected")

        resources = await self.session.list_resources()
        tools = await self.session.list_tools()
        prompts = await self.session.list_prompts()

        return {
            "resources": [r.name for r in resources.resources],
            "tools": [t.name for t in tools.tools],
            "prompts": [p.name for p in prompts.prompts]
        }

# Usage
async def main():
    client = MCPClient("python", ["server.py"])

    async with client.connect():
        caps = await client.list_capabilities()
        print(f"Server capabilities: {caps}")

        result = await client.call_tool_safe("get_user", {"user_id": 1})
        print(f"Result: {result}")
```

---

## Advanced MCP Patterns

### Exercise 3 - Advanced MCP Patterns: [exercises/exercise_advanced_3_mcp_patterns.py](exercises/exercise_advanced_3_mcp_patterns.py)

### Dynamic Resource Templates

```python
from mcp.server import Server
from mcp.types import Resource, ResourceTemplate, TextResourceContents

server = Server("template-server")

@server.list_resource_templates()
async def list_templates():
    """List resource templates for dynamic URIs."""
    return [
        ResourceTemplate(
            uriTemplate="user://{user_id}",
            name="User Profile",
            description="Get user profile by ID"
        ),
        ResourceTemplate(
            uriTemplate="file://{path}",
            name="File Content",
            description="Read any file by path"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    """Read resources including templated ones."""
    if uri.startswith("user://"):
        user_id = uri.replace("user://", "")
        # Fetch user data
        return TextResourceContents(
            uri=uri,
            mimeType="application/json",
            text=f'{{"id": "{user_id}", "name": "User {user_id}"}}'
        )

    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        with open(path, "r") as f:
            return TextResourceContents(
                uri=uri,
                mimeType="text/plain",
                text=f.read()
            )
```

### Streaming Results

```python
from mcp.server import Server
from mcp.types import Tool, CallToolResult, TextContent
import asyncio

server = Server("streaming-server")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="long_operation",
            description="A long-running operation with progress",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "default": 5}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "long_operation":
        steps = arguments.get("steps", 5)
        results = []

        for i in range(steps):
            await asyncio.sleep(0.5)  # Simulate work
            results.append(f"Completed step {i + 1}/{steps}")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(results))]
        )
```

### Multi-Server Client

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from dataclasses import dataclass

@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str]

class MultiServerClient:
    """Client that connects to multiple MCP servers."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.contexts = []

    async def add_server(self, config: ServerConfig):
        """Add and connect to a server."""
        server_params = {
            "command": config.command,
            "args": config.args
        }

        ctx1 = stdio_client(server_params)
        read, write = await ctx1.__aenter__()
        self.contexts.append(ctx1)

        ctx2 = ClientSession(read, write)
        session = await ctx2.__aenter__()
        self.contexts.append(ctx2)

        await session.initialize()
        self.sessions[config.name] = session

    async def call_tool(self, server: str, tool: str, arguments: dict):
        """Call a tool on a specific server."""
        if server not in self.sessions:
            raise ValueError(f"Unknown server: {server}")

        return await self.sessions[server].call_tool(tool, arguments)

    async def list_all_tools(self) -> dict[str, list[str]]:
        """List tools from all connected servers."""
        all_tools = {}
        for name, session in self.sessions.items():
            tools = await session.list_tools()
            all_tools[name] = [t.name for t in tools.tools]
        return all_tools

    async def close(self):
        """Close all connections."""
        for ctx in reversed(self.contexts):
            await ctx.__aexit__(None, None, None)
        self.sessions.clear()
        self.contexts.clear()
```

---

## Building Real-World MCP Servers

### File System Server

```python
from fastmcp import FastMCP
from pathlib import Path
import os

mcp = FastMCP("File System Server")

@mcp.resource("fs://list/{directory}")
def list_directory(directory: str) -> str:
    """List files in a directory."""
    path = Path(directory)
    if not path.exists():
        return f"Directory not found: {directory}"

    files = []
    for item in path.iterdir():
        files.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else 0
        })

    import json
    return json.dumps(files, indent=2)

@mcp.tool()
def read_file(path: str) -> str:
    """Read contents of a file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return file_path.read_text()

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"Written {len(content)} bytes to {path}"

@mcp.tool()
def search_files(directory: str, pattern: str) -> str:
    """Search for files matching a pattern."""
    import glob
    matches = glob.glob(f"{directory}/**/{pattern}", recursive=True)
    return "\n".join(matches) if matches else "No matches found"
```

### Database Query Server

```python
from fastmcp import FastMCP
import sqlite3
import json

mcp = FastMCP("SQLite Server")

def get_connection():
    return sqlite3.connect("database.db")

@mcp.tool()
def execute_query(sql: str) -> str:
    """Execute a SQL query and return results."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(sql)

        if sql.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            return json.dumps(results, indent=2)
        else:
            conn.commit()
            return f"Query executed. Rows affected: {cursor.rowcount}"
    finally:
        conn.close()

@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    return json.dumps(tables)

@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get schema for a table."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()

    schema = [
        {"name": col[1], "type": col[2], "nullable": not col[3]}
        for col in columns
    ]
    return json.dumps(schema, indent=2)
```

### API Gateway Server

```python
from fastmcp import FastMCP, Context
import httpx

mcp = FastMCP("API Gateway")

@mcp.tool()
async def http_get(url: str, headers: dict = None) -> str:
    """Make an HTTP GET request."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers or {})
        return response.text

@mcp.tool()
async def http_post(url: str, data: dict, headers: dict = None) -> str:
    """Make an HTTP POST request."""
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers or {})
        return response.text

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch weather for a city (example API)."""
    url = f"https://api.weatherapi.com/v1/current.json?key=KEY&q={city}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

---

## MCP Server Configuration

### Claude Desktop Configuration

To use your MCP server with Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/my_server.py"]
    },
    "file-server": {
      "command": "python",
      "args": ["/path/to/file_server.py"],
      "env": {
        "ROOT_DIR": "/home/user/documents"
      }
    }
  }
}
```

### Server Metadata

```python
from mcp.server import Server

server = Server(
    name="my-server",
    version="1.0.0"
)

# Server info is automatically included in initialization
```

---

## Testing MCP Servers

```python
import pytest
from mcp.server import Server

@pytest.fixture
def server():
    """Create a test server."""
    server = Server("test-server")

    @server.list_tools()
    async def list_tools():
        return [{"name": "test_tool", "inputSchema": {}}]

    @server.call_tool()
    async def call_tool(name, arguments):
        return {"result": "success"}

    return server

@pytest.mark.asyncio
async def test_list_tools(server):
    """Test tool listing."""
    # Use server's internal methods for testing
    tools = await server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"

@pytest.mark.asyncio
async def test_call_tool(server):
    """Test tool calling."""
    result = await server.call_tool("test_tool", {})
    assert result["result"] == "success"
```

---

## Weekly Project

Build a **Knowledge Base MCP Server** that provides AI assistants access to a document collection.

### [exercises/project_pipeline.py](exercises/project_pipeline.py)

Features:

- Document storage and retrieval
- Full-text search
- Document metadata management
- Tagging system
- Version history
- Export capabilities

---

## Interview Questions

1. **What is the Model Context Protocol and why was it created?**
   - MCP is an open standard for connecting AI to external tools/data
   - Created to standardize AI integrations (like USB for AI)
   - Eliminates need for custom integrations per AI model

2. **Explain the three main primitives in MCP.**
   - **Resources**: Data the AI can read (files, DB records)
   - **Tools**: Functions the AI can execute
   - **Prompts**: Pre-defined conversation templates

3. **How does MCP differ from traditional REST APIs for AI?**
   - MCP is bidirectional (server can push updates)
   - Built-in schema validation for tool inputs
   - Standardized discovery (list_tools, list_resources)
   - Designed for AI consumption vs human/app consumption

4. **What transport mechanisms does MCP support?**
   - stdio (standard input/output) for local servers
   - HTTP with SSE for remote servers
   - Custom transports can be implemented

5. **How would you handle authentication in an MCP server?**
   - Environment variables for API keys
   - OAuth tokens passed via initialization
   - Custom auth headers in HTTP transport

6. **What's the difference between MCP SDK and FastMCP?**
   - MCP SDK is low-level, full control
   - FastMCP is high-level, decorator-based, less boilerplate
   - FastMCP handles serialization automatically

7. **How do you implement resource templates?**
   - Use URI templates like `user://{id}`
   - List templates via list_resource_templates
   - Parse URIs in read_resource handler

8. **Explain error handling in MCP tool calls.**
   - Return CallToolResult with isError=True
   - Include error message in content
   - Client handles errors gracefully

---

## Takeaway Checklist

- [ ] Understand MCP architecture (client, server, protocol)
- [ ] Build MCP servers using both SDK and FastMCP
- [ ] Implement resources, tools, and prompts
- [ ] Connect clients to MCP servers
- [ ] Handle errors and edge cases properly
- [ ] Configure servers for Claude Desktop
- [ ] Test MCP servers effectively
- [ ] Build real-world integrations (files, databases, APIs)

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 13](../week-13/README.md)**

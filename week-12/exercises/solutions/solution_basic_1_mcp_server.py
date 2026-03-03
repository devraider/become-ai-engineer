"""
Week 12 - Exercise 1 (Basic): MCP Server Fundamentals - SOLUTIONS

Complete implementations for MCP server components.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import json
import uuid


# =============================================================================
# Part 1: Resource Definition
# =============================================================================
@dataclass
class Resource:
    """
    Represents an MCP resource that can be exposed to clients.

    Attributes:
        uri: Unique identifier for the resource (e.g., "file:///path/to/file")
        name: Human-readable name
        description: Optional description of the resource
        mime_type: Optional MIME type (e.g., "text/plain", "application/json")
    """

    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert resource to MCP-compatible dictionary."""
        result = {"uri": self.uri, "name": self.name}
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Resource":
        """Create Resource from dictionary."""
        return cls(
            uri=data["uri"],
            name=data["name"],
            description=data.get("description"),
            mime_type=data.get("mimeType"),
        )


# =============================================================================
# Part 2: Resource Content
# =============================================================================
@dataclass
class ResourceContent:
    """
    Represents the content of a resource.

    Attributes:
        uri: The resource URI this content belongs to
        mime_type: Content MIME type
        text: Text content (for text resources)
        blob: Binary content as base64 (for binary resources)
    """

    uri: str
    mime_type: str = "text/plain"
    text: Optional[str] = None
    blob: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to MCP-compatible dictionary."""
        result = {"uri": self.uri, "mimeType": self.mime_type}
        if self.text is not None:
            result["text"] = self.text
        if self.blob is not None:
            result["blob"] = self.blob
        return result

    def is_text(self) -> bool:
        """Check if content is text-based."""
        return self.text is not None

    def is_binary(self) -> bool:
        """Check if content is binary."""
        return self.blob is not None


# =============================================================================
# Part 3: Tool Parameter Schema
# =============================================================================
@dataclass
class ToolParameter:
    """
    Defines a parameter for an MCP tool.

    Attributes:
        name: Parameter name
        description: Parameter description
        type: JSON Schema type (string, number, boolean, object, array)
        required: Whether the parameter is required
        default: Optional default value
        enum: Optional list of allowed values
    """

    name: str
    description: str
    type: str = "string"
    required: bool = False
    default: Any = None
    enum: Optional[list] = None

    def to_schema(self) -> dict:
        """Convert to JSON Schema property definition."""
        schema = {"type": self.type, "description": self.description}
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        return schema


# =============================================================================
# Part 4: Tool Definition
# =============================================================================
@dataclass
class Tool:
    """
    Represents an MCP tool that can be called by clients.

    Attributes:
        name: Unique tool name
        description: Human-readable description
        parameters: List of tool parameters
        handler: Function to execute when tool is called
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_dict(self) -> dict:
        """Convert tool to MCP-compatible dictionary."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, arguments: dict) -> Any:
        """Execute the tool with given arguments."""
        if self.handler is None:
            raise ValueError(f"No handler defined for tool {self.name}")

        result = self.handler(arguments)
        if hasattr(result, "__await__"):
            return await result
        return result


# =============================================================================
# Part 5: Tool Result
# =============================================================================
@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.

    Attributes:
        content: List of content items (text, image, etc.)
        is_error: Whether this result represents an error
    """

    content: list = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def text(cls, text: str) -> "ToolResult":
        """Create a text result."""
        return cls(content=[{"type": "text", "text": text}])

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """Create an error result."""
        return cls(content=[{"type": "text", "text": message}], is_error=True)

    @classmethod
    def image(cls, data: str, mime_type: str = "image/png") -> "ToolResult":
        """Create an image result."""
        return cls(content=[{"type": "image", "data": data, "mimeType": mime_type}])

    def to_dict(self) -> dict:
        """Convert to MCP-compatible dictionary."""
        return {"content": self.content, "isError": self.is_error}


# =============================================================================
# Part 6: Prompt Message
# =============================================================================
class MessageRole(str, Enum):
    """Role of a message in a prompt."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class PromptMessage:
    """
    Represents a message in an MCP prompt.

    Attributes:
        role: Message role (user or assistant)
        content: Message content
    """

    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        """Convert to MCP-compatible dictionary."""
        return {
            "role": self.role.value,
            "content": {"type": "text", "text": self.content},
        }

    @classmethod
    def user(cls, content: str) -> "PromptMessage":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "PromptMessage":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


# =============================================================================
# Part 7: Prompt Definition
# =============================================================================
@dataclass
class PromptArgument:
    """Defines an argument for a prompt."""

    name: str
    description: str
    required: bool = False


@dataclass
class Prompt:
    """
    Represents an MCP prompt template.

    Attributes:
        name: Unique prompt name
        description: Human-readable description
        arguments: List of prompt arguments
        template: Template string or callable
    """

    name: str
    description: str
    arguments: list[PromptArgument] = field(default_factory=list)
    template: Optional[str | Callable] = None

    def to_dict(self) -> dict:
        """Convert to MCP-compatible dictionary."""
        args = []
        for arg in self.arguments:
            args.append(
                {
                    "name": arg.name,
                    "description": arg.description,
                    "required": arg.required,
                }
            )

        return {"name": self.name, "description": self.description, "arguments": args}

    def render(self, arguments: dict) -> list[PromptMessage]:
        """Render the prompt with given arguments."""
        if callable(self.template):
            return self.template(arguments)

        if self.template:
            rendered = self.template.format(**arguments)
            return [PromptMessage.user(rendered)]

        return []


# =============================================================================
# Part 8: Resource Registry
# =============================================================================
class ResourceRegistry:
    """
    Manages resource registration and retrieval.
    """

    def __init__(self):
        self._resources: dict[str, Resource] = {}
        self._handlers: dict[str, Callable] = {}

    def register(
        self,
        uri: str,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Resource:
        """Register a resource with its handler."""
        resource = Resource(
            uri=uri, name=name, description=description, mime_type=mime_type
        )
        self._resources[uri] = resource
        self._handlers[uri] = handler
        return resource

    def unregister(self, uri: str) -> bool:
        """Unregister a resource."""
        if uri in self._resources:
            del self._resources[uri]
            del self._handlers[uri]
            return True
        return False

    def get(self, uri: str) -> Optional[Resource]:
        """Get a resource by URI."""
        return self._resources.get(uri)

    def list_all(self) -> list[Resource]:
        """List all registered resources."""
        return list(self._resources.values())

    async def read(self, uri: str) -> Optional[ResourceContent]:
        """Read resource content."""
        if uri not in self._handlers:
            return None

        handler = self._handlers[uri]
        result = handler(uri)

        if hasattr(result, "__await__"):
            result = await result

        if isinstance(result, ResourceContent):
            return result

        resource = self._resources.get(uri)
        return ResourceContent(
            uri=uri,
            mime_type=resource.mime_type if resource else "text/plain",
            text=str(result),
        )


# =============================================================================
# Part 9: Tool Registry
# =============================================================================
class ToolRegistry:
    """
    Manages tool registration and execution.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[list[ToolParameter]] = None,
    ) -> Tool:
        """Register a tool."""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            handler=handler,
        )
        self._tools[name] = tool
        return tool

    def register_tool(self, tool: Tool) -> None:
        """Register an existing tool object."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    async def call(self, name: str, arguments: dict) -> ToolResult:
        """Call a tool with arguments."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.error(f"Unknown tool: {name}")

        try:
            result = await tool.execute(arguments)

            if isinstance(result, ToolResult):
                return result

            return ToolResult.text(str(result))
        except Exception as e:
            return ToolResult.error(f"Tool execution failed: {str(e)}")


# =============================================================================
# Part 10: Prompt Registry
# =============================================================================
class PromptRegistry:
    """
    Manages prompt registration and rendering.
    """

    def __init__(self):
        self._prompts: dict[str, Prompt] = {}

    def register(
        self,
        name: str,
        description: str,
        template: str | Callable,
        arguments: Optional[list[PromptArgument]] = None,
    ) -> Prompt:
        """Register a prompt."""
        prompt = Prompt(
            name=name,
            description=description,
            arguments=arguments or [],
            template=template,
        )
        self._prompts[name] = prompt
        return prompt

    def register_prompt(self, prompt: Prompt) -> None:
        """Register an existing prompt object."""
        self._prompts[prompt.name] = prompt

    def unregister(self, name: str) -> bool:
        """Unregister a prompt."""
        if name in self._prompts:
            del self._prompts[name]
            return True
        return False

    def get(self, name: str) -> Optional[Prompt]:
        """Get a prompt by name."""
        return self._prompts.get(name)

    def list_all(self) -> list[Prompt]:
        """List all registered prompts."""
        return list(self._prompts.values())

    def render(self, name: str, arguments: dict) -> list[PromptMessage]:
        """Render a prompt with arguments."""
        prompt = self._prompts.get(name)
        if not prompt:
            return []
        return prompt.render(arguments)


# =============================================================================
# Bonus Part 1: Simple MCP Server
# =============================================================================
class SimpleMCPServer:
    """
    A simple MCP server combining resources, tools, and prompts.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.resources = ResourceRegistry()
        self.tools = ToolRegistry()
        self.prompts = PromptRegistry()

    def get_server_info(self) -> dict:
        """Get server information."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
        }

    async def handle_request(self, method: str, params: dict) -> dict:
        """Handle an MCP request."""
        handlers = {
            "initialize": self._handle_initialize,
            "resources/list": self._handle_list_resources,
            "resources/read": self._handle_read_resource,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "prompts/list": self._handle_list_prompts,
            "prompts/get": self._handle_get_prompt,
        }

        handler = handlers.get(method)
        if not handler:
            return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

        return await handler(params)

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": self.get_server_info(),
            "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
        }

    async def _handle_list_resources(self, params: dict) -> dict:
        """Handle resources/list request."""
        resources = self.resources.list_all()
        return {"resources": [r.to_dict() for r in resources]}

    async def _handle_read_resource(self, params: dict) -> dict:
        """Handle resources/read request."""
        uri = params.get("uri")
        content = await self.resources.read(uri)

        if content is None:
            return {"error": {"code": -32602, "message": f"Resource not found: {uri}"}}

        return {"contents": [content.to_dict()]}

    async def _handle_list_tools(self, params: dict) -> dict:
        """Handle tools/list request."""
        tools = self.tools.list_all()
        return {"tools": [t.to_dict() for t in tools]}

    async def _handle_call_tool(self, params: dict) -> dict:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        result = await self.tools.call(name, arguments)
        return result.to_dict()

    async def _handle_list_prompts(self, params: dict) -> dict:
        """Handle prompts/list request."""
        prompts = self.prompts.list_all()
        return {"prompts": [p.to_dict() for p in prompts]}

    async def _handle_get_prompt(self, params: dict) -> dict:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        messages = self.prompts.render(name, arguments)
        return {"messages": [m.to_dict() for m in messages]}


# =============================================================================
# Bonus Part 2: Data Store Server Example
# =============================================================================
class DataStoreServer(SimpleMCPServer):
    """
    Example server implementing a simple key-value data store.
    """

    def __init__(self):
        super().__init__("data-store", "1.0.0")
        self._data: dict[str, Any] = {}
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()

    def _setup_resources(self) -> None:
        """Set up data store resources."""
        self.resources.register(
            uri="data://store/keys",
            name="All Keys",
            handler=lambda uri: json.dumps(list(self._data.keys())),
            description="List of all keys in the store",
            mime_type="application/json",
        )

        self.resources.register(
            uri="data://store/summary",
            name="Store Summary",
            handler=self._get_summary,
            description="Summary of the data store",
            mime_type="application/json",
        )

    def _get_summary(self, uri: str) -> str:
        """Get store summary."""
        return json.dumps(
            {"total_keys": len(self._data), "keys": list(self._data.keys())[:10]}
        )

    def _setup_tools(self) -> None:
        """Set up data store tools."""
        self.tools.register(
            name="get",
            description="Get a value from the store",
            handler=self._tool_get,
            parameters=[
                ToolParameter(
                    name="key",
                    description="Key to retrieve",
                    type="string",
                    required=True,
                )
            ],
        )

        self.tools.register(
            name="set",
            description="Set a value in the store",
            handler=self._tool_set,
            parameters=[
                ToolParameter(
                    name="key", description="Key to set", type="string", required=True
                ),
                ToolParameter(
                    name="value", description="Value to store", required=True
                ),
            ],
        )

        self.tools.register(
            name="delete",
            description="Delete a key from the store",
            handler=self._tool_delete,
            parameters=[
                ToolParameter(
                    name="key",
                    description="Key to delete",
                    type="string",
                    required=True,
                )
            ],
        )

        self.tools.register(
            name="list",
            description="List all keys in the store",
            handler=self._tool_list,
            parameters=[],
        )

    def _tool_get(self, args: dict) -> ToolResult:
        """Get value tool handler."""
        key = args["key"]
        if key in self._data:
            value = self._data[key]
            return ToolResult.text(json.dumps({"key": key, "value": value}))
        return ToolResult.error(f"Key not found: {key}")

    def _tool_set(self, args: dict) -> ToolResult:
        """Set value tool handler."""
        key = args["key"]
        value = args["value"]
        self._data[key] = value
        return ToolResult.text(f"Set {key} = {value}")

    def _tool_delete(self, args: dict) -> ToolResult:
        """Delete key tool handler."""
        key = args["key"]
        if key in self._data:
            del self._data[key]
            return ToolResult.text(f"Deleted key: {key}")
        return ToolResult.error(f"Key not found: {key}")

    def _tool_list(self, args: dict) -> ToolResult:
        """List keys tool handler."""
        keys = list(self._data.keys())
        return ToolResult.text(json.dumps(keys))

    def _setup_prompts(self) -> None:
        """Set up data store prompts."""
        self.prompts.register(
            name="query_store",
            description="Generate a query for the data store",
            template="Please help me {action} the key '{key}' in the data store.",
            arguments=[
                PromptArgument(
                    name="action",
                    description="Action to perform (get, set, delete)",
                    required=True,
                ),
                PromptArgument(
                    name="key", description="Key to operate on", required=True
                ),
            ],
        )


# =============================================================================
# Bonus Part 3: Server Builder Pattern
# =============================================================================
class MCPServerBuilder:
    """
    Builder pattern for creating MCP servers.
    """

    def __init__(self):
        self._name: str = "mcp-server"
        self._version: str = "1.0.0"
        self._resources: list[tuple] = []
        self._tools: list[tuple] = []
        self._prompts: list[tuple] = []

    def name(self, name: str) -> "MCPServerBuilder":
        """Set server name."""
        self._name = name
        return self

    def version(self, version: str) -> "MCPServerBuilder":
        """Set server version."""
        self._version = version
        return self

    def resource(
        self,
        uri: str,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "MCPServerBuilder":
        """Add a resource."""
        self._resources.append((uri, name, handler, description, mime_type))
        return self

    def tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[list[ToolParameter]] = None,
    ) -> "MCPServerBuilder":
        """Add a tool."""
        self._tools.append((name, description, handler, parameters))
        return self

    def prompt(
        self,
        name: str,
        description: str,
        template: str | Callable,
        arguments: Optional[list[PromptArgument]] = None,
    ) -> "MCPServerBuilder":
        """Add a prompt."""
        self._prompts.append((name, description, template, arguments))
        return self

    def build(self) -> SimpleMCPServer:
        """Build the server."""
        server = SimpleMCPServer(self._name, self._version)

        for uri, name, handler, desc, mime in self._resources:
            server.resources.register(uri, name, handler, desc, mime)

        for name, desc, handler, params in self._tools:
            server.tools.register(name, desc, handler, params)

        for name, desc, template, args in self._prompts:
            server.prompts.register(name, desc, template, args)

        return server


# =============================================================================
# Example Usage
# =============================================================================
def create_example_server() -> SimpleMCPServer:
    """Create an example MCP server using the builder pattern."""
    return (
        MCPServerBuilder()
        .name("example-server")
        .version("1.0.0")
        .resource(
            uri="example://greeting",
            name="Greeting",
            handler=lambda uri: "Hello from MCP!",
            description="A friendly greeting",
            mime_type="text/plain",
        )
        .tool(
            name="echo",
            description="Echo back the input",
            handler=lambda args: ToolResult.text(args.get("message", "")),
            parameters=[
                ToolParameter(
                    name="message",
                    description="Message to echo",
                    type="string",
                    required=True,
                )
            ],
        )
        .tool(
            name="add",
            description="Add two numbers",
            handler=lambda args: ToolResult.text(
                str(args.get("a", 0) + args.get("b", 0))
            ),
            parameters=[
                ToolParameter(
                    name="a", description="First number", type="number", required=True
                ),
                ToolParameter(
                    name="b", description="Second number", type="number", required=True
                ),
            ],
        )
        .prompt(
            name="greeting",
            description="Generate a greeting",
            template="Hello {name}! Welcome to the MCP server.",
            arguments=[
                PromptArgument(name="name", description="Name to greet", required=True)
            ],
        )
        .build()
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        # Create and test the example server
        server = create_example_server()

        # Initialize
        result = await server.handle_request("initialize", {})
        print("Initialize:", json.dumps(result, indent=2))

        # List tools
        result = await server.handle_request("tools/list", {})
        print("\nTools:", json.dumps(result, indent=2))

        # Call echo tool
        result = await server.handle_request(
            "tools/call", {"name": "echo", "arguments": {"message": "Hello MCP!"}}
        )
        print("\nEcho result:", json.dumps(result, indent=2))

        # Call add tool
        result = await server.handle_request(
            "tools/call", {"name": "add", "arguments": {"a": 5, "b": 3}}
        )
        print("\nAdd result:", json.dumps(result, indent=2))

        # Test data store server
        print("\n" + "=" * 50)
        print("Testing DataStoreServer")
        print("=" * 50)

        ds = DataStoreServer()

        # Set some values
        await ds.handle_request(
            "tools/call",
            {"name": "set", "arguments": {"key": "user", "value": "Alice"}},
        )

        # Get the value
        result = await ds.handle_request(
            "tools/call", {"name": "get", "arguments": {"key": "user"}}
        )
        print("\nGet user:", json.dumps(result, indent=2))

        # List all keys
        result = await ds.handle_request(
            "tools/call", {"name": "list", "arguments": {}}
        )
        print("\nAll keys:", json.dumps(result, indent=2))

    asyncio.run(main())

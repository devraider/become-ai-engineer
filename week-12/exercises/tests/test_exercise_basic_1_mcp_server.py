"""
Tests for Week 12 - Exercise 1 (Basic): MCP Server Fundamentals
"""

import pytest
from exercise_basic_1_mcp_server import (
    Resource,
    ResourceContent,
    ToolParameter,
    Tool,
    ToolResult,
    PromptMessage,
    Prompt,
    ResourceRegistry,
    ToolRegistry,
    PromptRegistry,
    SimpleMCPServer,
    DataStoreServer,
    MCPServerBuilder,
)


# =============================================================================
# Test Resource Definition
# =============================================================================
class TestResource:
    """Tests for Resource dataclass."""

    def test_resource_creation(self):
        """Test creating a resource."""
        resource = Resource(
            uri="file:///config.json",
            name="Configuration",
            description="App configuration",
            mime_type="application/json",
        )

        assert resource.uri == "file:///config.json"
        assert resource.name == "Configuration"
        assert resource.mime_type == "application/json"

    def test_resource_default_mime_type(self):
        """Test default MIME type."""
        resource = Resource(uri="test://data", name="Test Data")

        assert resource.mime_type == "text/plain"

    def test_resource_to_dict(self):
        """Test converting resource to dict."""
        resource = Resource(
            uri="test://data", name="Test", description="A test resource"
        )

        result = resource.to_dict()

        assert result["uri"] == "test://data"
        assert result["name"] == "Test"
        assert "description" in result


class TestResourceContent:
    """Tests for ResourceContent dataclass."""

    def test_text_content(self):
        """Test text content."""
        content = ResourceContent(
            uri="test://data", mime_type="text/plain", text="Hello, World!"
        )

        assert content.text == "Hello, World!"
        assert not content.is_binary

    def test_binary_content(self):
        """Test binary content."""
        content = ResourceContent(
            uri="test://image", mime_type="image/png", blob=b"\x89PNG"
        )

        assert content.blob == b"\x89PNG"
        assert content.is_binary


# =============================================================================
# Test Tool Definition
# =============================================================================
class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_parameter_creation(self):
        """Test creating a parameter."""
        param = ToolParameter(
            name="message",
            param_type="string",
            description="The message to echo",
            required=True,
        )

        assert param.name == "message"
        assert param.param_type == "string"
        assert param.required is True

    def test_parameter_to_schema(self):
        """Test converting to JSON schema."""
        param = ToolParameter(
            name="count",
            param_type="integer",
            description="Number of items",
            required=False,
            default=10,
        )

        schema = param.to_schema()

        assert schema["type"] == "integer"
        assert schema["description"] == "Number of items"


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a tool."""
        param = ToolParameter("name", "string", "User name", True)
        tool = Tool(name="greet", description="Greet a user", parameters=[param])

        assert tool.name == "greet"
        assert len(tool.parameters) == 1

    def test_tool_to_schema(self):
        """Test converting tool to schema."""
        param = ToolParameter("x", "number", "First number", True)
        tool = Tool(name="square", description="Square a number", parameters=[param])

        schema = tool.to_schema()

        assert schema["name"] == "square"
        assert "inputSchema" in schema
        assert "properties" in schema["inputSchema"]


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = ToolResult.success("Operation completed")

        assert result.content == "Operation completed"
        assert result.is_error is False

    def test_error_result(self):
        """Test creating error result."""
        result = ToolResult.error("Something went wrong")

        assert result.is_error is True
        assert result.error_message == "Something went wrong"


# =============================================================================
# Test Prompt Definition
# =============================================================================
class TestPromptMessage:
    """Tests for PromptMessage."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = PromptMessage(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_message_to_dict(self):
        """Test converting to dict."""
        msg = PromptMessage("assistant", "Hi there!")

        result = msg.to_dict()

        assert result["role"] == "assistant"
        assert result["content"] == "Hi there!"


class TestPrompt:
    """Tests for Prompt."""

    def test_prompt_creation(self):
        """Test creating a prompt."""
        prompt = Prompt(
            name="code-review",
            description="Review code",
            arguments=[],
            messages=[PromptMessage("user", "Review this code")],
        )

        assert prompt.name == "code-review"
        assert len(prompt.messages) == 1

    def test_prompt_render(self):
        """Test rendering prompt with arguments."""
        prompt = Prompt(
            name="greet",
            description="Greet user",
            arguments=[ToolParameter("name", "string", "Name", True)],
            messages=[PromptMessage("user", "Hello, {name}!")],
        )

        rendered = prompt.render({"name": "Alice"})

        assert rendered[0].content == "Hello, Alice!"


# =============================================================================
# Test Registries
# =============================================================================
class TestResourceRegistry:
    """Tests for ResourceRegistry."""

    def test_register_resource(self):
        """Test registering a resource."""
        registry = ResourceRegistry()
        resource = Resource("test://data", "Test")

        registry.register(resource, lambda: "data")

        assert registry.has_resource("test://data")

    def test_list_resources(self):
        """Test listing resources."""
        registry = ResourceRegistry()
        registry.register(Resource("a://", "A"), lambda: "a")
        registry.register(Resource("b://", "B"), lambda: "b")

        resources = registry.list_resources()

        assert len(resources) == 2

    def test_read_resource(self):
        """Test reading a resource."""
        registry = ResourceRegistry()
        registry.register(Resource("test://hello", "Hello"), lambda: "Hello, World!")

        content = registry.read_resource("test://hello")

        assert content.text == "Hello, World!"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = Tool("echo", "Echo message", [])

        registry.register(tool, lambda args: args.get("message", ""))

        assert registry.has_tool("echo")

    def test_call_tool(self):
        """Test calling a tool."""
        registry = ToolRegistry()
        tool = Tool("add", "Add numbers", [])
        registry.register(
            tool, lambda args: ToolResult.success(str(args["a"] + args["b"]))
        )

        result = registry.call_tool("add", {"a": 2, "b": 3})

        assert result.content == "5"


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_register_prompt(self):
        """Test registering a prompt."""
        registry = PromptRegistry()
        prompt = Prompt("test", "Test prompt", [], [])

        registry.register(prompt)

        assert registry.has_prompt("test")

    def test_get_prompt(self):
        """Test getting a prompt."""
        registry = PromptRegistry()
        prompt = Prompt("greet", "Greet", [], [PromptMessage("user", "Hi {name}")])
        registry.register(prompt)

        messages = registry.get_prompt("greet", {"name": "World"})

        assert messages[0].content == "Hi World"


# =============================================================================
# Test Simple MCP Server
# =============================================================================
class TestSimpleMCPServer:
    """Tests for SimpleMCPServer."""

    def test_server_creation(self):
        """Test creating a server."""
        server = SimpleMCPServer("test-server", "1.0.0")

        assert server.name == "test-server"
        assert server.version == "1.0.0"

    def test_add_resource(self):
        """Test adding a resource."""
        server = SimpleMCPServer("test")
        server.add_resource(
            uri="config://app",
            name="Config",
            handler=lambda: '{"key": "value"}',
            description="App config",
        )

        caps = server.get_capabilities()
        assert caps["resources"]

    def test_add_tool(self):
        """Test adding a tool."""
        server = SimpleMCPServer("test")
        server.add_tool(
            name="echo",
            description="Echo a message",
            parameters=[],
            handler=lambda args: ToolResult.success(args.get("msg", "")),
        )

        caps = server.get_capabilities()
        assert caps["tools"]

    def test_get_capabilities(self):
        """Test getting server capabilities."""
        server = SimpleMCPServer("test")

        caps = server.get_capabilities()

        assert "name" in caps
        assert "version" in caps
        assert "resources" in caps
        assert "tools" in caps
        assert "prompts" in caps


# =============================================================================
# Test Data Store Server
# =============================================================================
class TestDataStoreServer:
    """Tests for DataStoreServer."""

    def test_server_creation(self):
        """Test creating data store server."""
        server = DataStoreServer()

        assert server.name == "data-store"

    def test_add_item_tool(self):
        """Test add_item tool exists."""
        server = DataStoreServer()

        tools = server.tools.list_tools()
        tool_names = [t.name for t in tools]

        assert "add_item" in tool_names

    def test_data_operations(self):
        """Test basic data operations."""
        server = DataStoreServer()

        # Add item
        result = server.tools.call_tool(
            "add_item", {"name": "test", "value": {"key": "value"}}
        )

        assert not result.is_error


# =============================================================================
# Test Server Builder
# =============================================================================
class TestMCPServerBuilder:
    """Tests for MCPServerBuilder."""

    def test_builder_chain(self):
        """Test fluent builder pattern."""
        builder = MCPServerBuilder()

        result = builder.with_name("test").with_version("2.0")

        assert result is builder  # Should return self

    def test_build_server(self):
        """Test building a server."""
        server = (
            MCPServerBuilder().with_name("built-server").with_version("1.0.0").build()
        )

        assert server.name == "built-server"
        assert server.version == "1.0.0"

    def test_build_with_resource(self):
        """Test building server with resource."""
        server = (
            MCPServerBuilder()
            .with_name("test")
            .with_resource("test://data", "Data", lambda: "test")
            .build()
        )

        assert server.resources.has_resource("test://data")

    def test_build_with_tool(self):
        """Test building server with tool."""

        def handler(args):
            return ToolResult.success("ok")

        server = (
            MCPServerBuilder()
            .with_name("test")
            .with_tool("test_tool", handler, "A test tool", [])
            .build()
        )

        assert server.tools.has_tool("test_tool")

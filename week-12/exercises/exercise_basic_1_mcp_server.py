"""
Week 12 - Exercise 1 (Basic): MCP Server Fundamentals
=====================================================

Learn to build MCP servers with resources, tools, and prompts.

Concepts:
- MCP server architecture
- Defining resources
- Implementing tools
- Creating prompts
- Server lifecycle
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import json


# =============================================================================
# TASK 1: Resource Definition
# =============================================================================
@dataclass
class Resource:
    """
    Define an MCP resource.

    Resources are data that AI can read. Each resource has:
    - uri: Unique identifier (e.g., "file:///config.json")
    - name: Human-readable name
    - description: What the resource contains
    - mime_type: Content type (default: "text/plain")

    TODO: Implement the Resource dataclass with:
    - uri (str): The resource URI
    - name (str): Display name
    - description (str, optional): Resource description
    - mime_type (str): MIME type, default "text/plain"
    - to_dict() method that returns resource as dict
    """

    # TODO: Add fields and implement to_dict
    pass


@dataclass
class ResourceContent:
    """
    Content returned when reading a resource.

    TODO: Implement with:
    - uri (str): The resource URI
    - mime_type (str): Content MIME type
    - text (str, optional): Text content
    - blob (bytes, optional): Binary content
    - is_binary property that returns True if blob is set
    """

    # TODO: Add fields and is_binary property
    pass


# =============================================================================
# TASK 2: Tool Definition
# =============================================================================
@dataclass
class ToolParameter:
    """
    A parameter for an MCP tool.

    TODO: Implement with:
    - name (str): Parameter name
    - param_type (str): JSON schema type ("string", "number", etc.)
    - description (str): Parameter description
    - required (bool): Whether parameter is required
    - default (Any, optional): Default value
    - to_schema() method that returns JSON schema dict
    """

    # TODO: Implement fields and to_schema method
    pass


@dataclass
class Tool:
    """
    Define an MCP tool.

    Tools are functions that AI can call. Each tool has:
    - name: Unique identifier
    - description: What the tool does
    - parameters: List of ToolParameter objects
    - handler: The function to call

    TODO: Implement with:
    - name (str): Tool name
    - description (str): Tool description
    - parameters (list[ToolParameter]): Tool parameters
    - to_schema() method that returns full JSON schema
    """

    # TODO: Implement fields and to_schema method
    pass


# =============================================================================
# TASK 3: Tool Result
# =============================================================================
@dataclass
class ToolResult:
    """
    Result returned from tool execution.

    TODO: Implement with:
    - content (str): Result content
    - is_error (bool): Whether this is an error result
    - error_message (str, optional): Error details if is_error

    Implement class methods:
    - success(content: str) -> ToolResult
    - error(message: str) -> ToolResult
    """

    # TODO: Implement fields and class methods
    pass


# =============================================================================
# TASK 4: Prompt Definition
# =============================================================================
@dataclass
class PromptMessage:
    """
    A message in a prompt template.

    TODO: Implement with:
    - role (str): "user" or "assistant"
    - content (str): Message content
    - to_dict() method
    """

    # TODO: Implement
    pass


@dataclass
class Prompt:
    """
    Define an MCP prompt template.

    TODO: Implement with:
    - name (str): Prompt name
    - description (str): What the prompt is for
    - arguments (list[ToolParameter]): Required arguments
    - messages (list[PromptMessage]): Template messages
    - render(args: dict) method that substitutes {arg_name} in messages
    """

    # TODO: Implement
    pass


# =============================================================================
# TASK 5: Resource Registry
# =============================================================================
class ResourceRegistry:
    """
    Registry for managing MCP resources.

    TODO: Implement with:
    - register(resource: Resource, handler: callable) - Register a resource
    - list_resources() -> list[Resource] - List all resources
    - read_resource(uri: str) -> ResourceContent - Read a resource
    - has_resource(uri: str) -> bool - Check if resource exists
    - unregister(uri: str) - Remove a resource

    The handler is a function that returns the resource content.
    """

    def __init__(self):
        # TODO: Initialize storage
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 6: Tool Registry
# =============================================================================
class ToolRegistry:
    """
    Registry for managing MCP tools.

    TODO: Implement with:
    - register(tool: Tool, handler: callable) - Register a tool
    - list_tools() -> list[Tool] - List all tools
    - call_tool(name: str, arguments: dict) -> ToolResult - Execute a tool
    - has_tool(name: str) -> bool - Check if tool exists
    - get_tool(name: str) -> Tool - Get tool definition
    """

    def __init__(self):
        # TODO: Initialize storage
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 7: Prompt Registry
# =============================================================================
class PromptRegistry:
    """
    Registry for managing MCP prompts.

    TODO: Implement with:
    - register(prompt: Prompt) - Register a prompt
    - list_prompts() -> list[Prompt] - List all prompts
    - get_prompt(name: str, arguments: dict) -> list[PromptMessage] - Get rendered prompt
    - has_prompt(name: str) -> bool - Check if prompt exists
    """

    def __init__(self):
        # TODO: Initialize storage
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 8: Simple MCP Server
# =============================================================================
class SimpleMCPServer:
    """
    A simple MCP server combining resources, tools, and prompts.

    TODO: Implement with:
    - name (str): Server name
    - version (str): Server version
    - resources: ResourceRegistry
    - tools: ToolRegistry
    - prompts: PromptRegistry

    Methods:
    - add_resource(uri, name, handler, description, mime_type)
    - add_tool(name, description, parameters, handler)
    - add_prompt(name, description, arguments, messages)
    - get_capabilities() -> dict describing server capabilities
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        # TODO: Initialize
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 9: Data Store Server
# =============================================================================
class DataStoreServer(SimpleMCPServer):
    """
    An MCP server that provides access to a simple data store.

    TODO: Extend SimpleMCPServer to create a data store with:

    Resources:
    - "data://items" - List all items
    - "data://item/{id}" - Get specific item

    Tools:
    - "add_item" - Add an item (name: str, value: any)
    - "update_item" - Update an item (id: str, value: any)
    - "delete_item" - Delete an item (id: str)
    - "search_items" - Search items (query: str)

    The data store should be an in-memory dict.
    """

    def __init__(self, name: str = "data-store"):
        # TODO: Initialize parent and set up resources/tools
        pass

    # TODO: Implement data store logic
    pass


# =============================================================================
# TASK 10: Server Builder
# =============================================================================
class MCPServerBuilder:
    """
    Builder pattern for creating MCP servers.

    TODO: Implement fluent builder with:
    - with_name(name: str) -> self
    - with_version(version: str) -> self
    - with_resource(uri, name, handler, **kwargs) -> self
    - with_tool(name, handler, description, parameters) -> self
    - with_prompt(name, description, messages) -> self
    - build() -> SimpleMCPServer

    Example usage:
        server = (MCPServerBuilder()
            .with_name("my-server")
            .with_resource("config://app", "Config", get_config)
            .with_tool("greet", greet_handler, "Greet user", [...])
            .build())
    """

    def __init__(self):
        # TODO: Initialize builder state
        pass

    # TODO: Implement builder methods
    pass


# =============================================================================
# Example Usage and Testing
# =============================================================================
if __name__ == "__main__":
    print("Week 12 - Exercise 1: MCP Server Fundamentals")
    print("=" * 50)

    # Test Resource
    print("\n1. Resource Definition:")
    # resource = Resource(
    #     uri="file:///config.json",
    #     name="Configuration",
    #     description="App configuration"
    # )
    # print(f"   Resource: {resource.to_dict()}")

    # Test Tool
    print("\n2. Tool Definition:")
    # param = ToolParameter(
    #     name="message",
    #     param_type="string",
    #     description="Message to echo",
    #     required=True
    # )
    # tool = Tool(
    #     name="echo",
    #     description="Echo a message",
    #     parameters=[param]
    # )
    # print(f"   Tool schema: {tool.to_schema()}")

    # Test Simple Server
    print("\n3. Simple MCP Server:")
    # server = SimpleMCPServer("test-server")
    # server.add_resource(
    #     uri="test://hello",
    #     name="Hello",
    #     handler=lambda: "Hello, World!",
    #     description="A greeting"
    # )
    # print(f"   Capabilities: {server.get_capabilities()}")

    print("\nImplement the TODO sections to complete this exercise!")

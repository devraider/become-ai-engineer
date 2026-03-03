"""
Week 12 - Exercise 2 (Intermediate): MCP Client Implementation
==============================================================

Learn to build MCP clients that connect to servers.

Concepts:
- Client-server communication
- Protocol handling
- Tool invocation
- Resource reading
- Error handling
"""

from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import asyncio


# =============================================================================
# TASK 1: Message Types
# =============================================================================
class MessageType(Enum):
    """MCP message types."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPMessage:
    """
    Base MCP message structure (JSON-RPC style).

    TODO: Implement with:
    - jsonrpc (str): Always "2.0"
    - id (Optional[str]): Message ID (None for notifications)
    - method (Optional[str]): Method name (for requests)
    - params (Optional[dict]): Method parameters
    - result (Optional[Any]): Result (for responses)
    - error (Optional[dict]): Error info (for errors)

    Methods:
    - is_request() -> bool
    - is_response() -> bool
    - is_error() -> bool
    - to_json() -> str
    - from_json(json_str) -> MCPMessage (class method)
    """

    # TODO: Implement
    pass


# =============================================================================
# TASK 2: Request Builder
# =============================================================================
class MCPRequestBuilder:
    """
    Build MCP request messages.

    TODO: Implement with:
    - _id_counter (class variable for auto-incrementing IDs)

    Class methods:
    - initialize() -> MCPMessage - Create initialize request
    - list_resources() -> MCPMessage
    - read_resource(uri: str) -> MCPMessage
    - list_tools() -> MCPMessage
    - call_tool(name: str, arguments: dict) -> MCPMessage
    - list_prompts() -> MCPMessage
    - get_prompt(name: str, arguments: dict) -> MCPMessage
    """

    _id_counter: int = 0

    # TODO: Implement class methods
    pass


# =============================================================================
# TASK 3: Response Parser
# =============================================================================
@dataclass
class ParsedResponse:
    """Parsed response with typed data."""

    success: bool
    data: Any
    error_message: Optional[str] = None
    error_code: Optional[int] = None


class MCPResponseParser:
    """
    Parse MCP response messages.

    TODO: Implement with static methods:
    - parse(message: MCPMessage) -> ParsedResponse
    - parse_resources(message: MCPMessage) -> list[dict]
    - parse_tools(message: MCPMessage) -> list[dict]
    - parse_tool_result(message: MCPMessage) -> dict
    - parse_resource_content(message: MCPMessage) -> dict
    """

    # TODO: Implement static methods
    pass


# =============================================================================
# TASK 4: Transport Interface
# =============================================================================
class MCPTransport(ABC):
    """
    Abstract transport layer for MCP communication.

    TODO: Define abstract methods:
    - connect() -> None
    - disconnect() -> None
    - send(message: MCPMessage) -> None
    - receive() -> MCPMessage
    - is_connected() -> bool
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    # TODO: Define other abstract methods
    pass


# =============================================================================
# TASK 5: Mock Transport (for testing)
# =============================================================================
class MockTransport(MCPTransport):
    """
    Mock transport for testing without a real server.

    TODO: Implement with:
    - _connected (bool): Connection state
    - _responses (dict): Mapping of method -> response data
    - _call_history (list): Record of sent messages

    Methods:
    - add_response(method: str, response: Any) - Add mock response
    - get_call_history() -> list[MCPMessage]
    - clear_history() -> None
    """

    def __init__(self):
        # TODO: Initialize
        pass

    # TODO: Implement transport methods
    pass


# =============================================================================
# TASK 6: MCP Client
# =============================================================================
class MCPClient:
    """
    MCP client for communicating with servers.

    TODO: Implement with:
    - transport (MCPTransport): The transport layer
    - _initialized (bool): Whether client is initialized
    - _server_info (dict): Server capabilities after init

    Methods:
    - connect() -> None - Connect and initialize
    - disconnect() -> None
    - list_resources() -> list[dict]
    - read_resource(uri: str) -> dict
    - list_tools() -> list[dict]
    - call_tool(name: str, arguments: dict) -> dict
    - list_prompts() -> list[dict]
    - get_prompt(name: str, arguments: dict) -> list[dict]
    - get_server_info() -> dict
    """

    def __init__(self, transport: MCPTransport):
        # TODO: Initialize
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 7: Client with Retry Logic
# =============================================================================
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


class ResilientMCPClient(MCPClient):
    """
    MCP client with retry and error handling.

    TODO: Extend MCPClient with:
    - retry_config (RetryConfig): Retry settings

    Override methods to add retry logic:
    - _with_retry(operation: callable) -> Any
    - call_tool() - With retry
    - read_resource() - With retry

    Add methods:
    - _calculate_delay(attempt: int) -> float
    - _should_retry(error: Exception) -> bool
    """

    def __init__(
        self, transport: MCPTransport, retry_config: Optional[RetryConfig] = None
    ):
        # TODO: Initialize
        pass

    # TODO: Implement retry logic
    pass


# =============================================================================
# TASK 8: Multi-Server Client
# =============================================================================
@dataclass
class ServerConnection:
    """Information about a connected server."""

    name: str
    transport: MCPTransport
    client: MCPClient
    connected: bool = False
    capabilities: dict = field(default_factory=dict)


class MultiServerClient:
    """
    Client that manages connections to multiple MCP servers.

    TODO: Implement with:
    - _servers (dict[str, ServerConnection]): Connected servers

    Methods:
    - add_server(name: str, transport: MCPTransport) -> None
    - remove_server(name: str) -> None
    - connect_all() -> dict[str, bool] (server -> success)
    - disconnect_all() -> None
    - list_all_tools() -> dict[str, list[dict]]
    - call_tool(server: str, tool: str, arguments: dict) -> dict
    - find_tool(tool_name: str) -> Optional[str] (returns server name)
    - get_server_status() -> dict[str, dict]
    """

    def __init__(self):
        # TODO: Initialize
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 9: Client Pool
# =============================================================================
class ClientPool:
    """
    Pool of reusable MCP clients for efficiency.

    TODO: Implement with:
    - _pool (dict[str, list[MCPClient]]): Available clients by server
    - _in_use (dict[str, list[MCPClient]]): Clients currently in use
    - _max_per_server (int): Max clients per server
    - _transport_factory (callable): Factory for creating transports

    Methods:
    - acquire(server: str) -> MCPClient - Get a client
    - release(server: str, client: MCPClient) -> None - Return client
    - close_all() -> None

    Context manager support:
    - client(server: str) -> context manager that handles acquire/release
    """

    def __init__(
        self, transport_factory: Callable[[str], MCPTransport], max_per_server: int = 5
    ):
        # TODO: Initialize
        pass

    # TODO: Implement methods
    pass


# =============================================================================
# TASK 10: Client with Caching
# =============================================================================
@dataclass
class CacheEntry:
    """A cached response."""

    data: Any
    timestamp: float
    ttl: float  # Time to live in seconds


class CachingMCPClient(MCPClient):
    """
    MCP client with response caching.

    TODO: Extend MCPClient with:
    - _cache (dict[str, CacheEntry]): Cached responses
    - _default_ttl (float): Default TTL in seconds

    Override methods to add caching:
    - list_resources() - Cache with TTL
    - list_tools() - Cache with TTL
    - read_resource() - Cache with TTL

    Add methods:
    - _get_cached(key: str) -> Optional[Any]
    - _set_cached(key: str, data: Any, ttl: Optional[float]) -> None
    - _is_expired(entry: CacheEntry) -> bool
    - clear_cache() -> None
    - get_cache_stats() -> dict
    """

    def __init__(self, transport: MCPTransport, default_ttl: float = 300.0):
        # TODO: Initialize
        pass

    # TODO: Implement caching
    pass


# =============================================================================
# Example Usage and Testing
# =============================================================================
if __name__ == "__main__":
    print("Week 12 - Exercise 2: MCP Client Implementation")
    print("=" * 50)

    # Test Message
    print("\n1. MCP Message:")
    # msg = MCPMessage(
    #     jsonrpc="2.0",
    #     id="1",
    #     method="tools/list",
    #     params={}
    # )
    # print(f"   Message: {msg.to_json()}")

    # Test Request Builder
    print("\n2. Request Builder:")
    # request = MCPRequestBuilder.list_tools()
    # print(f"   Request: {request.to_json()}")

    # Test Mock Transport
    print("\n3. Mock Transport:")
    # transport = MockTransport()
    # transport.add_response("tools/list", {"tools": [{"name": "test"}]})
    # async def test():
    #     await transport.connect()
    #     print(f"   Connected: {transport.is_connected()}")
    # asyncio.run(test())

    # Test Client
    print("\n4. MCP Client:")
    # transport = MockTransport()
    # client = MCPClient(transport)
    # async def test_client():
    #     await client.connect()
    #     tools = await client.list_tools()
    #     print(f"   Tools: {tools}")
    # asyncio.run(test_client())

    print("\nImplement the TODO sections to complete this exercise!")

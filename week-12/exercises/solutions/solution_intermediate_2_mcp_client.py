"""
Week 12 - Exercise 2 (Intermediate): MCP Client Implementation - SOLUTIONS

Complete implementations for MCP client components.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from abc import ABC, abstractmethod
from enum import Enum
import json
import asyncio
import time
import uuid
import hashlib


# =============================================================================
# Part 1: Message Types
# =============================================================================
class MessageType(str, Enum):
    """Types of MCP messages."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPMessage:
    """
    Represents an MCP JSON-RPC message.

    Attributes:
        jsonrpc: JSON-RPC version (always "2.0")
        id: Message ID (optional for notifications)
        method: Method name (for requests)
        params: Method parameters (for requests)
        result: Result data (for responses)
        error: Error data (for error responses)
    """

    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[dict] = None

    def is_request(self) -> bool:
        """Check if message is a request."""
        return self.method is not None and self.id is not None

    def is_response(self) -> bool:
        """Check if message is a response."""
        return self.result is not None and self.method is None

    def is_notification(self) -> bool:
        """Check if message is a notification."""
        return self.method is not None and self.id is None

    def is_error(self) -> bool:
        """Check if message is an error response."""
        return self.error is not None

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        result = {"jsonrpc": self.jsonrpc}

        if self.id is not None:
            result["id"] = self.id
        if self.method is not None:
            result["method"] = self.method
        if self.params is not None:
            result["params"] = self.params
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = self.error

        return result

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "MCPMessage":
        """Deserialize message from JSON string."""
        parsed = json.loads(data)
        return cls(
            jsonrpc=parsed.get("jsonrpc", "2.0"),
            id=parsed.get("id"),
            method=parsed.get("method"),
            params=parsed.get("params"),
            result=parsed.get("result"),
            error=parsed.get("error"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "MCPMessage":
        """Create message from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


# =============================================================================
# Part 2: Request Builder
# =============================================================================
class MCPRequestBuilder:
    """
    Factory for creating MCP request messages.
    """

    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> str:
        """Generate next request ID."""
        cls._id_counter += 1
        return str(cls._id_counter)

    @classmethod
    def initialize(
        cls, client_name: str = "mcp-client", client_version: str = "1.0.0"
    ) -> MCPMessage:
        """Create an initialize request."""
        return MCPMessage(
            id=cls._next_id(),
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": client_name, "version": client_version},
                "capabilities": {},
            },
        )

    @classmethod
    def list_resources(cls, cursor: Optional[str] = None) -> MCPMessage:
        """Create a list_resources request."""
        params = {}
        if cursor:
            params["cursor"] = cursor

        return MCPMessage(id=cls._next_id(), method="resources/list", params=params)

    @classmethod
    def read_resource(cls, uri: str) -> MCPMessage:
        """Create a read_resource request."""
        return MCPMessage(
            id=cls._next_id(), method="resources/read", params={"uri": uri}
        )

    @classmethod
    def subscribe_resource(cls, uri: str) -> MCPMessage:
        """Create a subscribe request for a resource."""
        return MCPMessage(
            id=cls._next_id(), method="resources/subscribe", params={"uri": uri}
        )

    @classmethod
    def list_tools(cls, cursor: Optional[str] = None) -> MCPMessage:
        """Create a list_tools request."""
        params = {}
        if cursor:
            params["cursor"] = cursor

        return MCPMessage(id=cls._next_id(), method="tools/list", params=params)

    @classmethod
    def call_tool(cls, name: str, arguments: dict) -> MCPMessage:
        """Create a call_tool request."""
        return MCPMessage(
            id=cls._next_id(),
            method="tools/call",
            params={"name": name, "arguments": arguments},
        )

    @classmethod
    def list_prompts(cls, cursor: Optional[str] = None) -> MCPMessage:
        """Create a list_prompts request."""
        params = {}
        if cursor:
            params["cursor"] = cursor

        return MCPMessage(id=cls._next_id(), method="prompts/list", params=params)

    @classmethod
    def get_prompt(cls, name: str, arguments: dict) -> MCPMessage:
        """Create a get_prompt request."""
        return MCPMessage(
            id=cls._next_id(),
            method="prompts/get",
            params={"name": name, "arguments": arguments},
        )

    @classmethod
    def custom(cls, method: str, params: Optional[dict] = None) -> MCPMessage:
        """Create a custom request."""
        return MCPMessage(id=cls._next_id(), method=method, params=params or {})


# =============================================================================
# Part 3: Response Parser
# =============================================================================
@dataclass
class ParsedResponse:
    """Parsed response data."""

    success: bool
    data: Optional[Any] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    raw: Optional[MCPMessage] = None


class MCPResponseParser:
    """
    Parser for MCP response messages.
    """

    @classmethod
    def parse(cls, message: MCPMessage) -> ParsedResponse:
        """Parse a response message."""
        if message.is_error():
            error = message.error or {}
            return ParsedResponse(
                success=False,
                error_code=error.get("code"),
                error_message=error.get("message"),
                raw=message,
            )

        return ParsedResponse(success=True, data=message.result, raw=message)

    @classmethod
    def parse_resources(cls, message: MCPMessage) -> list[dict]:
        """Parse a resources/list response."""
        if message.result is None:
            return []
        return message.result.get("resources", [])

    @classmethod
    def parse_tools(cls, message: MCPMessage) -> list[dict]:
        """Parse a tools/list response."""
        if message.result is None:
            return []
        return message.result.get("tools", [])

    @classmethod
    def parse_prompts(cls, message: MCPMessage) -> list[dict]:
        """Parse a prompts/list response."""
        if message.result is None:
            return []
        return message.result.get("prompts", [])

    @classmethod
    def parse_tool_result(cls, message: MCPMessage) -> dict:
        """Parse a tools/call response."""
        if message.result is None:
            return {"content": [], "isError": True}
        return message.result

    @classmethod
    def parse_resource_content(cls, message: MCPMessage) -> list[dict]:
        """Parse a resources/read response."""
        if message.result is None:
            return []
        return message.result.get("contents", [])


# =============================================================================
# Part 4: Transport Abstraction
# =============================================================================
class MCPTransport(ABC):
    """
    Abstract base class for MCP transport implementations.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self) -> MCPMessage:
        """Receive a message."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass


# =============================================================================
# Part 5: Mock Transport for Testing
# =============================================================================
class MockTransport(MCPTransport):
    """
    Mock transport for testing MCP clients.
    """

    def __init__(self):
        self._connected: bool = False
        self._responses: dict[str, dict] = {}
        self._pending_responses: list[MCPMessage] = []
        self._call_history: list[MCPMessage] = []

    async def connect(self) -> None:
        """Establish mock connection."""
        self._connected = True

    async def disconnect(self) -> None:
        """Close mock connection."""
        self._connected = False

    async def send(self, message: MCPMessage) -> None:
        """Send a message (record for testing)."""
        if not self._connected:
            raise ConnectionError("Not connected")

        self._call_history.append(message)

        # Generate response if we have one configured
        if message.method and message.method in self._responses:
            response = MCPMessage(id=message.id, result=self._responses[message.method])
            self._pending_responses.append(response)

    async def receive(self) -> MCPMessage:
        """Receive a message."""
        if not self._connected:
            raise ConnectionError("Not connected")

        while not self._pending_responses:
            await asyncio.sleep(0.01)

        return self._pending_responses.pop(0)

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def add_response(self, method: str, result: dict) -> None:
        """Add a mock response for a method."""
        self._responses[method] = result

    def get_call_history(self) -> list[MCPMessage]:
        """Get history of sent messages."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()


# =============================================================================
# Part 6: MCP Client
# =============================================================================
class MCPClient:
    """
    MCP client for communicating with servers.
    """

    def __init__(self, transport: MCPTransport):
        self._transport = transport
        self._server_info: Optional[dict] = None
        self._capabilities: dict = {}

    async def connect(
        self, client_name: str = "mcp-client", client_version: str = "1.0.0"
    ) -> dict:
        """Connect and initialize with the server."""
        await self._transport.connect()

        request = MCPRequestBuilder.initialize(client_name, client_version)
        response = await self._send_request(request)

        if response.success:
            self._server_info = response.data.get("serverInfo", {})
            self._capabilities = response.data.get("capabilities", {})

        return response.data

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        await self._transport.disconnect()
        self._server_info = None

    async def _send_request(self, request: MCPMessage) -> ParsedResponse:
        """Send request and wait for response."""
        await self._transport.send(request)
        response = await self._transport.receive()
        return MCPResponseParser.parse(response)

    async def list_resources(self) -> list[dict]:
        """List available resources."""
        request = MCPRequestBuilder.list_resources()
        response = await self._send_request(request)

        if response.success:
            return response.data.get("resources", [])
        return []

    async def read_resource(self, uri: str) -> list[dict]:
        """Read a resource."""
        request = MCPRequestBuilder.read_resource(uri)
        response = await self._send_request(request)

        if response.success:
            return response.data.get("contents", [])
        return []

    async def list_tools(self) -> list[dict]:
        """List available tools."""
        request = MCPRequestBuilder.list_tools()
        response = await self._send_request(request)

        if response.success:
            return response.data.get("tools", [])
        return []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool."""
        request = MCPRequestBuilder.call_tool(name, arguments)
        response = await self._send_request(request)

        if response.success:
            return response.data
        return {"content": [], "isError": True}

    async def list_prompts(self) -> list[dict]:
        """List available prompts."""
        request = MCPRequestBuilder.list_prompts()
        response = await self._send_request(request)

        if response.success:
            return response.data.get("prompts", [])
        return []

    async def get_prompt(self, name: str, arguments: dict) -> dict:
        """Get a rendered prompt."""
        request = MCPRequestBuilder.get_prompt(name, arguments)
        response = await self._send_request(request)

        if response.success:
            return response.data
        return {"messages": []}

    def get_server_info(self) -> Optional[dict]:
        """Get server information."""
        return self._server_info

    def get_capabilities(self) -> dict:
        """Get server capabilities."""
        return self._capabilities


# =============================================================================
# Part 7: Resilient Client with Retry
# =============================================================================
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: set = field(default_factory=lambda: {-32000, -32001, -32002})


class ResilientMCPClient(MCPClient):
    """
    MCP client with retry and error handling.
    """

    def __init__(self, transport: MCPTransport, config: Optional[RetryConfig] = None):
        super().__init__(transport)
        self._config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self._config.base_delay * (self._config.exponential_base**attempt)
        return min(delay, self._config.max_delay)

    def _should_retry(self, error: Optional[dict], attempt: int) -> bool:
        """Determine if request should be retried."""
        if attempt >= self._config.max_retries:
            return False

        if error is None:
            return True

        error_code = error.get("code", 0)
        return error_code in self._config.retryable_errors

    async def _send_request(self, request: MCPMessage) -> ParsedResponse:
        """Send request with retry logic."""
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            try:
                await self._transport.send(request)
                response = await self._transport.receive()
                parsed = MCPResponseParser.parse(response)

                if parsed.success or not self._should_retry(response.error, attempt):
                    return parsed

                last_error = response.error

            except Exception as e:
                last_error = {"code": -32000, "message": str(e)}

                if attempt >= self._config.max_retries:
                    break

            delay = self._calculate_delay(attempt)
            await asyncio.sleep(delay)

        return ParsedResponse(
            success=False,
            error_code=last_error.get("code") if last_error else -32000,
            error_message=last_error.get("message") if last_error else "Unknown error",
        )

    async def reconnect(self) -> bool:
        """Attempt to reconnect to the server."""
        try:
            await self._transport.disconnect()
        except Exception:
            pass

        try:
            await self.connect()
            return True
        except Exception:
            return False


# =============================================================================
# Part 8: Multi-Server Client
# =============================================================================
@dataclass
class ServerConnection:
    """Information about a server connection."""

    name: str
    transport: MCPTransport
    client: Optional[MCPClient] = None
    server_info: Optional[dict] = None
    connected: bool = False


class MultiServerClient:
    """
    Client that can connect to multiple MCP servers.
    """

    def __init__(self):
        self._servers: dict[str, ServerConnection] = {}

    def add_server(
        self, name: str, transport: MCPTransport, auto_connect: bool = False
    ) -> None:
        """Add a server configuration."""
        self._servers[name] = ServerConnection(name=name, transport=transport)

    def remove_server(self, name: str) -> None:
        """Remove a server configuration."""
        if name in self._servers:
            del self._servers[name]

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all configured servers."""
        results = {}

        for name, conn in self._servers.items():
            try:
                client = MCPClient(conn.transport)
                await client.connect()
                conn.client = client
                conn.server_info = client.get_server_info()
                conn.connected = True
                results[name] = True
            except Exception:
                conn.connected = False
                results[name] = False

        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for conn in self._servers.values():
            if conn.client and conn.connected:
                await conn.client.disconnect()
                conn.connected = False

    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get client for a specific server."""
        conn = self._servers.get(name)
        if conn and conn.connected:
            return conn.client
        return None

    def list_servers(self) -> list[dict]:
        """List all configured servers."""
        return [
            {"name": name, "connected": conn.connected, "server_info": conn.server_info}
            for name, conn in self._servers.items()
        ]

    async def list_all_tools(self) -> dict[str, list[dict]]:
        """List tools from all connected servers."""
        results = {}

        for name, conn in self._servers.items():
            if conn.client and conn.connected:
                tools = await conn.client.list_tools()
                results[name] = tools

        return results

    async def find_tool(self, tool_name: str) -> Optional[tuple[str, MCPClient]]:
        """Find which server has a specific tool."""
        for name, conn in self._servers.items():
            if conn.client and conn.connected:
                tools = await conn.client.list_tools()
                for tool in tools:
                    if tool.get("name") == tool_name:
                        return (name, conn.client)
        return None

    async def call_tool_anywhere(
        self, tool_name: str, arguments: dict
    ) -> Optional[dict]:
        """Call a tool on whichever server has it."""
        result = await self.find_tool(tool_name)
        if result:
            _, client = result
            return await client.call_tool(tool_name, arguments)
        return None


# =============================================================================
# Part 9: Client Pool
# =============================================================================
class ClientPool:
    """
    Pool of MCP clients for concurrent access.
    """

    def __init__(
        self, transport_factory: Callable[[str], MCPTransport], max_per_server: int = 5
    ):
        self._factory = transport_factory
        self._max_per_server = max_per_server
        self._pools: dict[str, list[MCPClient]] = {}
        self._available: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def _ensure_pool(self, server: str) -> None:
        """Ensure pool exists for server."""
        if server not in self._pools:
            self._pools[server] = []
            self._available[server] = asyncio.Queue()

    async def _create_client(self, server: str) -> MCPClient:
        """Create a new client for the pool."""
        transport = self._factory(server)
        client = MCPClient(transport)
        await client.connect()
        return client

    async def acquire(self, server: str) -> MCPClient:
        """Acquire a client from the pool."""
        async with self._lock:
            await self._ensure_pool(server)

        queue = self._available[server]
        pool = self._pools[server]

        if not queue.empty():
            return await queue.get()

        if len(pool) < self._max_per_server:
            client = await self._create_client(server)
            pool.append(client)
            return client

        # Wait for available client
        return await queue.get()

    async def release(self, server: str, client: MCPClient) -> None:
        """Release a client back to the pool."""
        if server in self._available:
            await self._available[server].put(client)

    async def close_all(self) -> None:
        """Close all pooled clients."""
        for server, pool in self._pools.items():
            for client in pool:
                await client.disconnect()

        self._pools.clear()
        self._available.clear()


# =============================================================================
# Part 10: Caching Client Wrapper
# =============================================================================
@dataclass
class CacheEntry:
    """A cached response entry."""

    data: Any
    timestamp: float
    ttl: float

    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - self.timestamp < self.ttl


class CachingMCPClient(MCPClient):
    """
    MCP client with response caching.
    """

    def __init__(
        self,
        transport: MCPTransport,
        default_ttl: float = 300.0,
        cacheable_methods: Optional[set] = None,
    ):
        super().__init__(transport)
        self._default_ttl = default_ttl
        self._cacheable_methods = cacheable_methods or {
            "resources/list",
            "tools/list",
            "prompts/list",
        }
        self._cache: dict[str, CacheEntry] = {}

    def _cache_key(self, method: str, params: dict) -> str:
        """Generate cache key for request."""
        data = json.dumps({"method": method, "params": params}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached response if valid."""
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """Cache a response."""
        self._cache[key] = CacheEntry(
            data=data, timestamp=time.time(), ttl=ttl or self._default_ttl
        )

    async def _send_request(self, request: MCPMessage) -> ParsedResponse:
        """Send request with caching."""
        method = request.method or ""
        params = request.params or {}

        # Check if cacheable
        if method in self._cacheable_methods:
            key = self._cache_key(method, params)
            cached = self._get_cached(key)

            if cached is not None:
                return ParsedResponse(success=True, data=cached)

        # Make actual request
        response = await super()._send_request(request)

        # Cache successful responses
        if response.success and method in self._cacheable_methods:
            key = self._cache_key(method, params)
            self._set_cached(key, response.data)

        return response

    async def list_tools(self) -> list[dict]:
        """List tools (potentially cached)."""
        return await super().list_tools()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def invalidate(self, method: str, params: Optional[dict] = None) -> None:
        """Invalidate specific cached data."""
        if params is None:
            # Invalidate all entries for this method
            keys_to_remove = [
                k for k, v in self._cache.items() if method in k  # Simple check
            ]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            key = self._cache_key(method, params)
            if key in self._cache:
                del self._cache[key]

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        valid_count = sum(1 for e in self._cache.values() if e.is_valid())

        return {
            "entries": len(self._cache),
            "valid_entries": valid_count,
            "expired_entries": len(self._cache) - valid_count,
        }


# =============================================================================
# Example Usage
# =============================================================================
async def main():
    """Demonstrate client usage."""
    # Create mock transport
    transport = MockTransport()
    transport.add_response(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
            "capabilities": {},
        },
    )
    transport.add_response(
        "tools/list",
        {
            "tools": [
                {"name": "echo", "description": "Echo input"},
                {"name": "add", "description": "Add numbers"},
            ]
        },
    )
    transport.add_response(
        "tools/call",
        {"content": [{"type": "text", "text": "Result: Hello"}], "isError": False},
    )

    # Basic client usage
    client = MCPClient(transport)
    await client.connect()

    print("Server info:", client.get_server_info())

    tools = await client.list_tools()
    print("Tools:", tools)

    result = await client.call_tool("echo", {"message": "Hello"})
    print("Tool result:", result)

    await client.disconnect()

    # Multi-server usage
    print("\n--- Multi-Server Example ---")

    multi = MultiServerClient()

    t1 = MockTransport()
    t1.add_response("initialize", {"serverInfo": {"name": "server1"}})
    t1.add_response("tools/list", {"tools": [{"name": "tool1"}]})

    t2 = MockTransport()
    t2.add_response("initialize", {"serverInfo": {"name": "server2"}})
    t2.add_response("tools/list", {"tools": [{"name": "tool2"}]})

    multi.add_server("server1", t1)
    multi.add_server("server2", t2)

    results = await multi.connect_all()
    print("Connection results:", results)

    all_tools = await multi.list_all_tools()
    print("All tools:", all_tools)

    await multi.disconnect_all()

    # Caching client
    print("\n--- Caching Example ---")

    transport2 = MockTransport()
    transport2.add_response("initialize", {"serverInfo": {}})
    transport2.add_response("tools/list", {"tools": [{"name": "cached_tool"}]})

    caching_client = CachingMCPClient(transport2, default_ttl=60.0)
    await caching_client.connect()

    # First call - hits server
    tools1 = await caching_client.list_tools()
    print("First call:", tools1)

    # Second call - hits cache
    tools2 = await caching_client.list_tools()
    print("Second call (cached):", tools2)

    print("Cache stats:", caching_client.get_cache_stats())

    await caching_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

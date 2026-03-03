"""
Tests for Week 12 - Exercise 2 (Intermediate): MCP Client Implementation
"""

import pytest
import asyncio
from exercise_intermediate_2_mcp_client import (
    MessageType,
    MCPMessage,
    MCPRequestBuilder,
    MCPResponseParser,
    ParsedResponse,
    MCPTransport,
    MockTransport,
    MCPClient,
    RetryConfig,
    ResilientMCPClient,
    ServerConnection,
    MultiServerClient,
    ClientPool,
    CacheEntry,
    CachingMCPClient,
)


# =============================================================================
# Test Message Types
# =============================================================================
class TestMCPMessage:
    """Tests for MCPMessage."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = MCPMessage(jsonrpc="2.0", id="1", method="tools/list", params={})

        assert msg.jsonrpc == "2.0"
        assert msg.method == "tools/list"

    def test_is_request(self):
        """Test identifying request messages."""
        msg = MCPMessage(jsonrpc="2.0", id="1", method="tools/list", params={})

        assert msg.is_request()
        assert not msg.is_response()

    def test_is_response(self):
        """Test identifying response messages."""
        msg = MCPMessage(jsonrpc="2.0", id="1", result={"tools": []})

        assert msg.is_response()
        assert not msg.is_request()

    def test_is_error(self):
        """Test identifying error messages."""
        msg = MCPMessage(
            jsonrpc="2.0", id="1", error={"code": -32600, "message": "Invalid request"}
        )

        assert msg.is_error()

    def test_to_json(self):
        """Test serialization to JSON."""
        msg = MCPMessage(jsonrpc="2.0", id="1", method="test", params={"key": "value"})

        json_str = msg.to_json()

        assert "jsonrpc" in json_str
        assert "2.0" in json_str
        assert "test" in json_str

    def test_from_json(self):
        """Test deserialization from JSON."""
        json_str = '{"jsonrpc": "2.0", "id": "1", "result": {"data": "test"}}'

        msg = MCPMessage.from_json(json_str)

        assert msg.jsonrpc == "2.0"
        assert msg.result == {"data": "test"}


# =============================================================================
# Test Request Builder
# =============================================================================
class TestMCPRequestBuilder:
    """Tests for MCPRequestBuilder."""

    def test_initialize_request(self):
        """Test creating initialize request."""
        request = MCPRequestBuilder.initialize()

        assert request.method == "initialize"
        assert request.id is not None

    def test_list_resources_request(self):
        """Test creating list_resources request."""
        request = MCPRequestBuilder.list_resources()

        assert request.method == "resources/list"

    def test_read_resource_request(self):
        """Test creating read_resource request."""
        request = MCPRequestBuilder.read_resource("test://data")

        assert request.method == "resources/read"
        assert request.params["uri"] == "test://data"

    def test_list_tools_request(self):
        """Test creating list_tools request."""
        request = MCPRequestBuilder.list_tools()

        assert request.method == "tools/list"

    def test_call_tool_request(self):
        """Test creating call_tool request."""
        request = MCPRequestBuilder.call_tool("add", {"a": 1, "b": 2})

        assert request.method == "tools/call"
        assert request.params["name"] == "add"
        assert request.params["arguments"] == {"a": 1, "b": 2}

    def test_auto_incrementing_ids(self):
        """Test that request IDs auto-increment."""
        req1 = MCPRequestBuilder.list_tools()
        req2 = MCPRequestBuilder.list_tools()

        assert req1.id != req2.id


# =============================================================================
# Test Response Parser
# =============================================================================
class TestMCPResponseParser:
    """Tests for MCPResponseParser."""

    def test_parse_success(self):
        """Test parsing successful response."""
        msg = MCPMessage(jsonrpc="2.0", id="1", result={"data": "test"})

        result = MCPResponseParser.parse(msg)

        assert result.success is True
        assert result.data == {"data": "test"}

    def test_parse_error(self):
        """Test parsing error response."""
        msg = MCPMessage(
            jsonrpc="2.0", id="1", error={"code": -32600, "message": "Invalid"}
        )

        result = MCPResponseParser.parse(msg)

        assert result.success is False
        assert result.error_code == -32600

    def test_parse_resources(self):
        """Test parsing resource list."""
        msg = MCPMessage(
            jsonrpc="2.0",
            id="1",
            result={"resources": [{"uri": "test://", "name": "Test"}]},
        )

        resources = MCPResponseParser.parse_resources(msg)

        assert len(resources) == 1
        assert resources[0]["uri"] == "test://"

    def test_parse_tools(self):
        """Test parsing tool list."""
        msg = MCPMessage(
            jsonrpc="2.0",
            id="1",
            result={"tools": [{"name": "test", "description": "A test"}]},
        )

        tools = MCPResponseParser.parse_tools(msg)

        assert len(tools) == 1
        assert tools[0]["name"] == "test"


# =============================================================================
# Test Mock Transport
# =============================================================================
class TestMockTransport:
    """Tests for MockTransport."""

    @pytest.mark.asyncio
    async def test_connection(self):
        """Test connecting and disconnecting."""
        transport = MockTransport()

        assert not transport.is_connected()

        await transport.connect()
        assert transport.is_connected()

        await transport.disconnect()
        assert not transport.is_connected()

    @pytest.mark.asyncio
    async def test_add_response(self):
        """Test adding mock responses."""
        transport = MockTransport()
        transport.add_response("tools/list", {"tools": []})

        await transport.connect()

        request = MCPMessage(jsonrpc="2.0", id="1", method="tools/list", params={})
        await transport.send(request)
        response = await transport.receive()

        assert response.result == {"tools": []}

    @pytest.mark.asyncio
    async def test_call_history(self):
        """Test recording call history."""
        transport = MockTransport()
        await transport.connect()

        request = MCPMessage(jsonrpc="2.0", id="1", method="test", params={})
        await transport.send(request)

        history = transport.get_call_history()

        assert len(history) == 1
        assert history[0].method == "test"


# =============================================================================
# Test MCP Client
# =============================================================================
class TestMCPClient:
    """Tests for MCPClient."""

    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport with standard responses."""
        transport = MockTransport()
        transport.add_response(
            "initialize", {"serverInfo": {"name": "test", "version": "1.0"}}
        )
        transport.add_response("resources/list", {"resources": []})
        transport.add_response("tools/list", {"tools": []})
        return transport

    @pytest.mark.asyncio
    async def test_connect(self, mock_transport):
        """Test connecting to server."""
        client = MCPClient(mock_transport)

        await client.connect()

        assert mock_transport.is_connected()

    @pytest.mark.asyncio
    async def test_list_resources(self, mock_transport):
        """Test listing resources."""
        mock_transport.add_response(
            "resources/list", {"resources": [{"uri": "test://", "name": "Test"}]}
        )

        client = MCPClient(mock_transport)
        await client.connect()

        resources = await client.list_resources()

        assert len(resources) == 1

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_transport):
        """Test listing tools."""
        mock_transport.add_response(
            "tools/list", {"tools": [{"name": "echo", "description": "Echo"}]}
        )

        client = MCPClient(mock_transport)
        await client.connect()

        tools = await client.list_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_call_tool(self, mock_transport):
        """Test calling a tool."""
        mock_transport.add_response(
            "tools/call", {"content": [{"type": "text", "text": "result"}]}
        )

        client = MCPClient(mock_transport)
        await client.connect()

        result = await client.call_tool("test", {"arg": "value"})

        assert "content" in result

    @pytest.mark.asyncio
    async def test_get_server_info(self, mock_transport):
        """Test getting server info."""
        client = MCPClient(mock_transport)
        await client.connect()

        info = client.get_server_info()

        assert info["name"] == "test"


# =============================================================================
# Test Resilient Client
# =============================================================================
class TestResilientMCPClient:
    """Tests for ResilientMCPClient."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on failure."""
        transport = MockTransport()
        transport.add_response("initialize", {"serverInfo": {}})

        config = RetryConfig(max_retries=3, base_delay=0.01)
        client = ResilientMCPClient(transport, config)

        await client.connect()

        # Client should handle retries internally

    def test_calculate_delay(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0)
        client = ResilientMCPClient(MockTransport(), config)

        delay1 = client._calculate_delay(0)
        delay2 = client._calculate_delay(1)
        delay3 = client._calculate_delay(2)

        assert delay1 < delay2 < delay3


# =============================================================================
# Test Multi-Server Client
# =============================================================================
class TestMultiServerClient:
    """Tests for MultiServerClient."""

    def test_add_server(self):
        """Test adding a server."""
        client = MultiServerClient()
        transport = MockTransport()

        client.add_server("server1", transport)

        assert "server1" in client._servers

    def test_remove_server(self):
        """Test removing a server."""
        client = MultiServerClient()
        transport = MockTransport()

        client.add_server("server1", transport)
        client.remove_server("server1")

        assert "server1" not in client._servers

    @pytest.mark.asyncio
    async def test_connect_all(self):
        """Test connecting to all servers."""
        client = MultiServerClient()

        t1 = MockTransport()
        t1.add_response("initialize", {"serverInfo": {}})
        t1.add_response("tools/list", {"tools": []})

        t2 = MockTransport()
        t2.add_response("initialize", {"serverInfo": {}})
        t2.add_response("tools/list", {"tools": []})

        client.add_server("s1", t1)
        client.add_server("s2", t2)

        results = await client.connect_all()

        assert results["s1"] is True
        assert results["s2"] is True


# =============================================================================
# Test Client Pool
# =============================================================================
class TestClientPool:
    """Tests for ClientPool."""

    def test_pool_creation(self):
        """Test creating a pool."""

        def factory(server):
            return MockTransport()

        pool = ClientPool(factory, max_per_server=3)

        assert pool._max_per_server == 3

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquiring and releasing clients."""

        def factory(server):
            t = MockTransport()
            t.add_response("initialize", {"serverInfo": {}})
            return t

        pool = ClientPool(factory)

        client = await pool.acquire("server1")
        assert client is not None

        await pool.release("server1", client)


# =============================================================================
# Test Caching Client
# =============================================================================
class TestCachingMCPClient:
    """Tests for CachingMCPClient."""

    @pytest.fixture
    def caching_client(self):
        """Create a caching client."""
        transport = MockTransport()
        transport.add_response("initialize", {"serverInfo": {}})
        transport.add_response("tools/list", {"tools": [{"name": "test"}]})
        return CachingMCPClient(transport, default_ttl=60.0)

    @pytest.mark.asyncio
    async def test_caches_responses(self, caching_client):
        """Test that responses are cached."""
        await caching_client.connect()

        # First call - should hit server
        result1 = await caching_client.list_tools()

        # Second call - should hit cache
        result2 = await caching_client.list_tools()

        assert result1 == result2

    def test_clear_cache(self, caching_client):
        """Test clearing cache."""
        caching_client._set_cached("test", "data", 60)

        caching_client.clear_cache()

        assert caching_client._get_cached("test") is None

    def test_cache_stats(self, caching_client):
        """Test getting cache stats."""
        caching_client._set_cached("key1", "data1", 60)
        caching_client._set_cached("key2", "data2", 60)

        stats = caching_client.get_cache_stats()

        assert stats["entries"] == 2

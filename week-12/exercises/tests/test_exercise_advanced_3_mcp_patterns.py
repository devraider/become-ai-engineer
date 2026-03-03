"""
Tests for Week 12 - Exercise 3 (Advanced): MCP Advanced Patterns
"""

import pytest
import asyncio
from exercise_advanced_3_mcp_patterns import (
    StreamChunk,
    StreamingHandler,
    StreamCollector,
    StreamingToolExecutor,
    MiddlewareContext,
    Middleware,
    MiddlewareChain,
    LoggingMiddleware,
    RateLimitMiddleware,
    CachingMiddleware,
    MiddlewareConfig,
    DynamicToolGenerator,
    CompositeServer,
    Permission,
    SecurityContext,
    SecurityManager,
    SecureServer,
    ResourceSubscriptionManager,
    PipelineStage,
    ExecutionPipeline,
    IntegrationHub,
)


# =============================================================================
# Test Streaming
# =============================================================================
class TestStreamChunk:
    """Tests for StreamChunk."""

    def test_chunk_creation(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(data="Hello", index=0, is_final=False)

        assert chunk.data == "Hello"
        assert chunk.index == 0
        assert chunk.is_final is False

    def test_final_chunk(self):
        """Test marking final chunk."""
        chunk = StreamChunk(data="", index=5, is_final=True)

        assert chunk.is_final is True


class TestStreamingHandler:
    """Tests for StreamingHandler."""

    @pytest.mark.asyncio
    async def test_on_chunk(self):
        """Test handling chunks."""
        handler = StreamingHandler()

        await handler.on_chunk(StreamChunk("data", 0, False))
        await handler.on_complete()

        # Should not raise

    @pytest.mark.asyncio
    async def test_on_error(self):
        """Test handling errors."""
        handler = StreamingHandler()

        await handler.on_error(Exception("test error"))

        # Should not raise


class TestStreamCollector:
    """Tests for StreamCollector."""

    @pytest.mark.asyncio
    async def test_collects_chunks(self):
        """Test collecting chunks."""
        collector = StreamCollector()

        await collector.on_chunk(StreamChunk("Hello ", 0, False))
        await collector.on_chunk(StreamChunk("World", 1, False))
        await collector.on_complete()

        result = collector.get_result()

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_is_complete(self):
        """Test completion status."""
        collector = StreamCollector()

        assert not collector.is_complete()

        await collector.on_complete()

        assert collector.is_complete()


class TestStreamingToolExecutor:
    """Tests for StreamingToolExecutor."""

    def test_register_tool(self):
        """Test registering streaming tools."""
        executor = StreamingToolExecutor()

        async def stream_tool(args, handler):
            await handler.on_chunk(StreamChunk("result", 0, True))
            await handler.on_complete()

        executor.register_streaming_tool("test", stream_tool)

        assert "test" in executor._streaming_tools

    @pytest.mark.asyncio
    async def test_execute_streaming(self):
        """Test executing streaming tool."""
        executor = StreamingToolExecutor()

        async def stream_tool(args, handler):
            await handler.on_chunk(StreamChunk("result", 0, True))
            await handler.on_complete()

        executor.register_streaming_tool("test", stream_tool)

        result = await executor.execute_streaming("test", {})

        assert result == "result"


# =============================================================================
# Test Middleware
# =============================================================================
class TestMiddlewareContext:
    """Tests for MiddlewareContext."""

    def test_context_creation(self):
        """Test creating middleware context."""
        ctx = MiddlewareContext(
            method="tools/call", params={"name": "test"}, metadata={}
        )

        assert ctx.method == "tools/call"
        assert ctx.params["name"] == "test"

    def test_set_result(self):
        """Test setting result."""
        ctx = MiddlewareContext("test", {})

        ctx.set_result({"data": "value"})

        assert ctx.get_result() == {"data": "value"}


class TestMiddlewareChain:
    """Tests for MiddlewareChain."""

    @pytest.mark.asyncio
    async def test_execute_chain(self):
        """Test executing middleware chain."""
        chain = MiddlewareChain()

        class TestMiddleware(Middleware):
            async def process(self, ctx, next_middleware):
                ctx.metadata["processed"] = True
                return await next_middleware()

        chain.add(TestMiddleware())

        ctx = MiddlewareContext("test", {})

        async def handler():
            return {"result": "ok"}

        result = await chain.execute(ctx, handler)

        assert ctx.metadata.get("processed") is True

    def test_add_remove_middleware(self):
        """Test adding and removing middleware."""
        chain = MiddlewareChain()
        middleware = LoggingMiddleware()

        chain.add(middleware)
        assert len(chain._middlewares) == 1

        chain.remove(middleware)
        assert len(chain._middlewares) == 0


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_request(self):
        """Test logging requests."""
        middleware = LoggingMiddleware()
        ctx = MiddlewareContext("test", {})

        async def next_mw():
            return {"result": "ok"}

        result = await middleware.process(ctx, next_mw)

        assert result == {"result": "ok"}
        # Logs should be recorded
        assert len(middleware._logs) > 0


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self):
        """Test allowing requests within limit."""
        middleware = RateLimitMiddleware(max_requests=10, window_seconds=1.0)
        ctx = MiddlewareContext("test", {})

        async def next_mw():
            return {"result": "ok"}

        result = await middleware.process(ctx, next_mw)

        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        """Test blocking requests over limit."""
        middleware = RateLimitMiddleware(max_requests=2, window_seconds=60.0)
        ctx = MiddlewareContext("test", {})

        async def next_mw():
            return {"result": "ok"}

        # First two should pass
        await middleware.process(ctx, next_mw)
        await middleware.process(ctx, next_mw)

        # Third should be blocked
        with pytest.raises(Exception):
            await middleware.process(ctx, next_mw)


class TestCachingMiddleware:
    """Tests for CachingMiddleware."""

    @pytest.mark.asyncio
    async def test_caches_response(self):
        """Test caching responses."""
        middleware = CachingMiddleware(cacheable_methods=["resources/read"])

        call_count = 0

        async def next_mw():
            nonlocal call_count
            call_count += 1
            return {"data": "value"}

        ctx1 = MiddlewareContext("resources/read", {"uri": "test://"})
        await middleware.process(ctx1, next_mw)

        ctx2 = MiddlewareContext("resources/read", {"uri": "test://"})
        await middleware.process(ctx2, next_mw)

        # Should only call handler once
        assert call_count == 1


# =============================================================================
# Test Dynamic Tools
# =============================================================================
class TestDynamicToolGenerator:
    """Tests for DynamicToolGenerator."""

    def test_from_function(self):
        """Test generating tool from function."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        generator = DynamicToolGenerator()
        tool = generator.from_function(add)

        assert tool["name"] == "add"
        assert "Add two numbers" in tool["description"]

    def test_from_dict(self):
        """Test generating tool from dict schema."""
        schema = {
            "name": "greet",
            "description": "Greet a user",
            "properties": {"name": {"type": "string", "description": "User name"}},
        }

        generator = DynamicToolGenerator()
        tool = generator.from_dict(schema)

        assert tool["name"] == "greet"


# =============================================================================
# Test Composite Server
# =============================================================================
class TestCompositeServer:
    """Tests for CompositeServer."""

    def test_add_component(self):
        """Test adding component server."""
        composite = CompositeServer()

        class MockComponent:
            def get_resources(self):
                return [{"uri": "test://", "name": "Test"}]

            def get_tools(self):
                return []

        composite.add_component("mock", MockComponent())

        assert "mock" in composite._components

    def test_get_all_resources(self):
        """Test aggregating resources from components."""
        composite = CompositeServer()

        class Component1:
            def get_resources(self):
                return [{"uri": "c1://", "name": "C1"}]

            def get_tools(self):
                return []

        class Component2:
            def get_resources(self):
                return [{"uri": "c2://", "name": "C2"}]

            def get_tools(self):
                return []

        composite.add_component("c1", Component1())
        composite.add_component("c2", Component2())

        resources = composite.get_all_resources()

        assert len(resources) == 2


# =============================================================================
# Test Security
# =============================================================================
class TestSecurityContext:
    """Tests for SecurityContext."""

    def test_creation(self):
        """Test creating security context."""
        ctx = SecurityContext(
            user_id="user1",
            permissions={Permission.READ_RESOURCE, Permission.CALL_TOOL},
        )

        assert ctx.user_id == "user1"
        assert Permission.READ_RESOURCE in ctx.permissions

    def test_has_permission(self):
        """Test checking permissions."""
        ctx = SecurityContext(user_id="user1", permissions={Permission.READ_RESOURCE})

        assert ctx.has_permission(Permission.READ_RESOURCE)
        assert not ctx.has_permission(Permission.CALL_TOOL)


class TestSecurityManager:
    """Tests for SecurityManager."""

    def test_create_context(self):
        """Test creating security context."""
        manager = SecurityManager()
        manager.register_user("user1", {Permission.READ_RESOURCE})

        ctx = manager.create_context("user1")

        assert ctx.user_id == "user1"
        assert Permission.READ_RESOURCE in ctx.permissions

    def test_check_permission(self):
        """Test checking permission."""
        manager = SecurityManager()
        manager.register_user("user1", {Permission.READ_RESOURCE})

        ctx = manager.create_context("user1")

        assert manager.check_permission(ctx, Permission.READ_RESOURCE)
        assert not manager.check_permission(ctx, Permission.CALL_TOOL)


class TestSecureServer:
    """Tests for SecureServer."""

    @pytest.mark.asyncio
    async def test_authorized_access(self):
        """Test authorized access."""
        server = SecureServer()
        server._security_manager.register_user(
            "admin",
            {
                Permission.READ_RESOURCE,
                Permission.CALL_TOOL,
                Permission.LIST_RESOURCES,
            },
        )

        ctx = server._security_manager.create_context("admin")

        resources = await server.list_resources(ctx)

        # Should return resources without error
        assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_unauthorized_access(self):
        """Test unauthorized access."""
        server = SecureServer()
        server._security_manager.register_user("guest", set())

        ctx = server._security_manager.create_context("guest")

        with pytest.raises(PermissionError):
            await server.list_resources(ctx)


# =============================================================================
# Test Subscriptions
# =============================================================================
class TestResourceSubscriptionManager:
    """Tests for ResourceSubscriptionManager."""

    @pytest.mark.asyncio
    async def test_subscribe(self):
        """Test subscribing to resource."""
        manager = ResourceSubscriptionManager()

        callback_data = []

        async def callback(data):
            callback_data.append(data)

        subscription_id = await manager.subscribe("test://data", callback)

        assert subscription_id is not None

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from resource."""
        manager = ResourceSubscriptionManager()

        async def callback(data):
            pass

        subscription_id = await manager.subscribe("test://data", callback)
        await manager.unsubscribe(subscription_id)

        # Should not find subscription anymore
        assert subscription_id not in manager._subscriptions

    @pytest.mark.asyncio
    async def test_notify(self):
        """Test notifying subscribers."""
        manager = ResourceSubscriptionManager()

        received = []

        async def callback(data):
            received.append(data)

        await manager.subscribe("test://data", callback)
        await manager.notify("test://data", {"updated": True})

        assert len(received) == 1
        assert received[0]["updated"] is True


# =============================================================================
# Test Execution Pipeline
# =============================================================================
class TestExecutionPipeline:
    """Tests for ExecutionPipeline."""

    @pytest.mark.asyncio
    async def test_add_stage(self):
        """Test adding pipeline stage."""
        pipeline = ExecutionPipeline()

        async def stage(data):
            return {**data, "processed": True}

        pipeline.add_stage("process", stage)

        assert "process" in pipeline._stages

    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test executing pipeline."""
        pipeline = ExecutionPipeline()

        async def stage1(data):
            return {**data, "stage1": True}

        async def stage2(data):
            return {**data, "stage2": True}

        pipeline.add_stage("stage1", stage1)
        pipeline.add_stage("stage2", stage2)

        result = await pipeline.execute({"input": "data"})

        assert result["stage1"] is True
        assert result["stage2"] is True


# =============================================================================
# Test Integration Hub
# =============================================================================
class TestIntegrationHub:
    """Tests for IntegrationHub."""

    def test_register_server(self):
        """Test registering server."""
        hub = IntegrationHub()

        class MockServer:
            def get_tools(self):
                return []

            def get_resources(self):
                return []

        hub.register_server("mock", MockServer())

        assert "mock" in hub._servers

    def test_create_client(self):
        """Test creating client."""
        hub = IntegrationHub()

        client = hub.create_client("test-client")

        assert client is not None

    @pytest.mark.asyncio
    async def test_route_request(self):
        """Test routing request."""
        hub = IntegrationHub()

        class MockServer:
            def get_tools(self):
                return [{"name": "test_tool"}]

            def get_resources(self):
                return []

            async def call_tool(self, name, args):
                return {"result": "ok"}

        hub.register_server("mock", MockServer())

        result = await hub.route_tool_call("test_tool", {})

        assert result == {"result": "ok"}

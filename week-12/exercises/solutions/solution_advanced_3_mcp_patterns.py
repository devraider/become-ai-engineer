"""
Week 12 - Exercise 3 (Advanced): MCP Advanced Patterns - SOLUTIONS

Complete implementations for advanced MCP patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import wraps
import asyncio
import hashlib
import inspect
import json
import time
import uuid


# =============================================================================
# Part 1: Streaming Handler
# =============================================================================
@dataclass
class StreamChunk:
    """A chunk of streaming data."""

    data: Any
    index: int
    is_final: bool
    metadata: dict = field(default_factory=dict)


class StreamingHandler(ABC):
    """
    Handler for processing streaming tool responses.
    """

    async def on_chunk(self, chunk: StreamChunk) -> None:
        """Handle a chunk of data."""
        pass

    async def on_complete(self) -> None:
        """Handle stream completion."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Handle stream error."""
        pass


class StreamCollector(StreamingHandler):
    """
    Collector that accumulates all chunks into a result.
    """

    def __init__(self):
        self._chunks: list[StreamChunk] = []
        self._complete: bool = False
        self._error: Optional[Exception] = None

    async def on_chunk(self, chunk: StreamChunk) -> None:
        """Collect a chunk."""
        self._chunks.append(chunk)

    async def on_complete(self) -> None:
        """Mark as complete."""
        self._complete = True

    async def on_error(self, error: Exception) -> None:
        """Record error."""
        self._error = error

    def get_result(self) -> str:
        """Get concatenated result."""
        return "".join(str(chunk.data) for chunk in self._chunks)

    def get_chunks(self) -> list[StreamChunk]:
        """Get all collected chunks."""
        return self._chunks.copy()

    def is_complete(self) -> bool:
        """Check if stream is complete."""
        return self._complete

    def get_error(self) -> Optional[Exception]:
        """Get error if any."""
        return self._error


class StreamingToolExecutor:
    """
    Executor for streaming tool calls.
    """

    def __init__(self):
        self._streaming_tools: dict[str, Callable] = {}

    def register_streaming_tool(
        self, name: str, handler: Callable[[dict, StreamingHandler], None]
    ) -> None:
        """Register a streaming tool."""
        self._streaming_tools[name] = handler

    async def execute_streaming(
        self, name: str, arguments: dict, handler: Optional[StreamingHandler] = None
    ) -> str:
        """Execute a streaming tool."""
        if name not in self._streaming_tools:
            raise ValueError(f"Unknown streaming tool: {name}")

        collector = handler or StreamCollector()

        tool_handler = self._streaming_tools[name]
        await tool_handler(arguments, collector)

        if isinstance(collector, StreamCollector):
            return collector.get_result()
        return ""

    async def stream_to_async_generator(self, name: str, arguments: dict):
        """Execute streaming tool as async generator."""
        if name not in self._streaming_tools:
            raise ValueError(f"Unknown streaming tool: {name}")

        queue: asyncio.Queue[Optional[StreamChunk]] = asyncio.Queue()

        class QueueHandler(StreamingHandler):
            async def on_chunk(self, chunk: StreamChunk) -> None:
                await queue.put(chunk)

            async def on_complete(self) -> None:
                await queue.put(None)

            async def on_error(self, error: Exception) -> None:
                await queue.put(None)

        handler = QueueHandler()

        asyncio.create_task(self._streaming_tools[name](arguments, handler))

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk


# =============================================================================
# Part 2: Middleware System
# =============================================================================
@dataclass
class MiddlewareContext:
    """Context passed through middleware chain."""

    method: str
    params: dict
    metadata: dict = field(default_factory=dict)
    _result: Optional[Any] = None

    def set_result(self, result: Any) -> None:
        """Set the result."""
        self._result = result

    def get_result(self) -> Optional[Any]:
        """Get the result."""
        return self._result


class Middleware(ABC):
    """Base class for MCP middleware."""

    @abstractmethod
    async def process(
        self, context: MiddlewareContext, next_middleware: Callable
    ) -> Any:
        """Process the request, optionally calling next middleware."""
        pass


class MiddlewareChain:
    """Chain of middleware to process requests."""

    def __init__(self):
        self._middlewares: list[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        """Add middleware to chain."""
        self._middlewares.append(middleware)

    def remove(self, middleware: Middleware) -> None:
        """Remove middleware from chain."""
        if middleware in self._middlewares:
            self._middlewares.remove(middleware)

    async def execute(self, context: MiddlewareContext, handler: Callable) -> Any:
        """Execute the middleware chain."""

        async def build_chain(index: int):
            if index >= len(self._middlewares):
                return await handler()

            middleware = self._middlewares[index]
            return await middleware.process(context, lambda: build_chain(index + 1))

        return await build_chain(0)


@dataclass
class MiddlewareConfig:
    """Configuration for middleware."""

    enabled: bool = True
    priority: int = 0
    options: dict = field(default_factory=dict)


class LoggingMiddleware(Middleware):
    """Middleware that logs requests and responses."""

    def __init__(self, log_level: str = "info"):
        self._log_level = log_level
        self._logs: list[dict] = []

    async def process(
        self, context: MiddlewareContext, next_middleware: Callable
    ) -> Any:
        """Log and process request."""
        start_time = time.time()

        self._logs.append(
            {
                "type": "request",
                "method": context.method,
                "params": context.params,
                "timestamp": start_time,
            }
        )

        try:
            result = await next_middleware()

            self._logs.append(
                {
                    "type": "response",
                    "method": context.method,
                    "duration": time.time() - start_time,
                    "success": True,
                }
            )

            return result
        except Exception as e:
            self._logs.append(
                {
                    "type": "error",
                    "method": context.method,
                    "error": str(e),
                    "duration": time.time() - start_time,
                }
            )
            raise

    def get_logs(self) -> list[dict]:
        """Get all logs."""
        return self._logs.copy()

    def clear_logs(self) -> None:
        """Clear logs."""
        self._logs.clear()


class RateLimitMiddleware(Middleware):
    """Middleware that enforces rate limits."""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: list[float] = []

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the window."""
        cutoff = time.time() - self._window
        self._requests = [t for t in self._requests if t > cutoff]

    async def process(
        self, context: MiddlewareContext, next_middleware: Callable
    ) -> Any:
        """Check rate limit and process."""
        self._cleanup_old_requests()

        if len(self._requests) >= self._max_requests:
            raise Exception("Rate limit exceeded")

        self._requests.append(time.time())
        return await next_middleware()

    def get_remaining(self) -> int:
        """Get remaining requests in window."""
        self._cleanup_old_requests()
        return max(0, self._max_requests - len(self._requests))


class CachingMiddleware(Middleware):
    """Middleware that caches responses."""

    def __init__(self, ttl: float = 300.0, cacheable_methods: Optional[set] = None):
        self._ttl = ttl
        self._cacheable = cacheable_methods or {"resources/read", "resources/list"}
        self._cache: dict[str, tuple[Any, float]] = {}

    def _cache_key(self, method: str, params: dict) -> str:
        """Generate cache key."""
        data = json.dumps({"method": method, "params": params}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    async def process(
        self, context: MiddlewareContext, next_middleware: Callable
    ) -> Any:
        """Check cache or process."""
        if context.method not in self._cacheable:
            return await next_middleware()

        key = self._cache_key(context.method, context.params)

        # Check cache
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return result

        # Execute and cache
        result = await next_middleware()
        self._cache[key] = (result, time.time())

        return result

    def invalidate(self, method: str, params: dict) -> None:
        """Invalidate cached entry."""
        key = self._cache_key(method, params)
        if key in self._cache:
            del self._cache[key]


# =============================================================================
# Part 3: Dynamic Tool Generation
# =============================================================================
class DynamicToolGenerator:
    """Generator for creating tools dynamically."""

    def from_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """Generate tool definition from function."""
        func_name = name or func.__name__
        func_desc = description or func.__doc__ or f"Executes {func_name}"

        sig = inspect.signature(func)
        hints = getattr(func, "__annotations__", {})

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            json_type = self._python_to_json_type(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func_name,
            "description": func_desc,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _python_to_json_type(self, python_type: type) -> str:
        """Convert Python type to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(python_type, "string")

    def from_dict(self, schema: dict) -> dict:
        """Generate tool definition from dictionary schema."""
        return {
            "name": schema.get("name", "unknown"),
            "description": schema.get("description", ""),
            "inputSchema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }

    def from_openapi(self, operation: dict, path: str) -> dict:
        """Generate tool from OpenAPI operation."""
        name = operation.get("operationId", path.replace("/", "_"))
        description = operation.get("summary", operation.get("description", ""))

        properties = {}
        required = []

        for param in operation.get("parameters", []):
            param_name = param["name"]
            properties[param_name] = {
                "type": param.get("schema", {}).get("type", "string"),
                "description": param.get("description", ""),
            }
            if param.get("required", False):
                required.append(param_name)

        return {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# =============================================================================
# Part 4: Server Composition
# =============================================================================
class CompositeServer:
    """Server that composes multiple component servers."""

    def __init__(self):
        self._components: dict[str, Any] = {}

    def add_component(self, name: str, component: Any) -> None:
        """Add a component server."""
        self._components[name] = component

    def remove_component(self, name: str) -> None:
        """Remove a component server."""
        if name in self._components:
            del self._components[name]

    def get_all_resources(self) -> list[dict]:
        """Get all resources from all components."""
        resources = []

        for name, component in self._components.items():
            if hasattr(component, "get_resources"):
                for resource in component.get_resources():
                    resource = resource.copy()
                    resource["_component"] = name
                    resources.append(resource)

        return resources

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all components."""
        tools = []

        for name, component in self._components.items():
            if hasattr(component, "get_tools"):
                for tool in component.get_tools():
                    tool = tool.copy()
                    tool["_component"] = name
                    tools.append(tool)

        return tools

    async def route_tool_call(self, tool_name: str, arguments: dict) -> Any:
        """Route tool call to appropriate component."""
        for name, component in self._components.items():
            if hasattr(component, "get_tools"):
                tools = component.get_tools()
                for tool in tools:
                    if tool.get("name") == tool_name:
                        if hasattr(component, "call_tool"):
                            return await component.call_tool(tool_name, arguments)

        raise ValueError(f"Tool not found: {tool_name}")


# =============================================================================
# Part 5: Security Layer
# =============================================================================
class Permission(Enum):
    """Permission types."""

    READ_RESOURCE = auto()
    WRITE_RESOURCE = auto()
    CALL_TOOL = auto()
    LIST_RESOURCES = auto()
    LIST_TOOLS = auto()
    ADMIN = auto()


@dataclass
class SecurityContext:
    """Security context for a request."""

    user_id: str
    permissions: set[Permission] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has permission."""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions


class SecurityManager:
    """Manager for security and permissions."""

    def __init__(self):
        self._users: dict[str, set[Permission]] = {}
        self._resource_permissions: dict[str, set[Permission]] = {}
        self._tool_permissions: dict[str, set[Permission]] = {}

    def register_user(self, user_id: str, permissions: set[Permission]) -> None:
        """Register a user with permissions."""
        self._users[user_id] = permissions

    def create_context(self, user_id: str) -> SecurityContext:
        """Create security context for user."""
        permissions = self._users.get(user_id, set())
        return SecurityContext(user_id=user_id, permissions=permissions)

    def check_permission(
        self, context: SecurityContext, permission: Permission
    ) -> bool:
        """Check if context has permission."""
        return context.has_permission(permission)

    def require_permission(
        self, context: SecurityContext, permission: Permission
    ) -> None:
        """Require permission or raise error."""
        if not context.has_permission(permission):
            raise PermissionError(
                f"User {context.user_id} lacks permission {permission.name}"
            )

    def set_resource_permission(self, uri: str, required: set[Permission]) -> None:
        """Set required permissions for a resource."""
        self._resource_permissions[uri] = required

    def check_resource_access(self, context: SecurityContext, uri: str) -> bool:
        """Check if context can access resource."""
        required = self._resource_permissions.get(uri, {Permission.READ_RESOURCE})
        return all(context.has_permission(p) for p in required)


class SecureServer:
    """MCP server with security enforcement."""

    def __init__(self):
        self._security_manager = SecurityManager()
        self._resources: dict[str, dict] = {}
        self._tools: dict[str, dict] = {}

    async def list_resources(self, context: SecurityContext) -> list[dict]:
        """List resources the user can access."""
        self._security_manager.require_permission(context, Permission.LIST_RESOURCES)

        accessible = []
        for uri, resource in self._resources.items():
            if self._security_manager.check_resource_access(context, uri):
                accessible.append(resource)

        return accessible

    async def read_resource(self, context: SecurityContext, uri: str) -> dict:
        """Read a resource with permission check."""
        self._security_manager.require_permission(context, Permission.READ_RESOURCE)

        if not self._security_manager.check_resource_access(context, uri):
            raise PermissionError(f"Access denied to resource: {uri}")

        return self._resources.get(uri, {})

    async def call_tool(
        self, context: SecurityContext, name: str, arguments: dict
    ) -> dict:
        """Call a tool with permission check."""
        self._security_manager.require_permission(context, Permission.CALL_TOOL)

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        return {"result": "executed"}


# =============================================================================
# Part 6: Resource Subscriptions
# =============================================================================
class ResourceSubscriptionManager:
    """Manager for resource change subscriptions."""

    def __init__(self):
        self._subscriptions: dict[str, dict[str, Callable]] = {}
        self._subscriber_resources: dict[str, set[str]] = {}

    async def subscribe(self, uri: str, callback: Callable[[dict], None]) -> str:
        """Subscribe to resource changes."""
        subscription_id = str(uuid.uuid4())

        if uri not in self._subscriptions:
            self._subscriptions[uri] = {}

        self._subscriptions[uri][subscription_id] = callback

        if subscription_id not in self._subscriber_resources:
            self._subscriber_resources[subscription_id] = set()
        self._subscriber_resources[subscription_id].add(uri)

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a resource."""
        if subscription_id not in self._subscriber_resources:
            return

        for uri in self._subscriber_resources[subscription_id]:
            if uri in self._subscriptions:
                self._subscriptions[uri].pop(subscription_id, None)

        del self._subscriber_resources[subscription_id]

    async def notify(self, uri: str, data: dict) -> None:
        """Notify subscribers of a resource change."""
        if uri not in self._subscriptions:
            return

        for callback in self._subscriptions[uri].values():
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    def get_subscribers(self, uri: str) -> int:
        """Get number of subscribers for a resource."""
        return len(self._subscriptions.get(uri, {}))


# =============================================================================
# Part 7: Execution Pipeline
# =============================================================================
@dataclass
class PipelineStage:
    """A stage in the execution pipeline."""

    name: str
    handler: Callable[[dict], dict]
    condition: Optional[Callable[[dict], bool]] = None
    timeout: Optional[float] = None


class ExecutionPipeline:
    """Pipeline for executing tool calls through stages."""

    def __init__(self):
        self._stages: dict[str, PipelineStage] = {}
        self._order: list[str] = []

    def add_stage(
        self,
        name: str,
        handler: Callable[[dict], dict],
        condition: Optional[Callable[[dict], bool]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Add a stage to the pipeline."""
        self._stages[name] = PipelineStage(
            name=name, handler=handler, condition=condition, timeout=timeout
        )
        self._order.append(name)

    def remove_stage(self, name: str) -> None:
        """Remove a stage from the pipeline."""
        if name in self._stages:
            del self._stages[name]
            self._order.remove(name)

    async def execute(self, data: dict) -> dict:
        """Execute the pipeline."""
        result = data.copy()

        for name in self._order:
            stage = self._stages[name]

            # Check condition
            if stage.condition and not stage.condition(result):
                continue

            # Execute with optional timeout
            if stage.timeout:
                try:
                    handler_result = stage.handler(result)
                    if asyncio.iscoroutine(handler_result):
                        handler_result = await asyncio.wait_for(
                            handler_result, timeout=stage.timeout
                        )
                    result = handler_result
                except asyncio.TimeoutError:
                    result["_timeout"] = name
                    break
            else:
                handler_result = stage.handler(result)
                if asyncio.iscoroutine(handler_result):
                    handler_result = await handler_result
                result = handler_result

        return result


# =============================================================================
# Part 8: Integration Hub
# =============================================================================
class IntegrationHub:
    """Central hub for integrating multiple MCP servers."""

    def __init__(self):
        self._servers: dict[str, Any] = {}
        self._clients: dict[str, Any] = {}
        self._middleware = MiddlewareChain()
        self._tool_index: dict[str, str] = {}  # tool_name -> server_name

    def register_server(self, name: str, server: Any) -> None:
        """Register an MCP server."""
        self._servers[name] = server

        # Index tools
        if hasattr(server, "get_tools"):
            for tool in server.get_tools():
                tool_name = tool.get("name")
                if tool_name:
                    self._tool_index[tool_name] = name

    def unregister_server(self, name: str) -> None:
        """Unregister an MCP server."""
        if name in self._servers:
            # Remove from tool index
            self._tool_index = {k: v for k, v in self._tool_index.items() if v != name}
            del self._servers[name]

    def create_client(self, name: str) -> "HubClient":
        """Create a client for accessing the hub."""
        client = HubClient(self, name)
        self._clients[name] = client
        return client

    def add_middleware(self, middleware: Middleware) -> None:
        """Add middleware to the processing chain."""
        self._middleware.add(middleware)

    async def route_tool_call(self, tool_name: str, arguments: dict) -> Any:
        """Route a tool call to the appropriate server."""
        server_name = self._tool_index.get(tool_name)

        if server_name is None:
            raise ValueError(f"Tool not found: {tool_name}")

        server = self._servers[server_name]

        if hasattr(server, "call_tool"):
            return await server.call_tool(tool_name, arguments)

        raise ValueError(f"Server {server_name} cannot call tools")

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all servers."""
        tools = []

        for server in self._servers.values():
            if hasattr(server, "get_tools"):
                tools.extend(server.get_tools())

        return tools

    def get_all_resources(self) -> list[dict]:
        """Get all resources from all servers."""
        resources = []

        for server in self._servers.values():
            if hasattr(server, "get_resources"):
                resources.extend(server.get_resources())

        return resources


class HubClient:
    """Client for accessing the integration hub."""

    def __init__(self, hub: IntegrationHub, name: str):
        self._hub = hub
        self._name = name

    async def list_tools(self) -> list[dict]:
        """List all available tools."""
        return self._hub.get_all_tools()

    async def list_resources(self) -> list[dict]:
        """List all available resources."""
        return self._hub.get_all_resources()

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool through the hub."""
        return await self._hub.route_tool_call(name, arguments)


# =============================================================================
# Example Usage
# =============================================================================
async def main():
    """Demonstrate advanced patterns."""
    # Streaming example
    print("=== Streaming Example ===")

    executor = StreamingToolExecutor()

    async def stream_count(args: dict, handler: StreamingHandler):
        """Stream a count."""
        count = args.get("count", 5)
        for i in range(count):
            await handler.on_chunk(
                StreamChunk(data=f"{i + 1} ", index=i, is_final=(i == count - 1))
            )
            await asyncio.sleep(0.1)
        await handler.on_complete()

    executor.register_streaming_tool("count", stream_count)

    result = await executor.execute_streaming("count", {"count": 5})
    print(f"Stream result: {result}")

    # Middleware example
    print("\n=== Middleware Example ===")

    chain = MiddlewareChain()
    chain.add(LoggingMiddleware())
    chain.add(RateLimitMiddleware(max_requests=10, window_seconds=60))

    ctx = MiddlewareContext(method="tools/call", params={"name": "test"})

    async def handler():
        return {"result": "success"}

    result = await chain.execute(ctx, handler)
    print(f"Middleware result: {result}")

    # Dynamic tool generation
    print("\n=== Dynamic Tool Generation ===")

    generator = DynamicToolGenerator()

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    tool_def = generator.from_function(greet)
    print(f"Generated tool: {json.dumps(tool_def, indent=2)}")

    # Security example
    print("\n=== Security Example ===")

    secure_server = SecureServer()
    secure_server._security_manager.register_user("admin", {Permission.ADMIN})
    secure_server._security_manager.register_user(
        "reader", {Permission.READ_RESOURCE, Permission.LIST_RESOURCES}
    )

    admin_ctx = secure_server._security_manager.create_context("admin")
    reader_ctx = secure_server._security_manager.create_context("reader")

    print(f"Admin has ADMIN: {admin_ctx.has_permission(Permission.ADMIN)}")
    print(f"Reader has ADMIN: {reader_ctx.has_permission(Permission.ADMIN)}")

    # Subscription example
    print("\n=== Subscription Example ===")

    sub_manager = ResourceSubscriptionManager()

    notifications = []

    async def on_change(data: dict):
        notifications.append(data)
        print(f"Notification received: {data}")

    sub_id = await sub_manager.subscribe("test://data", on_change)
    print(f"Subscribed with ID: {sub_id}")

    await sub_manager.notify("test://data", {"changed": True})

    await sub_manager.unsubscribe(sub_id)

    # Pipeline example
    print("\n=== Pipeline Example ===")

    pipeline = ExecutionPipeline()

    async def validate(data: dict) -> dict:
        data["validated"] = True
        return data

    async def process(data: dict) -> dict:
        data["processed"] = True
        return data

    async def format_output(data: dict) -> dict:
        data["formatted"] = True
        return data

    pipeline.add_stage("validate", validate)
    pipeline.add_stage("process", process)
    pipeline.add_stage("format", format_output)

    result = await pipeline.execute({"input": "test"})
    print(f"Pipeline result: {result}")

    # Integration hub example
    print("\n=== Integration Hub Example ===")

    hub = IntegrationHub()

    class Server1:
        def get_tools(self):
            return [{"name": "tool1", "description": "Tool 1"}]

        def get_resources(self):
            return [{"uri": "s1://data", "name": "Data"}]

        async def call_tool(self, name, args):
            return {"from": "server1", "tool": name}

    class Server2:
        def get_tools(self):
            return [{"name": "tool2", "description": "Tool 2"}]

        def get_resources(self):
            return []

        async def call_tool(self, name, args):
            return {"from": "server2", "tool": name}

    hub.register_server("server1", Server1())
    hub.register_server("server2", Server2())

    client = hub.create_client("main")

    tools = await client.list_tools()
    print(f"All tools: {tools}")

    result = await client.call_tool("tool1", {})
    print(f"Tool1 result: {result}")

    result = await client.call_tool("tool2", {})
    print(f"Tool2 result: {result}")


if __name__ == "__main__":
    asyncio.run(main())

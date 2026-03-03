"""
Solution for Week 14 - Exercise 3: Multi-Agent Systems

Complete implementations for multi-agent coordination and orchestration.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio
import json
import random
import uuid


# =============================================================================
# Part 1: Agent Role - SOLUTION
# =============================================================================
class Capability(Enum):
    """Agent capabilities."""

    SEARCH = "search"
    ANALYZE = "analyze"
    WRITE = "write"
    CODE = "code"
    PLAN = "plan"
    COORDINATE = "coordinate"
    EXECUTE = "execute"


@dataclass
class AgentRole:
    """
    Defines an agent's role and capabilities.

    Solution implements:
    - Capability checking
    - Authority levels
    - Role comparison
    """

    name: str
    description: str
    capabilities: List[Capability]
    priority: int = 1  # Higher = more authority

    def can(self, capability: Capability) -> bool:
        """Check if role has a capability."""
        return capability in self.capabilities

    def has_authority_over(self, other: "AgentRole") -> bool:
        """Check if this role has authority over another."""
        return self.priority > other.priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [c.value for c in self.capabilities],
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRole":
        """Create role from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            capabilities=[Capability(c) for c in data.get("capabilities", [])],
            priority=data.get("priority", 1),
        )


# Pre-defined roles
RESEARCHER_ROLE = AgentRole(
    name="researcher",
    description="Searches and gathers information",
    capabilities=[Capability.SEARCH, Capability.ANALYZE],
    priority=1,
)

ANALYST_ROLE = AgentRole(
    name="analyst",
    description="Analyzes data and provides insights",
    capabilities=[Capability.ANALYZE, Capability.WRITE],
    priority=2,
)

WRITER_ROLE = AgentRole(
    name="writer",
    description="Writes and formats content",
    capabilities=[Capability.WRITE],
    priority=1,
)

COORDINATOR_ROLE = AgentRole(
    name="coordinator",
    description="Coordinates other agents",
    capabilities=[Capability.COORDINATE, Capability.PLAN],
    priority=10,
)


# =============================================================================
# Part 2: Agent Message - SOLUTION
# =============================================================================
class MessageType(Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """
    Message for inter-agent communication.

    Solution implements:
    - Message types
    - Priority handling
    - Correlation tracking
    """

    sender: str
    receiver: str
    content: Any
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_broadcast(self) -> bool:
        """Check if message is a broadcast."""
        return self.receiver == "*" or self.message_type == MessageType.BROADCAST

    def is_response_to(self, other: "AgentMessage") -> bool:
        """Check if this message is a response to another."""
        return self.correlation_id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def request(
        cls, sender: str, receiver: str, content: Any, **kwargs
    ) -> "AgentMessage":
        """Create a request message."""
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=MessageType.REQUEST,
            **kwargs,
        )

    @classmethod
    def response(
        cls, request: "AgentMessage", sender: str, content: Any, **kwargs
    ) -> "AgentMessage":
        """Create a response to a request."""
        return cls(
            sender=sender,
            receiver=request.sender,
            content=content,
            message_type=MessageType.RESPONSE,
            correlation_id=request.id,
            **kwargs,
        )

    @classmethod
    def broadcast(cls, sender: str, content: Any, **kwargs) -> "AgentMessage":
        """Create a broadcast message."""
        return cls(
            sender=sender,
            receiver="*",
            content=content,
            message_type=MessageType.BROADCAST,
            **kwargs,
        )


# =============================================================================
# Part 3: Message Broker - SOLUTION
# =============================================================================
class MessageBroker:
    """
    Handles message routing between agents.

    Solution implements:
    - Agent registration
    - Message routing
    - Broadcast handling
    - Message history
    """

    def __init__(self, max_history: int = 1000):
        self._handlers: Dict[str, Callable[[AgentMessage], None]] = {}
        self._subscriptions: Dict[str, Set[str]] = {}  # topic -> agents
        self._history: List[AgentMessage] = []
        self._max_history = max_history

    def register(self, agent_id: str, handler: Callable[[AgentMessage], None]) -> None:
        """Register an agent's message handler."""
        self._handlers[agent_id] = handler

    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self._handlers:
            del self._handlers[agent_id]
            # Remove from subscriptions
            for topic in self._subscriptions.values():
                topic.discard(agent_id)
            return True
        return False

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = set()
        self._subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe from a topic."""
        if topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id)
            return True
        return False

    def send(self, message: AgentMessage) -> bool:
        """Send a message to its destination."""
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        if message.is_broadcast():
            return self._broadcast(message)
        else:
            return self._direct_send(message)

    def _direct_send(self, message: AgentMessage) -> bool:
        """Send to a specific agent."""
        handler = self._handlers.get(message.receiver)
        if handler:
            try:
                handler(message)
                return True
            except Exception:
                return False
        return False

    def _broadcast(self, message: AgentMessage) -> bool:
        """Broadcast to all agents except sender."""
        success = True
        for agent_id, handler in self._handlers.items():
            if agent_id != message.sender:
                try:
                    handler(message)
                except Exception:
                    success = False
        return success

    def publish(self, topic: str, message: AgentMessage) -> int:
        """Publish to a topic."""
        subscribers = self._subscriptions.get(topic, set())
        sent = 0

        for agent_id in subscribers:
            handler = self._handlers.get(agent_id)
            if handler and agent_id != message.sender:
                try:
                    handler(message)
                    sent += 1
                except Exception:
                    pass

        return sent

    def get_history(
        self, limit: int = 100, agent_id: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get message history."""
        history = self._history

        if agent_id:
            history = [
                m for m in history if m.sender == agent_id or m.receiver == agent_id
            ]

        return history[-limit:]

    def get_pending(self, agent_id: str) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        return [
            m
            for m in self._history
            if m.receiver == agent_id and m.message_type == MessageType.REQUEST
        ]


# =============================================================================
# Part 4: Agent Coordinator - SOLUTION
# =============================================================================
class CoordinationStrategy(Enum):
    """Coordination strategies."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    id: str
    role: AgentRole
    status: str = "idle"  # idle, busy, error
    current_task: Optional[str] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator:
    """
    Coordinates multiple agents.

    Solution implements:
    - Agent registration
    - Task distribution
    - Status tracking
    - Load balancing
    """

    def __init__(
        self, strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    ):
        self._agents: Dict[str, AgentInfo] = {}
        self._handlers: Dict[str, Callable] = {}
        self.strategy = strategy
        self._task_queue: List[Dict[str, Any]] = []
        self._completed_tasks: List[Dict[str, Any]] = []

    def register_agent(
        self, agent_id: str, role: AgentRole, handler: Optional[Callable] = None
    ) -> None:
        """Register an agent."""
        self._agents[agent_id] = AgentInfo(id=agent_id, role=role)
        if handler:
            self._handlers[agent_id] = handler

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._handlers.pop(agent_id, None)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info."""
        return self._agents.get(agent_id)

    def get_available_agents(
        self, capability: Optional[Capability] = None
    ) -> List[AgentInfo]:
        """Get available agents, optionally filtered by capability."""
        agents = [a for a in self._agents.values() if a.status == "idle"]

        if capability:
            agents = [a for a in agents if a.role.can(capability)]

        return agents

    def assign_task(self, agent_id: str, task: Dict[str, Any]) -> bool:
        """Assign a task to an agent."""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != "idle":
            return False

        agent.status = "busy"
        agent.current_task = task.get("id", str(uuid.uuid4()))
        agent.last_activity = datetime.now()

        return True

    def complete_task(self, agent_id: str, result: Any = None) -> bool:
        """Mark agent's current task as complete."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        task_id = agent.current_task
        agent.status = "idle"
        agent.current_task = None
        agent.last_activity = datetime.now()

        if task_id:
            self._completed_tasks.append(
                {
                    "id": task_id,
                    "agent_id": agent_id,
                    "result": result,
                    "completed_at": datetime.now().isoformat(),
                }
            )

        return True

    def distribute_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Distribute tasks to available agents."""
        distribution = {}

        for task in tasks:
            required_cap = task.get("required_capability")
            if required_cap:
                required_cap = (
                    Capability(required_cap)
                    if isinstance(required_cap, str)
                    else required_cap
                )

            available = self.get_available_agents(required_cap)

            if available:
                # Simple round-robin for now
                agent = available[0]
                task_id = task.get("id", str(uuid.uuid4()))

                if self.assign_task(agent.id, task):
                    distribution[task_id] = agent.id

        return distribution

    def get_status(self) -> Dict[str, Any]:
        """Get overall coordination status."""
        return {
            "total_agents": len(self._agents),
            "idle_agents": len(
                [a for a in self._agents.values() if a.status == "idle"]
            ),
            "busy_agents": len(
                [a for a in self._agents.values() if a.status == "busy"]
            ),
            "pending_tasks": len(self._task_queue),
            "completed_tasks": len(self._completed_tasks),
            "strategy": self.strategy.value,
        }


# =============================================================================
# Part 5: Workflow Step - SOLUTION
# =============================================================================
class StepType(Enum):
    """Types of workflow steps."""

    ACTION = "action"
    DECISION = "decision"
    PARALLEL = "parallel"
    WAIT = "wait"
    END = "end"


@dataclass
class WorkflowStep:
    """
    A single step in a workflow.

    Solution implements:
    - Step types
    - Conditional execution
    - Input/output handling
    """

    id: str
    name: str
    step_type: StepType = StepType.ACTION
    required_capability: Optional[Capability] = None
    handler: Optional[Callable] = None
    next_steps: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict], bool]] = None
    timeout: float = 60.0
    retry_count: int = 0

    def get_next(self, context: Dict[str, Any]) -> List[str]:
        """Get next steps based on context."""
        if self.step_type == StepType.DECISION and self.condition:
            # For decision steps, condition determines path
            # Return first step if condition is true, second if false
            if len(self.next_steps) >= 2:
                if self.condition(context):
                    return [self.next_steps[0]]
                else:
                    return [self.next_steps[1]]

        return self.next_steps

    def can_execute(self, role: AgentRole) -> bool:
        """Check if a role can execute this step."""
        if not self.required_capability:
            return True
        return role.can(self.required_capability)

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the step."""
        if not self.handler:
            return context

        if asyncio.iscoroutinefunction(self.handler):
            return await asyncio.wait_for(self.handler(context), timeout=self.timeout)
        else:
            return self.handler(context)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type.value,
            "required_capability": (
                self.required_capability.value if self.required_capability else None
            ),
            "next_steps": self.next_steps,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }


# =============================================================================
# Part 6: Workflow Engine - SOLUTION
# =============================================================================
@dataclass
class WorkflowContext:
    """Context for workflow execution."""

    workflow_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.data.get(key, default)


class WorkflowEngine:
    """
    Executes agent workflows.

    Solution implements:
    - Workflow definition
    - Step execution
    - Error handling
    - Progress tracking
    """

    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self._workflows: Dict[str, List[WorkflowStep]] = {}
        self._running: Dict[str, WorkflowContext] = {}

    def define_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """Define a workflow."""
        self._workflows[name] = steps

    def get_workflow(self, name: str) -> Optional[List[WorkflowStep]]:
        """Get a workflow definition."""
        return self._workflows.get(name)

    async def run(
        self, workflow_name: str, initial_data: Dict[str, Any]
    ) -> WorkflowContext:
        """Run a workflow."""
        steps = self._workflows.get(workflow_name)
        if not steps:
            raise ValueError(f"Workflow not found: {workflow_name}")

        # Create context
        context = WorkflowContext(workflow_id=str(uuid.uuid4()), data=initial_data)

        self._running[context.workflow_id] = context

        # Build step lookup
        step_map = {s.id: s for s in steps}

        # Start from first step
        current_steps = [steps[0].id] if steps else []

        try:
            while current_steps:
                next_steps = []

                for step_id in current_steps:
                    step = step_map.get(step_id)
                    if not step:
                        continue

                    context.current_step = step_id

                    try:
                        # Find agent for this step
                        agent = self._find_agent_for_step(step)

                        if agent:
                            self.coordinator.assign_task(
                                agent.id, {"id": step_id, "type": step.step_type.value}
                            )

                        # Execute step
                        result = await step.execute(context.data)

                        if isinstance(result, dict):
                            context.data.update(result)

                        context.completed_steps.append(step_id)

                        if agent:
                            self.coordinator.complete_task(agent.id)

                        # Get next steps
                        next_steps.extend(step.get_next(context.data))

                    except Exception as e:
                        context.errors.append(
                            {
                                "step": step_id,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        # Retry logic
                        if step.retry_count > 0:
                            step.retry_count -= 1
                            next_steps.append(step_id)

                current_steps = list(set(next_steps))

        finally:
            del self._running[context.workflow_id]

        return context

    def _find_agent_for_step(self, step: WorkflowStep) -> Optional[AgentInfo]:
        """Find an agent that can execute a step."""
        if not step.required_capability:
            available = self.coordinator.get_available_agents()
        else:
            available = self.coordinator.get_available_agents(step.required_capability)

        return available[0] if available else None

    def get_progress(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow progress."""
        context = self._running.get(workflow_id)
        if not context:
            return None

        return {
            "workflow_id": workflow_id,
            "current_step": context.current_step,
            "completed_steps": len(context.completed_steps),
            "errors": len(context.errors),
            "started_at": context.started_at.isoformat(),
        }

    def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self._running:
            del self._running[workflow_id]
            return True
        return False


# =============================================================================
# Part 7: Agent Pool - SOLUTION
# =============================================================================
class PoolStrategy(Enum):
    """Agent pool strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    RANDOM = "random"
    CAPABILITY_MATCH = "capability_match"


class AgentPool:
    """
    Pool of agents for task execution.

    Solution implements:
    - Agent pooling
    - Acquisition/release
    - Strategy-based selection
    """

    def __init__(self, strategy: PoolStrategy = PoolStrategy.ROUND_ROBIN):
        self._agents: Dict[str, AgentRole] = {}
        self._in_use: Set[str] = set()
        self._usage_count: Dict[str, int] = {}
        self.strategy = strategy
        self._robin_index = 0

    def add(self, agent_id: str, role: AgentRole) -> None:
        """Add an agent to the pool."""
        self._agents[agent_id] = role
        self._usage_count[agent_id] = 0

    def remove(self, agent_id: str) -> bool:
        """Remove an agent from the pool."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._in_use.discard(agent_id)
            self._usage_count.pop(agent_id, None)
            return True
        return False

    def acquire(self, capability: Optional[Capability] = None) -> Optional[str]:
        """Acquire an agent from the pool."""
        available = self._get_available(capability)

        if not available:
            return None

        # Select based on strategy
        if self.strategy == PoolStrategy.ROUND_ROBIN:
            agent_id = self._round_robin_select(available)
        elif self.strategy == PoolStrategy.LEAST_BUSY:
            agent_id = self._least_busy_select(available)
        elif self.strategy == PoolStrategy.RANDOM:
            agent_id = random.choice(available)
        else:
            agent_id = available[0]

        self._in_use.add(agent_id)
        self._usage_count[agent_id] = self._usage_count.get(agent_id, 0) + 1

        return agent_id

    def release(self, agent_id: str) -> bool:
        """Release an agent back to the pool."""
        if agent_id in self._in_use:
            self._in_use.discard(agent_id)
            return True
        return False

    def _get_available(self, capability: Optional[Capability] = None) -> List[str]:
        """Get available agents."""
        available = [aid for aid in self._agents.keys() if aid not in self._in_use]

        if capability:
            available = [aid for aid in available if self._agents[aid].can(capability)]

        return available

    def _round_robin_select(self, available: List[str]) -> str:
        """Round-robin selection."""
        sorted_available = sorted(available)
        self._robin_index = self._robin_index % len(sorted_available)
        selected = sorted_available[self._robin_index]
        self._robin_index += 1
        return selected

    def _least_busy_select(self, available: List[str]) -> str:
        """Select least used agent."""
        return min(available, key=lambda x: self._usage_count.get(x, 0))

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total": len(self._agents),
            "available": len(self._agents) - len(self._in_use),
            "in_use": len(self._in_use),
            "usage_counts": self._usage_count.copy(),
        }


# =============================================================================
# Part 8: Conflict Resolver - SOLUTION
# =============================================================================
class ConflictType(Enum):
    """Types of conflicts."""

    RESOURCE = "resource"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    DEADLINE = "deadline"


@dataclass
class Conflict:
    """Represents a conflict between agents."""

    id: str
    conflict_type: ConflictType
    agents: List[str]
    description: str
    resource: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None


class ConflictResolver:
    """
    Resolves conflicts between agents.

    Solution implements:
    - Conflict detection
    - Resolution strategies
    - Priority-based resolution
    """

    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self._conflicts: List[Conflict] = []
        self._resolutions: Dict[ConflictType, Callable] = {
            ConflictType.RESOURCE: self._resolve_resource,
            ConflictType.PRIORITY: self._resolve_priority,
            ConflictType.DEPENDENCY: self._resolve_dependency,
            ConflictType.DEADLINE: self._resolve_deadline,
        }

    def resolve(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve a conflict."""
        self._conflicts.append(conflict)

        resolver = self._resolutions.get(conflict.conflict_type)
        if resolver:
            result = resolver(conflict)
            conflict.resolved = True
            conflict.resolution = result.get("action", "resolved")
            return result

        return {"action": "no_resolution", "conflict_id": conflict.id}

    def _resolve_resource(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve resource conflict."""
        # Get agent priorities
        agent_priorities = []
        for agent_id in conflict.agents:
            agent = self.coordinator.get_agent(agent_id)
            if agent:
                agent_priorities.append((agent_id, agent.role.priority))

        if not agent_priorities:
            return {"action": "no_agents"}

        # Highest priority wins
        agent_priorities.sort(key=lambda x: x[1], reverse=True)
        winner = agent_priorities[0][0]

        return {
            "action": "assign_resource",
            "winner": winner,
            "resource": conflict.resource,
        }

    def _resolve_priority(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve priority conflict."""
        agents = []
        for agent_id in conflict.agents:
            agent = self.coordinator.get_agent(agent_id)
            if agent:
                agents.append((agent_id, agent.role.priority))

        if not agents:
            return {"action": "no_agents"}

        # Sort by priority
        agents.sort(key=lambda x: x[1], reverse=True)

        return {"action": "priority_order", "order": [a[0] for a in agents]}

    def _resolve_dependency(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve dependency conflict."""
        # For dependencies, create execution order
        return {
            "action": "sequence",
            "order": conflict.agents,  # Simple sequential order
        }

    def _resolve_deadline(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve deadline conflict."""
        # Prioritize based on urgency
        return {
            "action": "prioritize",
            "priority_agent": conflict.agents[0] if conflict.agents else None,
        }

    def get_history(self) -> List[Conflict]:
        """Get conflict history."""
        return self._conflicts.copy()

    def clear_history(self) -> None:
        """Clear conflict history."""
        self._conflicts = []


# =============================================================================
# Part 9: Agent Supervisor - SOLUTION
# =============================================================================
class SupervisorAction(Enum):
    """Supervisor actions."""

    NONE = "none"
    WARN = "warn"
    RESTART = "restart"
    REASSIGN = "reassign"
    TERMINATE = "terminate"


@dataclass
class AgentHealth:
    """Health status of an agent."""

    agent_id: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    response_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return self.status == "healthy"


class AgentSupervisor:
    """
    Supervises agent health and performance.

    Solution implements:
    - Health monitoring
    - Performance tracking
    - Automatic actions
    """

    def __init__(self, coordinator: AgentCoordinator, check_interval: float = 30.0):
        self.coordinator = coordinator
        self.check_interval = check_interval
        self._health: Dict[str, AgentHealth] = {}
        self._actions_taken: List[Dict[str, Any]] = []
        self._thresholds = {"error_threshold": 5, "response_time_threshold": 10.0}

    def check_health(self, agent_id: str) -> AgentHealth:
        """Check health of a specific agent."""
        agent = self.coordinator.get_agent(agent_id)

        if not agent:
            return AgentHealth(agent_id=agent_id, status="unknown")

        # Determine health status
        health = self._health.get(
            agent_id, AgentHealth(agent_id=agent_id, status="healthy")
        )

        # Check for issues
        if health.error_count >= self._thresholds["error_threshold"]:
            health.status = "unhealthy"
        elif health.response_time > self._thresholds["response_time_threshold"]:
            health.status = "degraded"
        else:
            health.status = "healthy"

        health.last_check = datetime.now()
        self._health[agent_id] = health

        return health

    def check_all(self) -> Dict[str, AgentHealth]:
        """Check all agents."""
        status = self.coordinator.get_status()

        for agent_id in list(self.coordinator._agents.keys()):
            self.check_health(agent_id)

        return self._health.copy()

    def report_error(self, agent_id: str, error: str) -> None:
        """Report an error for an agent."""
        if agent_id not in self._health:
            self._health[agent_id] = AgentHealth(agent_id=agent_id, status="healthy")

        self._health[agent_id].error_count += 1

        # Check if action needed
        health = self.check_health(agent_id)
        if not health.is_healthy():
            self.take_action(agent_id, SupervisorAction.WARN)

    def take_action(self, agent_id: str, action: SupervisorAction) -> bool:
        """Take action on an agent."""
        agent = self.coordinator.get_agent(agent_id)
        if not agent:
            return False

        action_record = {
            "agent_id": agent_id,
            "action": action.value,
            "timestamp": datetime.now().isoformat(),
        }

        if action == SupervisorAction.WARN:
            # Log warning
            pass
        elif action == SupervisorAction.RESTART:
            # Reset agent
            if agent_id in self._health:
                self._health[agent_id].error_count = 0
                self._health[agent_id].status = "healthy"
        elif action == SupervisorAction.REASSIGN:
            # Reassign tasks
            self.coordinator.complete_task(agent_id)
        elif action == SupervisorAction.TERMINATE:
            # Remove agent
            self.coordinator.unregister_agent(agent_id)
            self._health.pop(agent_id, None)

        self._actions_taken.append(action_record)
        return True

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all agent health."""
        total = len(self._health)
        healthy = sum(1 for h in self._health.values() if h.status == "healthy")
        degraded = sum(1 for h in self._health.values() if h.status == "degraded")
        unhealthy = sum(1 for h in self._health.values() if h.status == "unhealthy")

        return {
            "total": total,
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
        }

    def set_threshold(self, name: str, value: float) -> None:
        """Set a threshold value."""
        self._thresholds[name] = value


# =============================================================================
# Part 10: Multi-Agent System - SOLUTION
# =============================================================================
@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system."""

    name: str
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    max_agents: int = 100
    supervisor_enabled: bool = True
    pool_strategy: PoolStrategy = PoolStrategy.ROUND_ROBIN


class MultiAgentSystem:
    """
    Complete multi-agent system.

    Solution implements:
    - Full agent lifecycle
    - Workflow execution
    - Monitoring
    - Coordination
    """

    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.coordinator = AgentCoordinator(config.coordination_strategy)
        self.broker = MessageBroker()
        self.pool = AgentPool(config.pool_strategy)
        self.workflow_engine = WorkflowEngine(self.coordinator)
        self.resolver = ConflictResolver(self.coordinator)

        if config.supervisor_enabled:
            self.supervisor = AgentSupervisor(self.coordinator)
        else:
            self.supervisor = None

        self._handlers: Dict[str, Callable] = {}
        self._running = False

    def add_agent(
        self, agent_id: str, role: AgentRole, handler: Callable[[Dict], Awaitable[Dict]]
    ) -> None:
        """Add an agent to the system."""
        self.coordinator.register_agent(agent_id, role, handler)
        self.pool.add(agent_id, role)
        self._handlers[agent_id] = handler

        # Set up message handling
        self.broker.register(agent_id, lambda msg: self._handle_message(agent_id, msg))

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system."""
        self.coordinator.unregister_agent(agent_id)
        self.pool.remove(agent_id)
        self.broker.unregister(agent_id)
        self._handlers.pop(agent_id, None)
        return True

    def _handle_message(self, agent_id: str, message: AgentMessage) -> None:
        """Handle incoming message for an agent."""
        handler = self._handlers.get(agent_id)
        if handler:
            # Queue for async processing
            pass

    def define_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """Define a workflow."""
        self.workflow_engine.define_workflow(name, steps)

    async def execute(
        self, workflow_name: str, data: Dict[str, Any]
    ) -> WorkflowContext:
        """Execute a workflow."""
        return await self.workflow_engine.run(workflow_name, data)

    async def send_task(self, agent_id: str, task: Dict[str, Any]) -> Any:
        """Send a task to an agent."""
        handler = self._handlers.get(agent_id)
        if not handler:
            return None

        self.coordinator.assign_task(agent_id, task)

        try:
            result = await handler(task)
            self.coordinator.complete_task(agent_id, result)
            return result
        except Exception as e:
            if self.supervisor:
                self.supervisor.report_error(agent_id, str(e))
            return None

    def start(self) -> None:
        """Start the system."""
        self._running = True

    def stop(self) -> None:
        """Stop the system."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = {
            "name": self.config.name,
            "running": self._running,
            "coordinator": self.coordinator.get_status(),
            "pool": self.pool.get_stats(),
        }

        if self.supervisor:
            status["health"] = self.supervisor.get_health_summary()

        return status


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":

    async def main():
        # Create system
        config = MultiAgentConfig(
            name="research-system",
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
        )

        system = MultiAgentSystem(config)

        # Define agent handlers
        async def researcher_handler(task: Dict) -> Dict:
            print(f"Researching: {task}")
            return {"findings": "Research results"}

        async def writer_handler(task: Dict) -> Dict:
            print(f"Writing: {task}")
            return {"report": "Final report"}

        # Add agents
        system.add_agent("researcher", RESEARCHER_ROLE, researcher_handler)
        system.add_agent("writer", WRITER_ROLE, writer_handler)

        # Define workflow
        workflow_steps = [
            WorkflowStep(
                id="research",
                name="Research",
                step_type=StepType.ACTION,
                required_capability=Capability.SEARCH,
                next_steps=["write"],
            ),
            WorkflowStep(
                id="write",
                name="Write",
                step_type=StepType.ACTION,
                required_capability=Capability.WRITE,
            ),
        ]

        system.define_workflow("research_pipeline", workflow_steps)

        # Start system
        system.start()

        # Execute workflow
        result = await system.execute("research_pipeline", {"topic": "AI trends"})

        print(f"Workflow result: {result.data}")
        print(f"System status: {system.get_status()}")

        system.stop()

    asyncio.run(main())

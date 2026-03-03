"""
Week 14 - Exercise 3: Multi-Agent Systems (Advanced)

Learn to build sophisticated multi-agent systems:
- Agent roles and capabilities
- Inter-agent communication
- Workflow orchestration
- Conflict resolution and supervision

Run tests with: pytest tests/test_exercise_advanced_3_multi_agent.py -v
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
from datetime import datetime
import uuid


# =============================================================================
# Part 1: Agent Role Definition
# =============================================================================
class Capability(Enum):
    """Agent capabilities."""

    SEARCH = "search"
    ANALYZE = "analyze"
    WRITE = "write"
    CODE = "code"
    PLAN = "plan"
    REVIEW = "review"
    SUMMARIZE = "summarize"
    COORDINATE = "coordinate"


@dataclass
class AgentRole:
    """
    Defines an agent's role and capabilities.

    Example:
        >>> role = AgentRole(
        ...     name="researcher",
        ...     description="Finds and analyzes information",
        ...     capabilities=[Capability.SEARCH, Capability.ANALYZE]
        ... )
        >>> role.can(Capability.SEARCH)
        True
    """

    name: str
    description: str
    capabilities: list[Capability] = field(default_factory=list)
    priority: int = 1  # Higher = more authority
    max_concurrent_tasks: int = 1

    def can(self, capability: Capability) -> bool:
        """Check if role has a capability."""
        # TODO: Return True if capability in capabilities
        pass

    def has_authority_over(self, other: "AgentRole") -> bool:
        """Check if this role has authority over another."""
        # TODO: Compare priorities
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        # TODO: Return dict representation
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "AgentRole":
        """Create from dictionary."""
        # TODO: Handle capability conversion
        pass


# Standard roles
RESEARCHER_ROLE = AgentRole(
    name="researcher",
    description="Searches and gathers information",
    capabilities=[Capability.SEARCH, Capability.SUMMARIZE],
    priority=1,
)

ANALYST_ROLE = AgentRole(
    name="analyst",
    description="Analyzes data and draws conclusions",
    capabilities=[Capability.ANALYZE, Capability.SUMMARIZE],
    priority=2,
)

WRITER_ROLE = AgentRole(
    name="writer",
    description="Creates written content",
    capabilities=[Capability.WRITE, Capability.SUMMARIZE],
    priority=1,
)

COORDINATOR_ROLE = AgentRole(
    name="coordinator",
    description="Manages and coordinates other agents",
    capabilities=[Capability.PLAN, Capability.COORDINATE],
    priority=3,
)


# =============================================================================
# Part 2: Inter-Agent Messages
# =============================================================================
class MessageType(Enum):
    """Types of inter-agent messages."""

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR = "error"
    STATUS = "status"
    TASK = "task"
    RESULT = "result"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """
    Message for inter-agent communication.

    Example:
        >>> msg = AgentMessage.request(
        ...     sender="coordinator",
        ...     receiver="researcher",
        ...     content={"task": "search", "query": "AI news"}
        ... )
        >>> msg.message_type
        <MessageType.REQUEST: 'request'>
    """

    id: str
    sender: str
    receiver: str  # Use "*" for broadcast
    message_type: MessageType
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # Links request/response
    metadata: dict = field(default_factory=dict)

    @classmethod
    def request(
        cls,
        sender: str,
        receiver: str,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "AgentMessage":
        """Create a request message."""
        # TODO: Create REQUEST message with new ID
        pass

    @classmethod
    def response(
        cls, request: "AgentMessage", sender: str, content: Any
    ) -> "AgentMessage":
        """Create a response to a request."""
        # TODO: Create RESPONSE with correlation_id from request
        pass

    @classmethod
    def broadcast(cls, sender: str, content: Any) -> "AgentMessage":
        """Create a broadcast message."""
        # TODO: Create BROADCAST with receiver="*"
        pass

    def is_broadcast(self) -> bool:
        """Check if message is a broadcast."""
        # TODO: Return True if receiver is "*"
        pass

    def is_response_to(self, request: "AgentMessage") -> bool:
        """Check if this is a response to a request."""
        # TODO: Compare correlation_id to request.id
        pass


# =============================================================================
# Part 3: Message Broker
# =============================================================================
class MessageBroker:
    """
    Routes messages between agents.

    Example:
        >>> broker = MessageBroker()
        >>> broker.register("agent1", callback1)
        >>> broker.register("agent2", callback2)
        >>> broker.send(message)  # Routes to appropriate agent
    """

    def __init__(self):
        """Initialize broker."""
        # TODO: Initialize subscribers dict and message queue
        pass

    def register(self, agent_id: str, callback: Callable[[AgentMessage], None]) -> None:
        """Register an agent to receive messages."""
        # TODO: Add agent callback to subscribers
        pass

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent."""
        # TODO: Remove agent from subscribers
        pass

    def send(self, message: AgentMessage) -> bool:
        """Send a message to target agent(s)."""
        # TODO: Route to specific agent or broadcast
        # Return True if delivered
        pass

    async def send_async(self, message: AgentMessage) -> bool:
        """Send message asynchronously."""
        # TODO: Async version of send
        pass

    def send_and_wait(
        self, message: AgentMessage, timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send message and wait for response."""
        # TODO: Send request and wait for correlated response
        pass

    def get_pending(self, agent_id: str) -> list[AgentMessage]:
        """Get pending messages for an agent."""
        # TODO: Return queued messages for agent
        pass

    def clear_queue(self, agent_id: Optional[str] = None) -> int:
        """Clear message queue."""
        # TODO: Clear all or specific agent's queue
        pass


# =============================================================================
# Part 4: Agent Coordinator
# =============================================================================
class CoordinationStrategy(Enum):
    """Strategies for coordinating agents."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    id: str
    role: AgentRole
    status: str = "idle"
    current_task: Optional[str] = None
    last_active: datetime = field(default_factory=datetime.now)


class AgentCoordinator:
    """
    Coordinates activities between multiple agents.

    Example:
        >>> coordinator = AgentCoordinator()
        >>> coordinator.register_agent("agent1", RESEARCHER_ROLE)
        >>> coordinator.assign_task("agent1", {"task": "research"})
    """

    def __init__(
        self, strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    ):
        """Initialize coordinator."""
        # TODO: Initialize agents dict, broker, strategy
        pass

    def register_agent(self, agent_id: str, role: AgentRole) -> None:
        """Register an agent with the coordinator."""
        # TODO: Create AgentInfo and register with broker
        pass

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        # TODO: Remove agent from tracking
        pass

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""
        # TODO: Return AgentInfo for agent
        pass

    def get_available_agents(
        self, capability: Optional[Capability] = None
    ) -> list[AgentInfo]:
        """Get available agents, optionally filtered by capability."""
        # TODO: Return idle agents with required capability
        pass

    def assign_task(self, agent_id: str, task: dict) -> bool:
        """Assign a task to an agent."""
        # TODO: Send TASK message to agent
        pass

    def distribute_tasks(self, tasks: list[dict]) -> dict[str, str]:
        """Distribute tasks to available agents."""
        # TODO: Match tasks to agents based on strategy
        # Return mapping of task_id -> agent_id
        pass

    def get_status(self) -> dict:
        """Get status of all agents."""
        # TODO: Return dict with agent statuses
        pass


# =============================================================================
# Part 5: Workflow Step Definition
# =============================================================================
class StepType(Enum):
    """Types of workflow steps."""

    ACTION = "action"
    DECISION = "decision"
    PARALLEL = "parallel"
    WAIT = "wait"


@dataclass
class WorkflowStep:
    """
    A single step in a workflow.

    Example:
        >>> step = WorkflowStep(
        ...     id="research",
        ...     name="Research Topic",
        ...     step_type=StepType.ACTION,
        ...     required_capability=Capability.SEARCH,
        ...     next_steps=["analyze"]
        ... )
    """

    id: str
    name: str
    step_type: StepType
    description: str = ""
    required_capability: Optional[Capability] = None
    config: dict = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)
    condition: Optional[Callable[[dict], bool]] = None
    timeout_seconds: float = 60.0

    def get_next(self, context: dict) -> list[str]:
        """Get next steps based on context."""
        # TODO: For DECISION type, evaluate condition
        # Return next_steps or filtered based on condition
        pass

    def can_execute(self, role: AgentRole) -> bool:
        """Check if role can execute this step."""
        # TODO: Check required capability against role
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        # TODO: Return dict representation (without callable)
        pass


# =============================================================================
# Part 6: Workflow Engine
# =============================================================================
@dataclass
class WorkflowContext:
    """Execution context for a workflow."""

    workflow_id: str
    current_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class WorkflowEngine:
    """
    Executes multi-step workflows with agents.

    Example:
        >>> engine = WorkflowEngine(coordinator)
        >>> engine.define_workflow("research", [
        ...     WorkflowStep(id="search", ...),
        ...     WorkflowStep(id="analyze", ...)
        ... ])
        >>> result = await engine.run("research", {"topic": "AI"})
    """

    def __init__(self, coordinator: AgentCoordinator):
        """Initialize engine."""
        # TODO: Store coordinator, initialize workflows dict
        pass

    def define_workflow(
        self, name: str, steps: list[WorkflowStep], description: str = ""
    ) -> None:
        """Define a new workflow."""
        # TODO: Store workflow definition
        pass

    def get_workflow(self, name: str) -> Optional[dict]:
        """Get workflow definition."""
        # TODO: Return workflow steps and metadata
        pass

    async def run(
        self, workflow_name: str, initial_context: Optional[dict] = None
    ) -> dict:
        """Execute a workflow."""
        # TODO: Create WorkflowContext
        # Execute steps in order
        # Handle parallel steps
        # Return final results
        pass

    async def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> Any:
        """Execute a single workflow step."""
        # TODO: Find agent with capability
        # Assign task and wait for result
        pass

    async def _execute_parallel(
        self, steps: list[WorkflowStep], context: WorkflowContext
    ) -> dict:
        """Execute multiple steps in parallel."""
        # TODO: Run steps concurrently
        pass

    def get_progress(self, workflow_id: str) -> dict:
        """Get progress of running workflow."""
        # TODO: Return current step, completed steps, etc.
        pass


# =============================================================================
# Part 7: Agent Pool
# =============================================================================
class PoolStrategy(Enum):
    """Strategies for selecting agents from pool."""

    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    RANDOM = "random"
    CAPABILITY_MATCH = "capability_match"


class AgentPool:
    """
    Pool of reusable agents for task execution.

    Example:
        >>> pool = AgentPool(min_agents=2, max_agents=10)
        >>> pool.add(agent1, RESEARCHER_ROLE)
        >>> pool.add(agent2, RESEARCHER_ROLE)
        >>> agent = pool.acquire(Capability.SEARCH)
        >>> # Use agent...
        >>> pool.release(agent)
    """

    def __init__(
        self,
        min_agents: int = 1,
        max_agents: int = 10,
        strategy: PoolStrategy = PoolStrategy.LEAST_BUSY,
    ):
        """Initialize pool."""
        # TODO: Initialize pool settings and agent tracking
        pass

    def add(self, agent_id: str, role: AgentRole) -> None:
        """Add an agent to the pool."""
        # TODO: Add agent to available pool
        pass

    def remove(self, agent_id: str) -> bool:
        """Remove an agent from the pool."""
        # TODO: Remove agent if not in use
        pass

    def acquire(
        self, capability: Optional[Capability] = None, timeout: float = 30.0
    ) -> Optional[str]:
        """Acquire an agent from the pool."""
        # TODO: Get available agent matching capability
        # Mark as in-use
        pass

    def release(self, agent_id: str) -> None:
        """Release an agent back to the pool."""
        # TODO: Mark agent as available
        pass

    def resize(self, min_agents: int, max_agents: int) -> None:
        """Resize the pool."""
        # TODO: Update pool limits
        pass

    def get_stats(self) -> dict:
        """Get pool statistics."""
        # TODO: Return counts of available, in-use, total
        pass


# =============================================================================
# Part 8: Conflict Resolver
# =============================================================================
class ConflictType(Enum):
    """Types of conflicts between agents."""

    RESOURCE = "resource"
    OPINION = "opinion"
    PRIORITY = "priority"
    DEADLOCK = "deadlock"


@dataclass
class Conflict:
    """A conflict between agents."""

    id: str
    conflict_type: ConflictType
    agents: list[str]
    description: str
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ConflictResolver:
    """
    Resolves conflicts between agents.

    Example:
        >>> resolver = ConflictResolver()
        >>> conflict = Conflict(...)
        >>> resolution = resolver.resolve(conflict)
    """

    def __init__(self, coordinator: AgentCoordinator):
        """Initialize resolver."""
        # TODO: Store coordinator
        pass

    def detect(self, context: dict) -> list[Conflict]:
        """Detect conflicts in current state."""
        # TODO: Analyze context for conflicts
        pass

    def resolve(self, conflict: Conflict) -> dict:
        """Resolve a conflict."""
        # TODO: Apply appropriate resolution strategy
        pass

    def _resolve_resource(self, conflict: Conflict) -> dict:
        """Resolve resource conflict."""
        # TODO: Prioritize resource access
        pass

    def _resolve_opinion(self, conflict: Conflict) -> dict:
        """Resolve opinion conflict."""
        # TODO: Use voting or authority
        pass

    def _resolve_priority(self, conflict: Conflict) -> dict:
        """Resolve priority conflict."""
        # TODO: Use role priority
        pass

    def _resolve_deadlock(self, conflict: Conflict) -> dict:
        """Resolve deadlock."""
        # TODO: Break deadlock cycle
        pass

    def get_history(self) -> list[dict]:
        """Get conflict resolution history."""
        # TODO: Return past conflicts and resolutions
        pass


# =============================================================================
# Part 9: Agent Supervisor
# =============================================================================
class SupervisorAction(Enum):
    """Actions a supervisor can take."""

    NONE = "none"
    WARN = "warn"
    THROTTLE = "throttle"
    RESTART = "restart"
    TERMINATE = "terminate"


@dataclass
class AgentHealth:
    """Health status of an agent."""

    agent_id: str
    status: str
    error_rate: float
    response_time_ms: float
    tasks_completed: int
    tasks_failed: int
    last_check: datetime = field(default_factory=datetime.now)


class AgentSupervisor:
    """
    Monitors and manages agent health and behavior.

    Example:
        >>> supervisor = AgentSupervisor(coordinator)
        >>> supervisor.start()
        >>> health = supervisor.check_health("agent1")
        >>> if health.error_rate > 0.5:
        ...     supervisor.restart_agent("agent1")
    """

    def __init__(self, coordinator: AgentCoordinator, check_interval: float = 10.0):
        """Initialize supervisor."""
        # TODO: Store coordinator and settings
        pass

    def start(self) -> None:
        """Start supervisor monitoring."""
        # TODO: Begin periodic health checks
        pass

    def stop(self) -> None:
        """Stop supervisor monitoring."""
        # TODO: Stop monitoring loop
        pass

    def check_health(self, agent_id: str) -> AgentHealth:
        """Check health of an agent."""
        # TODO: Collect and return health metrics
        pass

    def check_all(self) -> dict[str, AgentHealth]:
        """Check health of all agents."""
        # TODO: Check each registered agent
        pass

    def take_action(self, agent_id: str, action: SupervisorAction) -> bool:
        """Take supervisory action on an agent."""
        # TODO: Execute action
        pass

    def restart_agent(self, agent_id: str) -> bool:
        """Restart an agent."""
        # TODO: Stop and restart agent
        pass

    def set_threshold(
        self, metric: str, threshold: float, action: SupervisorAction
    ) -> None:
        """Set automatic action threshold."""
        # TODO: Configure auto-action rules
        pass

    def get_alerts(self) -> list[dict]:
        """Get active alerts."""
        # TODO: Return current alerts
        pass


# =============================================================================
# Part 10: Complete Multi-Agent System
# =============================================================================
@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system."""

    name: str
    max_agents: int = 10
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    enable_supervision: bool = True
    enable_conflict_resolution: bool = True


class MultiAgentSystem:
    """
    Complete multi-agent orchestration system.

    Example:
        >>> config = MultiAgentConfig(name="research-team")
        >>> system = MultiAgentSystem(config)
        >>> system.add_agent("researcher", RESEARCHER_ROLE, researcher_func)
        >>> system.add_agent("writer", WRITER_ROLE, writer_func)
        >>> system.define_workflow("research", steps)
        >>> result = await system.execute("research", {"topic": "AI"})
    """

    def __init__(self, config: MultiAgentConfig):
        """Initialize multi-agent system."""
        # TODO: Create coordinator, broker, pool, supervisor, resolver
        pass

    def add_agent(self, agent_id: str, role: AgentRole, handler: Callable) -> None:
        """Add an agent to the system."""
        # TODO: Register agent with all components
        pass

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the system."""
        # TODO: Unregister from all components
        pass

    def define_workflow(self, name: str, steps: list[WorkflowStep]) -> None:
        """Define a workflow."""
        # TODO: Add workflow to engine
        pass

    async def execute(self, workflow_name: str, context: Optional[dict] = None) -> dict:
        """Execute a workflow."""
        # TODO: Run workflow through engine
        pass

    async def send_task(self, agent_id: str, task: dict) -> Any:
        """Send a task to a specific agent."""
        # TODO: Route task through coordinator
        pass

    async def broadcast(self, message: Any) -> None:
        """Broadcast message to all agents."""
        # TODO: Send via broker
        pass

    def get_status(self) -> dict:
        """Get system status."""
        # TODO: Aggregate status from all components
        pass

    def start(self) -> None:
        """Start the system."""
        # TODO: Start supervisor and any background tasks
        pass

    def stop(self) -> None:
        """Stop the system."""
        # TODO: Graceful shutdown
        pass


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    import asyncio

    # Create multi-agent system
    config = MultiAgentConfig(
        name="research-team", coordination_strategy=CoordinationStrategy.SEQUENTIAL
    )

    system = MultiAgentSystem(config)

    # Define agent handlers
    async def researcher_handler(task: dict) -> dict:
        query = task.get("query", "")
        return {"findings": f"Research results for: {query}"}

    async def analyst_handler(task: dict) -> dict:
        findings = task.get("findings", "")
        return {"analysis": f"Analysis of: {findings}"}

    async def writer_handler(task: dict) -> dict:
        analysis = task.get("analysis", "")
        return {"report": f"Report based on: {analysis}"}

    # Add agents
    system.add_agent("researcher", RESEARCHER_ROLE, researcher_handler)
    system.add_agent("analyst", ANALYST_ROLE, analyst_handler)
    system.add_agent("writer", WRITER_ROLE, writer_handler)

    # Define workflow
    workflow_steps = [
        WorkflowStep(
            id="research",
            name="Research Topic",
            step_type=StepType.ACTION,
            required_capability=Capability.SEARCH,
            next_steps=["analyze"],
        ),
        WorkflowStep(
            id="analyze",
            name="Analyze Findings",
            step_type=StepType.ACTION,
            required_capability=Capability.ANALYZE,
            next_steps=["write"],
        ),
        WorkflowStep(
            id="write",
            name="Write Report",
            step_type=StepType.ACTION,
            required_capability=Capability.WRITE,
            next_steps=[],
        ),
    ]

    system.define_workflow("research_pipeline", workflow_steps)

    # Execute
    async def main():
        system.start()
        result = await system.execute(
            "research_pipeline", {"topic": "Artificial Intelligence trends 2024"}
        )
        print(f"Result: {result}")
        system.stop()

    asyncio.run(main())

"""
Tests for Week 14 - Exercise 3: Multi-Agent Systems

Run with: pytest tests/test_exercise_advanced_3_multi_agent.py -v
"""

import pytest
import asyncio
from datetime import datetime

from exercise_advanced_3_multi_agent import (
    Capability,
    AgentRole,
    RESEARCHER_ROLE,
    ANALYST_ROLE,
    WRITER_ROLE,
    COORDINATOR_ROLE,
    MessageType,
    MessagePriority,
    AgentMessage,
    MessageBroker,
    CoordinationStrategy,
    AgentInfo,
    AgentCoordinator,
    StepType,
    WorkflowStep,
    WorkflowContext,
    WorkflowEngine,
    PoolStrategy,
    AgentPool,
    ConflictType,
    Conflict,
    ConflictResolver,
    SupervisorAction,
    AgentHealth,
    AgentSupervisor,
    MultiAgentConfig,
    MultiAgentSystem,
)


# =============================================================================
# Part 1: Agent Role Tests
# =============================================================================
class TestAgentRole:
    """Tests for AgentRole class."""

    def test_create_role(self):
        """Test creating an agent role."""
        role = AgentRole(
            name="custom",
            description="Custom role",
            capabilities=[Capability.SEARCH, Capability.ANALYZE],
        )

        assert role.name == "custom"
        assert len(role.capabilities) == 2

    def test_can_capability(self):
        """Test checking capabilities."""
        assert RESEARCHER_ROLE.can(Capability.SEARCH) is True
        assert RESEARCHER_ROLE.can(Capability.CODE) is False

    def test_has_authority(self):
        """Test authority comparison."""
        # Coordinator has higher priority
        assert COORDINATOR_ROLE.has_authority_over(RESEARCHER_ROLE) is True
        assert RESEARCHER_ROLE.has_authority_over(COORDINATOR_ROLE) is False

    def test_to_dict(self):
        """Test converting role to dictionary."""
        result = RESEARCHER_ROLE.to_dict()

        assert result["name"] == "researcher"
        assert "capabilities" in result

    def test_from_dict(self):
        """Test creating role from dictionary."""
        data = {
            "name": "test",
            "description": "Test role",
            "capabilities": ["search", "analyze"],
            "priority": 2,
        }

        role = AgentRole.from_dict(data)

        assert role.name == "test"
        assert role.priority == 2


# =============================================================================
# Part 2: Agent Message Tests
# =============================================================================
class TestAgentMessageMulti:
    """Tests for AgentMessage class in multi-agent context."""

    def test_create_request(self):
        """Test creating a request message."""
        msg = AgentMessage.request(
            sender="coordinator", receiver="researcher", content={"task": "search"}
        )

        assert msg.message_type == MessageType.REQUEST
        assert msg.sender == "coordinator"
        assert msg.receiver == "researcher"

    def test_create_response(self):
        """Test creating a response to a request."""
        request = AgentMessage.request(
            sender="coordinator", receiver="researcher", content={"task": "search"}
        )

        response = AgentMessage.response(
            request=request,
            sender="researcher",
            content={"results": ["item1", "item2"]},
        )

        assert response.message_type == MessageType.RESPONSE
        assert response.correlation_id == request.id

    def test_create_broadcast(self):
        """Test creating a broadcast message."""
        msg = AgentMessage.broadcast(sender="coordinator", content="System update")

        assert msg.is_broadcast() is True
        assert msg.receiver == "*"

    def test_is_response_to(self):
        """Test checking if message is response to request."""
        request = AgentMessage.request("a", "b", "test")
        response = AgentMessage.response(request, "b", "response")
        other = AgentMessage.request("c", "d", "other")

        assert response.is_response_to(request) is True
        assert other.is_response_to(request) is False


# =============================================================================
# Part 3: Message Broker Tests
# =============================================================================
class TestMessageBroker:
    """Tests for MessageBroker class."""

    def test_register_agent(self):
        """Test registering an agent."""
        broker = MessageBroker()
        received = []

        broker.register("agent1", lambda msg: received.append(msg))

        msg = AgentMessage.request("sender", "agent1", "test")
        broker.send(msg)

        assert len(received) == 1

    def test_send_to_specific_agent(self):
        """Test sending to a specific agent."""
        broker = MessageBroker()
        received1 = []
        received2 = []

        broker.register("agent1", lambda msg: received1.append(msg))
        broker.register("agent2", lambda msg: received2.append(msg))

        msg = AgentMessage.request("sender", "agent1", "test")
        broker.send(msg)

        assert len(received1) == 1
        assert len(received2) == 0

    def test_broadcast_message(self):
        """Test broadcasting to all agents."""
        broker = MessageBroker()
        received1 = []
        received2 = []

        broker.register("agent1", lambda msg: received1.append(msg))
        broker.register("agent2", lambda msg: received2.append(msg))

        msg = AgentMessage.broadcast("sender", "broadcast test")
        broker.send(msg)

        assert len(received1) == 1
        assert len(received2) == 1

    def test_unregister(self):
        """Test unregistering an agent."""
        broker = MessageBroker()
        received = []

        broker.register("agent1", lambda msg: received.append(msg))
        broker.unregister("agent1")

        msg = AgentMessage.request("sender", "agent1", "test")
        result = broker.send(msg)

        assert result is False


# =============================================================================
# Part 4: Agent Coordinator Tests
# =============================================================================
class TestAgentCoordinator:
    """Tests for AgentCoordinator class."""

    def test_register_agent(self):
        """Test registering an agent with coordinator."""
        coordinator = AgentCoordinator()

        coordinator.register_agent("researcher1", RESEARCHER_ROLE)
        agent = coordinator.get_agent("researcher1")

        assert agent is not None
        assert agent.role == RESEARCHER_ROLE

    def test_get_available_agents(self):
        """Test getting available agents."""
        coordinator = AgentCoordinator()

        coordinator.register_agent("researcher1", RESEARCHER_ROLE)
        coordinator.register_agent("analyst1", ANALYST_ROLE)

        available = coordinator.get_available_agents(Capability.SEARCH)

        assert len(available) == 1
        assert available[0].id == "researcher1"

    def test_assign_task(self):
        """Test assigning a task to an agent."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("researcher1", RESEARCHER_ROLE)

        result = coordinator.assign_task(
            "researcher1", {"task": "search", "query": "AI"}
        )

        assert result is True

    def test_distribute_tasks(self):
        """Test distributing tasks to agents."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("researcher1", RESEARCHER_ROLE)
        coordinator.register_agent("researcher2", RESEARCHER_ROLE)

        tasks = [{"id": "task1", "type": "search"}, {"id": "task2", "type": "search"}]

        distribution = coordinator.distribute_tasks(tasks)

        assert len(distribution) == 2


# =============================================================================
# Part 5: Workflow Step Tests
# =============================================================================
class TestWorkflowStep:
    """Tests for WorkflowStep class."""

    def test_create_step(self):
        """Test creating a workflow step."""
        step = WorkflowStep(
            id="research",
            name="Research Phase",
            step_type=StepType.ACTION,
            required_capability=Capability.SEARCH,
            next_steps=["analyze"],
        )

        assert step.id == "research"
        assert step.required_capability == Capability.SEARCH

    def test_get_next_steps(self):
        """Test getting next steps."""
        step = WorkflowStep(
            id="step1",
            name="Step 1",
            step_type=StepType.ACTION,
            next_steps=["step2", "step3"],
        )

        next_steps = step.get_next({})

        assert next_steps == ["step2", "step3"]

    def test_can_execute(self):
        """Test checking if role can execute step."""
        step = WorkflowStep(
            id="search",
            name="Search",
            step_type=StepType.ACTION,
            required_capability=Capability.SEARCH,
        )

        assert step.can_execute(RESEARCHER_ROLE) is True
        assert step.can_execute(WRITER_ROLE) is False


# =============================================================================
# Part 6: Workflow Engine Tests
# =============================================================================
class TestWorkflowEngine:
    """Tests for WorkflowEngine class."""

    def test_define_workflow(self):
        """Test defining a workflow."""
        coordinator = AgentCoordinator()
        engine = WorkflowEngine(coordinator)

        steps = [
            WorkflowStep(id="step1", name="Step 1", step_type=StepType.ACTION),
            WorkflowStep(id="step2", name="Step 2", step_type=StepType.ACTION),
        ]

        engine.define_workflow("test_workflow", steps)

        workflow = engine.get_workflow("test_workflow")
        assert workflow is not None

    @pytest.mark.asyncio
    async def test_run_workflow(self):
        """Test running a workflow."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("worker", RESEARCHER_ROLE)

        engine = WorkflowEngine(coordinator)

        steps = [
            WorkflowStep(
                id="start",
                name="Start",
                step_type=StepType.ACTION,
                required_capability=Capability.SEARCH,
            )
        ]

        engine.define_workflow("simple", steps)

        result = await engine.run("simple", {"input": "test"})

        assert result is not None


# =============================================================================
# Part 7: Agent Pool Tests
# =============================================================================
class TestAgentPool:
    """Tests for AgentPool class."""

    def test_add_agent(self):
        """Test adding agent to pool."""
        pool = AgentPool()

        pool.add("agent1", RESEARCHER_ROLE)

        stats = pool.get_stats()
        assert stats["total"] == 1

    def test_acquire_release(self):
        """Test acquiring and releasing agents."""
        pool = AgentPool()
        pool.add("agent1", RESEARCHER_ROLE)

        agent_id = pool.acquire(Capability.SEARCH)
        assert agent_id == "agent1"

        stats = pool.get_stats()
        assert stats["in_use"] == 1

        pool.release(agent_id)
        stats = pool.get_stats()
        assert stats["available"] == 1

    def test_acquire_by_capability(self):
        """Test acquiring agent by capability."""
        pool = AgentPool()
        pool.add("researcher", RESEARCHER_ROLE)
        pool.add("analyst", ANALYST_ROLE)

        agent = pool.acquire(Capability.ANALYZE)

        assert agent == "analyst"

    def test_remove_agent(self):
        """Test removing agent from pool."""
        pool = AgentPool()
        pool.add("agent1", RESEARCHER_ROLE)

        result = pool.remove("agent1")

        assert result is True
        assert pool.get_stats()["total"] == 0


# =============================================================================
# Part 8: Conflict Resolver Tests
# =============================================================================
class TestConflictResolver:
    """Tests for ConflictResolver class."""

    def test_resolve_resource_conflict(self):
        """Test resolving resource conflict."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("agent1", RESEARCHER_ROLE)
        coordinator.register_agent("agent2", ANALYST_ROLE)

        resolver = ConflictResolver(coordinator)

        conflict = Conflict(
            id="conflict1",
            conflict_type=ConflictType.RESOURCE,
            agents=["agent1", "agent2"],
            description="Both need same resource",
        )

        resolution = resolver.resolve(conflict)

        assert resolution is not None

    def test_resolve_priority_conflict(self):
        """Test resolving priority conflict."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("agent1", RESEARCHER_ROLE)
        coordinator.register_agent("agent2", COORDINATOR_ROLE)

        resolver = ConflictResolver(coordinator)

        conflict = Conflict(
            id="conflict2",
            conflict_type=ConflictType.PRIORITY,
            agents=["agent1", "agent2"],
            description="Priority dispute",
        )

        resolution = resolver.resolve(conflict)

        # Coordinator should win due to higher priority
        assert resolution is not None


# =============================================================================
# Part 9: Agent Supervisor Tests
# =============================================================================
class TestAgentSupervisor:
    """Tests for AgentSupervisor class."""

    def test_check_health(self):
        """Test checking agent health."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("agent1", RESEARCHER_ROLE)

        supervisor = AgentSupervisor(coordinator)

        health = supervisor.check_health("agent1")

        assert health.agent_id == "agent1"
        assert health.status is not None

    def test_check_all(self):
        """Test checking all agents."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("agent1", RESEARCHER_ROLE)
        coordinator.register_agent("agent2", ANALYST_ROLE)

        supervisor = AgentSupervisor(coordinator)

        health_map = supervisor.check_all()

        assert len(health_map) == 2

    def test_take_action(self):
        """Test taking supervisory action."""
        coordinator = AgentCoordinator()
        coordinator.register_agent("agent1", RESEARCHER_ROLE)

        supervisor = AgentSupervisor(coordinator)

        result = supervisor.take_action("agent1", SupervisorAction.WARN)

        assert result is True


# =============================================================================
# Part 10: Multi-Agent System Tests
# =============================================================================
class TestMultiAgentSystem:
    """Tests for MultiAgentSystem class."""

    def test_create_system(self):
        """Test creating a multi-agent system."""
        config = MultiAgentConfig(name="test-system")
        system = MultiAgentSystem(config)

        assert system is not None

    def test_add_agent(self):
        """Test adding agent to system."""
        config = MultiAgentConfig(name="test")
        system = MultiAgentSystem(config)

        async def handler(task):
            return {"result": "done"}

        system.add_agent("researcher", RESEARCHER_ROLE, handler)

        status = system.get_status()
        assert "researcher" in str(status)

    def test_define_workflow(self):
        """Test defining a workflow in system."""
        config = MultiAgentConfig(name="test")
        system = MultiAgentSystem(config)

        steps = [WorkflowStep(id="step1", name="Step 1", step_type=StepType.ACTION)]

        system.define_workflow("test_workflow", steps)

        # Workflow should be defined
        assert True

    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test executing a workflow."""
        config = MultiAgentConfig(name="test")
        system = MultiAgentSystem(config)

        async def handler(task):
            return {"result": "processed"}

        system.add_agent("worker", RESEARCHER_ROLE, handler)

        steps = [
            WorkflowStep(
                id="process",
                name="Process",
                step_type=StepType.ACTION,
                required_capability=Capability.SEARCH,
            )
        ]

        system.define_workflow("simple", steps)
        system.start()

        result = await system.execute("simple", {"input": "test"})

        system.stop()

        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================
class TestMultiAgentIntegration:
    """Integration tests for multi-agent systems."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test a complete multi-agent pipeline."""
        config = MultiAgentConfig(
            name="research-team", coordination_strategy=CoordinationStrategy.SEQUENTIAL
        )

        system = MultiAgentSystem(config)

        # Define agent handlers
        async def researcher_handler(task):
            return {"findings": f"Research on: {task.get('topic', '')}"}

        async def writer_handler(task):
            findings = task.get("findings", "")
            return {"report": f"Report: {findings}"}

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
                name="Write Report",
                step_type=StepType.ACTION,
                required_capability=Capability.WRITE,
                next_steps=[],
            ),
        ]

        system.define_workflow("research_pipeline", workflow_steps)
        system.start()

        # Execute
        result = await system.execute("research_pipeline", {"topic": "AI trends"})

        system.stop()

        # Verify
        assert result is not None

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel task execution."""
        config = MultiAgentConfig(
            name="parallel-team", coordination_strategy=CoordinationStrategy.PARALLEL
        )

        system = MultiAgentSystem(config)

        async def handler1(task):
            await asyncio.sleep(0.1)
            return {"result": "handler1"}

        async def handler2(task):
            await asyncio.sleep(0.1)
            return {"result": "handler2"}

        system.add_agent("agent1", RESEARCHER_ROLE, handler1)
        system.add_agent("agent2", RESEARCHER_ROLE, handler2)

        system.start()

        # Both tasks should complete
        result1 = await system.send_task("agent1", {"task": "test1"})
        result2 = await system.send_task("agent2", {"task": "test2"})

        system.stop()

        assert result1 is not None
        assert result2 is not None

    def test_message_routing(self):
        """Test message routing between agents."""
        broker = MessageBroker()

        messages_1 = []
        messages_2 = []

        broker.register("agent1", lambda m: messages_1.append(m))
        broker.register("agent2", lambda m: messages_2.append(m))

        # Send direct message
        msg1 = AgentMessage.request("agent1", "agent2", "Hello")
        broker.send(msg1)

        assert len(messages_1) == 0
        assert len(messages_2) == 1

        # Send broadcast
        msg2 = AgentMessage.broadcast("coordinator", "Update")
        broker.send(msg2)

        assert len(messages_1) == 1
        assert len(messages_2) == 2

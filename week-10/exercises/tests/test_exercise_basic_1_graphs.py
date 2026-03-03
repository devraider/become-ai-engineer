"""
Tests for Week 10 - Exercise 1: Basic Graph Building
"""

import pytest
from exercise_basic_1_graphs import (
    BasicState,
    AnnotatedState,
    process_input_node,
    transform_node,
    SimpleGraph,
    merge_state,
    PipelineState,
    count_words,
    count_chars,
    summarize,
    ConfigurableGraph,
    ParallelState,
    branch_a,
    branch_b,
    branch_c,
    combine_results,
    ParallelGraph,
    GraphFactory,
)


class TestBasicState:
    """Tests for Task 1: Basic State."""

    def test_basic_state_has_required_fields(self):
        """BasicState should have input, output, and steps fields."""
        state: BasicState = {"input": "test", "output": "", "steps": []}
        assert "input" in state
        assert "output" in state
        assert "steps" in state

    def test_basic_state_types(self):
        """BasicState fields should have correct types."""
        state: BasicState = {"input": "hello", "output": "world", "steps": ["a", "b"]}
        assert isinstance(state["input"], str)
        assert isinstance(state["output"], str)
        assert isinstance(state["steps"], list)


class TestProcessInputNode:
    """Tests for Task 3: Process Input Node."""

    def test_process_input_uppercase(self):
        """process_input_node should convert input to uppercase."""
        state = {"input": "hello world", "output": "", "steps": []}
        result = process_input_node(state)
        assert result["output"] == "HELLO WORLD"

    def test_process_input_adds_step(self):
        """process_input_node should add 'processed' to steps."""
        state = {"input": "test", "output": "", "steps": []}
        result = process_input_node(state)
        assert "processed" in result["steps"]


class TestTransformNode:
    """Tests for Task 4: Transform Node."""

    def test_transform_reverses_output(self):
        """transform_node should reverse the output string."""
        state = {"input": "hello", "output": "HELLO", "steps": ["processed"]}
        result = transform_node(state)
        assert result["output"] == "OLLEH"

    def test_transform_adds_step(self):
        """transform_node should add 'transformed' to steps."""
        state = {"input": "hello", "output": "HELLO", "steps": []}
        result = transform_node(state)
        assert "transformed" in result["steps"]


class TestSimpleGraph:
    """Tests for Task 5: Simple Graph."""

    def test_graph_creation(self):
        """SimpleGraph should be creatable."""
        graph = SimpleGraph()
        assert graph is not None

    def test_graph_invoke(self):
        """SimpleGraph should process input through all nodes."""
        graph = SimpleGraph()
        result = graph.invoke({"input": "hello", "output": "", "steps": []})
        # After process: HELLO, after transform: OLLEH
        assert result["output"] == "OLLEH"
        assert "processed" in result["steps"]
        assert "transformed" in result["steps"]


class TestMergeState:
    """Tests for Task 6: State Merging."""

    def test_merge_lists(self):
        """merge_state should append lists."""
        current = {"a": [1, 2]}
        update = {"a": [3]}
        result = merge_state(current, update)
        assert result["a"] == [1, 2, 3]

    def test_merge_dicts(self):
        """merge_state should merge dictionaries."""
        current = {"a": {"x": 1}}
        update = {"a": {"y": 2}}
        result = merge_state(current, update)
        assert result["a"] == {"x": 1, "y": 2}

    def test_merge_replace_scalars(self):
        """merge_state should replace scalar values."""
        current = {"a": "old"}
        update = {"a": "new"}
        result = merge_state(current, update)
        assert result["a"] == "new"

    def test_merge_preserves_unchanged(self):
        """merge_state should preserve unchanged keys."""
        current = {"a": 1, "b": 2}
        update = {"a": 10}
        result = merge_state(current, update)
        assert result["b"] == 2
        assert result["a"] == 10


class TestPipelineNodes:
    """Tests for Task 7: Pipeline Nodes."""

    def test_count_words(self):
        """count_words should count words correctly."""
        state: PipelineState = {
            "text": "hello world test",
            "steps_completed": [],
            "word_count": 0,
            "char_count": 0,
        }
        result = count_words(state)
        assert result["word_count"] == 3
        assert "word_count" in result["steps_completed"]

    def test_count_chars(self):
        """count_chars should count non-space characters."""
        state: PipelineState = {
            "text": "hello world",
            "steps_completed": [],
            "word_count": 0,
            "char_count": 0,
        }
        result = count_chars(state)
        assert result["char_count"] == 10  # "helloworld" without space

    def test_summarize(self):
        """summarize should create summary string."""
        state: PipelineState = {
            "text": "",
            "steps_completed": [],
            "word_count": 5,
            "char_count": 20,
        }
        result = summarize(state)
        assert "5 words" in result["text"]
        assert "20 characters" in result["text"]


class TestConfigurableGraph:
    """Tests for Task 8: Configurable Graph."""

    def test_configurable_graph_chaining(self):
        """ConfigurableGraph methods should support chaining."""
        graph = ConfigurableGraph(BasicState)
        result = graph.add_node("test", lambda x: x).add_edge("test", "end")
        assert result is graph

    def test_configurable_graph_execution(self):
        """ConfigurableGraph should execute nodes."""
        graph = ConfigurableGraph(BasicState)
        graph.add_node("upper", lambda s: {"output": s["input"].upper()})
        graph.set_entry_point("upper")
        graph.set_finish_point("upper")
        compiled = graph.compile()
        result = compiled.invoke({"input": "test", "output": "", "steps": []})
        assert result["output"] == "TEST"


class TestParallelGraph:
    """Tests for Task 9: Parallel Execution."""

    def test_branch_a(self):
        """branch_a should uppercase input."""
        state: ParallelState = {"input": "Hello", "results": {}, "final": ""}
        result = branch_a(state)
        assert result["results"]["a"] == "HELLO"

    def test_branch_b(self):
        """branch_b should lowercase input."""
        state: ParallelState = {"input": "Hello", "results": {}, "final": ""}
        result = branch_b(state)
        assert result["results"]["b"] == "hello"

    def test_branch_c(self):
        """branch_c should title case input."""
        state: ParallelState = {"input": "hello world", "results": {}, "final": ""}
        result = branch_c(state)
        assert result["results"]["c"] == "Hello World"

    def test_parallel_graph_combines_all(self):
        """ParallelGraph should run all branches and combine."""
        graph = ParallelGraph()
        graph.add_parallel_branch("a", branch_a)
        graph.add_parallel_branch("b", branch_b)
        graph.add_parallel_branch("c", branch_c)
        graph.set_combiner(combine_results)

        result = graph.invoke({"input": "Test", "results": {}, "final": ""})
        assert "TEST" in result["final"]
        assert "test" in result["final"]
        assert "Test" in result["final"]


class TestGraphFactory:
    """Tests for Task 10: Graph Factory."""

    def test_create_sequential(self):
        """GraphFactory.create_sequential should create linear graph."""
        nodes = [
            ("step1", lambda s: {"output": s["input"] + "_1"}),
            ("step2", lambda s: {"output": s["output"] + "_2"}),
        ]
        graph = GraphFactory.create_sequential(nodes, BasicState)
        result = graph.invoke({"input": "start", "output": "", "steps": []})
        assert result["output"] == "start_1_2"

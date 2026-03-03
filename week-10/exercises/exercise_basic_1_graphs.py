"""
Week 10 - Exercise 1: Basic Graph Building
===========================================
Learn to build fundamental LangGraph structures.

Topics covered:
- State definition with TypedDict
- Node functions
- Edge connections
- Graph compilation and execution
"""

from typing import TypedDict, Annotated, Any
from abc import ABC, abstractmethod


# =============================================================================
# TASK 1: Define a Basic State
# =============================================================================
class BasicState(TypedDict):
    """
    TODO: Define a state with:
    - input: str - The input text
    - output: str - The processed output
    - steps: list - List of processing steps taken

    This is the simplest form of state definition.
    """

    pass


# =============================================================================
# TASK 2: Create a State with Annotations
# =============================================================================
def list_reducer(existing: list, new: list) -> list:
    """Reducer that appends new items to existing list."""
    return existing + new


class AnnotatedState(TypedDict):
    """
    TODO: Define a state with a reducer annotation:
    - messages: list - Should use list_reducer to accumulate messages
    - current_node: str - Current node name (no reducer, just replace)
    - metadata: dict - Metadata dictionary

    Hint: Use Annotated[list, list_reducer] for the messages field
    """

    pass


# =============================================================================
# TASK 3: Implement a Node Function
# =============================================================================
def process_input_node(state: BasicState) -> dict:
    """
    TODO: Create a node that:
    1. Takes the input from state
    2. Converts it to uppercase
    3. Returns a dict with 'output' set to the uppercase text
    4. Appends "processed" to the steps list

    Remember: Nodes return partial state updates, not full state

    Example:
        state = {"input": "hello", "output": "", "steps": []}
        result = process_input_node(state)
        # result = {"output": "HELLO", "steps": ["processed"]}
    """
    pass


# =============================================================================
# TASK 4: Implement a Transformation Node
# =============================================================================
def transform_node(state: BasicState) -> dict:
    """
    TODO: Create a node that:
    1. Takes the current output from state
    2. Reverses the string
    3. Returns updated output
    4. Appends "transformed" to steps

    Example:
        state = {"input": "hello", "output": "HELLO", "steps": ["processed"]}
        result = transform_node(state)
        # result = {"output": "OLLEH", "steps": ["transformed"]}
    """
    pass


# =============================================================================
# TASK 5: Build a Simple Sequential Graph
# =============================================================================
class SimpleGraph:
    """
    TODO: Implement a simple graph that:
    1. Has two nodes: "process" and "transform"
    2. Flows: START -> process -> transform -> END
    3. Uses BasicState as the state schema

    Methods to implement:
    - __init__: Set up nodes and edges
    - compile: Return the compiled graph
    - invoke: Run the graph with input
    """

    def __init__(self):
        """Initialize the graph builder and add nodes/edges."""
        # TODO: Create StateGraph, add nodes, add edges
        self.nodes = {}
        self.edges = []
        pass

    def add_node(self, name: str, func: callable) -> None:
        """Add a node to the graph."""
        # TODO: Store the node
        pass

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between nodes."""
        # TODO: Store the edge
        pass

    def compile(self) -> "SimpleGraph":
        """Compile the graph for execution."""
        # TODO: Return self or compiled version
        return self

    def invoke(self, initial_state: dict) -> dict:
        """
        Execute the graph with the initial state.

        For this exercise, simulate execution by:
        1. Starting with initial_state
        2. Running each node in order (following edges)
        3. Merging node outputs into state
        4. Returning final state
        """
        # TODO: Execute nodes in order based on edges
        pass


# =============================================================================
# TASK 6: Implement State Merging
# =============================================================================
def merge_state(current_state: dict, update: dict) -> dict:
    """
    TODO: Merge a state update into current state.

    Rules:
    - For lists: append new items
    - For dicts: merge (update existing)
    - For other types: replace

    Example:
        current = {"a": [1, 2], "b": {"x": 1}, "c": "old"}
        update = {"a": [3], "b": {"y": 2}, "c": "new"}
        result = merge_state(current, update)
        # result = {"a": [1, 2, 3], "b": {"x": 1, "y": 2}, "c": "new"}
    """
    pass


# =============================================================================
# TASK 7: Create a Multi-Step Pipeline
# =============================================================================
class PipelineState(TypedDict):
    """State for multi-step pipeline."""

    text: str
    steps_completed: list
    word_count: int
    char_count: int


def count_words(state: PipelineState) -> dict:
    """
    TODO: Count words in text.
    Return word_count and append "word_count" to steps_completed.
    """
    pass


def count_chars(state: PipelineState) -> dict:
    """
    TODO: Count characters in text (excluding spaces).
    Return char_count and append "char_count" to steps_completed.
    """
    pass


def summarize(state: PipelineState) -> dict:
    """
    TODO: Create a summary string in text field.
    Format: "Text has {word_count} words and {char_count} characters"
    Append "summarize" to steps_completed.
    """
    pass


# =============================================================================
# TASK 8: Build a Configurable Graph
# =============================================================================
class ConfigurableGraph:
    """
    TODO: A graph that can be configured with different node functions.

    This allows building graphs dynamically.
    """

    def __init__(self, state_schema: type):
        """Initialize with a state schema type."""
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.start_node = None
        self.end_nodes = []

    def add_node(self, name: str, func: callable) -> "ConfigurableGraph":
        """Add a node. Returns self for chaining."""
        # TODO: Add node and return self
        pass

    def set_entry_point(self, node_name: str) -> "ConfigurableGraph":
        """Set the starting node."""
        # TODO: Set start_node
        pass

    def set_finish_point(self, node_name: str) -> "ConfigurableGraph":
        """Set an ending node."""
        # TODO: Add to end_nodes
        pass

    def add_edge(self, from_node: str, to_node: str) -> "ConfigurableGraph":
        """Add edge between nodes. Returns self for chaining."""
        # TODO: Add edge and return self
        pass

    def compile(self) -> "ConfigurableGraph":
        """Validate and compile the graph."""
        # TODO: Validate graph has start and end, return self
        pass

    def invoke(self, initial_state: dict) -> dict:
        """Execute the compiled graph."""
        # TODO: Run the graph
        pass


# =============================================================================
# TASK 9: Implement Parallel Node Execution
# =============================================================================
class ParallelState(TypedDict):
    """State for parallel execution."""

    input: str
    results: dict  # Results from parallel branches
    final: str


def branch_a(state: ParallelState) -> dict:
    """Process in branch A - uppercase."""
    # TODO: Return {"results": {"a": input.upper()}}
    pass


def branch_b(state: ParallelState) -> dict:
    """Process in branch B - lowercase."""
    # TODO: Return {"results": {"b": input.lower()}}
    pass


def branch_c(state: ParallelState) -> dict:
    """Process in branch C - title case."""
    # TODO: Return {"results": {"c": input.title()}}
    pass


def combine_results(state: ParallelState) -> dict:
    """
    TODO: Combine all branch results into final string.
    Format: "A: {a}, B: {b}, C: {c}"
    """
    pass


class ParallelGraph:
    """
    TODO: A graph that simulates parallel execution.

    Structure:
    START -> [branch_a, branch_b, branch_c] -> combine -> END

    Note: True parallel execution requires async, but we can
    simulate by running all branches and merging results.
    """

    def __init__(self):
        self.parallel_nodes = []
        self.combiner = None

    def add_parallel_branch(self, name: str, func: callable) -> "ParallelGraph":
        """Add a parallel branch."""
        # TODO: Add to parallel_nodes
        pass

    def set_combiner(self, func: callable) -> "ParallelGraph":
        """Set the combining function."""
        # TODO: Set combiner
        pass

    def invoke(self, initial_state: dict) -> dict:
        """
        Execute all parallel branches, then combine.

        Steps:
        1. Run each parallel node with initial state
        2. Merge all results into state
        3. Run combiner
        4. Return final state
        """
        pass


# =============================================================================
# TASK 10: Create a Graph Factory
# =============================================================================
class GraphFactory:
    """
    TODO: Factory for creating common graph patterns.

    Patterns:
    - sequential: A -> B -> C
    - fan_out: A -> [B, C, D] -> E
    - conditional: A -> (condition) -> B or C
    """

    @staticmethod
    def create_sequential(
        nodes: list[tuple[str, callable]], state_schema: type
    ) -> ConfigurableGraph:
        """
        Create a sequential graph from a list of (name, function) tuples.

        Example:
            nodes = [("step1", func1), ("step2", func2)]
            graph = GraphFactory.create_sequential(nodes, MyState)
        """
        # TODO: Create and configure graph
        pass

    @staticmethod
    def create_fan_out(
        entry_node: tuple[str, callable],
        parallel_nodes: list[tuple[str, callable]],
        combiner_node: tuple[str, callable],
        state_schema: type,
    ) -> dict:
        """
        Create a fan-out pattern graph.

        Returns a dict with 'graph' and 'invoke' function since
        fan-out requires special handling.
        """
        # TODO: Create fan-out structure
        pass


# =============================================================================
# MAIN - Test your implementations
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 10 - Exercise 1: Basic Graph Building")
    print("=" * 60)

    # Test Task 1: Basic State
    print("\n--- Task 1: Basic State ---")
    # state: BasicState = {"input": "test", "output": "", "steps": []}
    # print(f"State: {state}")

    # Test Task 3 & 4: Node Functions
    print("\n--- Task 3 & 4: Node Functions ---")
    # test_state = {"input": "hello world", "output": "", "steps": []}
    # result = process_input_node(test_state)
    # print(f"After process: {result}")

    # Test Task 5: Simple Graph
    print("\n--- Task 5: Simple Graph ---")
    # graph = SimpleGraph()
    # result = graph.invoke({"input": "hello", "output": "", "steps": []})
    # print(f"Graph result: {result}")

    # Test Task 6: State Merging
    print("\n--- Task 6: State Merging ---")
    # current = {"a": [1, 2], "b": {"x": 1}, "c": "old"}
    # update = {"a": [3], "b": {"y": 2}, "c": "new"}
    # merged = merge_state(current, update)
    # print(f"Merged: {merged}")

    print("\n✅ Uncomment tests as you implement each task!")

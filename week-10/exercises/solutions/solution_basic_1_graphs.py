"""
Solutions for Week 10 - Exercise 1: Basic Graph Building
========================================================
"""

from typing import TypedDict, Annotated, Any


# =============================================================================
# TASK 1: Define a Basic State
# =============================================================================
class BasicState(TypedDict):
    """State with input, output, and steps."""

    input: str
    output: str
    steps: list


# =============================================================================
# TASK 2: Create a State with Annotations
# =============================================================================
def list_reducer(existing: list, new: list) -> list:
    """Reducer that appends new items to existing list."""
    return existing + new


class AnnotatedState(TypedDict):
    """State with reducer annotation for messages."""

    messages: Annotated[list, list_reducer]
    current_node: str
    metadata: dict


# =============================================================================
# TASK 3: Implement a Node Function
# =============================================================================
def process_input_node(state: BasicState) -> dict:
    """Process input by converting to uppercase."""
    return {"output": state["input"].upper(), "steps": ["processed"]}


# =============================================================================
# TASK 4: Implement a Transformation Node
# =============================================================================
def transform_node(state: BasicState) -> dict:
    """Transform by reversing the output string."""
    return {"output": state["output"][::-1], "steps": ["transformed"]}


# =============================================================================
# TASK 5: Build a Simple Sequential Graph
# =============================================================================
class SimpleGraph:
    """Simple graph with two nodes: process and transform."""

    def __init__(self):
        self.nodes = {"process": process_input_node, "transform": transform_node}
        self.edges = [
            ("START", "process"),
            ("process", "transform"),
            ("transform", "END"),
        ]

    def add_node(self, name: str, func: callable) -> None:
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges.append((from_node, to_node))

    def compile(self) -> "SimpleGraph":
        return self

    def invoke(self, initial_state: dict) -> dict:
        state = dict(initial_state)

        # Build execution order from edges
        order = []
        current = "START"
        while current != "END":
            for from_n, to_n in self.edges:
                if from_n == current:
                    if to_n != "END":
                        order.append(to_n)
                    current = to_n
                    break

        # Execute nodes in order
        for node_name in order:
            if node_name in self.nodes:
                update = self.nodes[node_name](state)
                state = merge_state(state, update)

        return state


# =============================================================================
# TASK 6: Implement State Merging
# =============================================================================
def merge_state(current_state: dict, update: dict) -> dict:
    """Merge update into current state."""
    result = dict(current_state)

    for key, value in update.items():
        if key in result:
            existing = result[key]
            if isinstance(existing, list) and isinstance(value, list):
                result[key] = existing + value
            elif isinstance(existing, dict) and isinstance(value, dict):
                result[key] = {**existing, **value}
            else:
                result[key] = value
        else:
            result[key] = value

    return result


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
    """Count words in text."""
    words = state["text"].split()
    return {"word_count": len(words), "steps_completed": ["word_count"]}


def count_chars(state: PipelineState) -> dict:
    """Count characters excluding spaces."""
    char_count = len(state["text"].replace(" ", ""))
    return {"char_count": char_count, "steps_completed": ["char_count"]}


def summarize(state: PipelineState) -> dict:
    """Create summary string."""
    summary = (
        f"Text has {state['word_count']} words and {state['char_count']} characters"
    )
    return {"text": summary, "steps_completed": ["summarize"]}


# =============================================================================
# TASK 8: Build a Configurable Graph
# =============================================================================
class ConfigurableGraph:
    """Configurable graph that can be built dynamically."""

    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.start_node = None
        self.end_nodes = []

    def add_node(self, name: str, func: callable) -> "ConfigurableGraph":
        self.nodes[name] = func
        return self

    def set_entry_point(self, node_name: str) -> "ConfigurableGraph":
        self.start_node = node_name
        return self

    def set_finish_point(self, node_name: str) -> "ConfigurableGraph":
        self.end_nodes.append(node_name)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "ConfigurableGraph":
        self.edges.append((from_node, to_node))
        return self

    def compile(self) -> "ConfigurableGraph":
        if not self.start_node:
            raise ValueError("No entry point set")
        if not self.end_nodes:
            raise ValueError("No finish point set")
        return self

    def invoke(self, initial_state: dict) -> dict:
        state = dict(initial_state)

        # Simple execution: start -> follow edges
        current = self.start_node
        visited = set()

        while current and current not in visited:
            visited.add(current)

            if current in self.nodes:
                update = self.nodes[current](state)
                state = merge_state(state, update)

            if current in self.end_nodes:
                break

            # Find next node
            next_node = None
            for from_n, to_n in self.edges:
                if from_n == current:
                    next_node = to_n
                    break
            current = next_node

        return state


# =============================================================================
# TASK 9: Implement Parallel Node Execution
# =============================================================================
class ParallelState(TypedDict):
    """State for parallel execution."""

    input: str
    results: dict
    final: str


def branch_a(state: ParallelState) -> dict:
    """Process in branch A - uppercase."""
    return {"results": {"a": state["input"].upper()}}


def branch_b(state: ParallelState) -> dict:
    """Process in branch B - lowercase."""
    return {"results": {"b": state["input"].lower()}}


def branch_c(state: ParallelState) -> dict:
    """Process in branch C - title case."""
    return {"results": {"c": state["input"].title()}}


def combine_results(state: ParallelState) -> dict:
    """Combine all branch results."""
    r = state["results"]
    return {"final": f"A: {r.get('a', '')}, B: {r.get('b', '')}, C: {r.get('c', '')}"}


class ParallelGraph:
    """Graph that simulates parallel execution."""

    def __init__(self):
        self.parallel_nodes = []
        self.combiner = None

    def add_parallel_branch(self, name: str, func: callable) -> "ParallelGraph":
        self.parallel_nodes.append((name, func))
        return self

    def set_combiner(self, func: callable) -> "ParallelGraph":
        self.combiner = func
        return self

    def invoke(self, initial_state: dict) -> dict:
        state = dict(initial_state)

        # Run all parallel nodes and merge results
        for name, func in self.parallel_nodes:
            update = func(state)
            # Merge results dict specially
            if "results" in update and "results" in state:
                state["results"] = {**state["results"], **update["results"]}
            else:
                state = merge_state(state, update)

        # Run combiner
        if self.combiner:
            update = self.combiner(state)
            state = merge_state(state, update)

        return state


# =============================================================================
# TASK 10: Create a Graph Factory
# =============================================================================
class GraphFactory:
    """Factory for creating common graph patterns."""

    @staticmethod
    def create_sequential(
        nodes: list[tuple[str, callable]], state_schema: type
    ) -> ConfigurableGraph:
        """Create a sequential graph."""
        graph = ConfigurableGraph(state_schema)

        for i, (name, func) in enumerate(nodes):
            graph.add_node(name, func)

            if i == 0:
                graph.set_entry_point(name)
            if i == len(nodes) - 1:
                graph.set_finish_point(name)
            if i > 0:
                prev_name = nodes[i - 1][0]
                graph.add_edge(prev_name, name)

        return graph.compile()

    @staticmethod
    def create_fan_out(
        entry_node: tuple[str, callable],
        parallel_nodes: list[tuple[str, callable]],
        combiner_node: tuple[str, callable],
        state_schema: type,
    ) -> dict:
        """Create a fan-out pattern graph."""
        parallel_graph = ParallelGraph()

        for name, func in parallel_nodes:
            parallel_graph.add_parallel_branch(name, func)

        parallel_graph.set_combiner(combiner_node[1])

        def invoke(initial_state: dict) -> dict:
            state = dict(initial_state)
            # Run entry node
            update = entry_node[1](state)
            state = merge_state(state, update)
            # Run parallel + combiner
            state = parallel_graph.invoke(state)
            return state

        return {"graph": parallel_graph, "invoke": invoke}


if __name__ == "__main__":
    print("=" * 60)
    print("Week 10 - Solution 1: Basic Graph Building")
    print("=" * 60)

    # Test Simple Graph
    print("\n--- Simple Graph ---")
    graph = SimpleGraph()
    result = graph.invoke({"input": "hello", "output": "", "steps": []})
    print(f"Result: {result}")

    # Test State Merging
    print("\n--- State Merging ---")
    current = {"a": [1, 2], "b": {"x": 1}, "c": "old"}
    update = {"a": [3], "b": {"y": 2}, "c": "new"}
    merged = merge_state(current, update)
    print(f"Merged: {merged}")

    # Test Parallel Graph
    print("\n--- Parallel Graph ---")
    pg = ParallelGraph()
    pg.add_parallel_branch("a", branch_a)
    pg.add_parallel_branch("b", branch_b)
    pg.add_parallel_branch("c", branch_c)
    pg.set_combiner(combine_results)
    result = pg.invoke({"input": "Test", "results": {}, "final": ""})
    print(f"Parallel result: {result}")

    print("\n✅ All solutions implemented!")

"""
Solutions for Week 9 - Exercise Basic 1: LangChain Chains and LCEL
==================================================================
"""

from typing import Any, Callable
import json


# =============================================================================
# TASK 1: Implement a simple Runnable class
# =============================================================================
class SimpleRunnable:
    """A basic runnable that applies a function to input."""

    def __init__(self, func: Callable[[Any], Any]):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

    def __or__(self, other: "SimpleRunnable") -> "SimpleRunnable":
        def chained(x):
            return other.invoke(self.invoke(x))

        return SimpleRunnable(chained)


# =============================================================================
# TASK 2: Implement a Prompt Template
# =============================================================================
class PromptTemplate:
    """A template for creating prompts with variable substitution."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def invoke(self, input_dict: dict) -> str:
        return self.format(**input_dict)


# =============================================================================
# TASK 3: Implement Output Parsers
# =============================================================================
class StringOutputParser:
    """Parses output as a string (identity parser)."""

    def invoke(self, input_data: Any) -> str:
        return str(input_data).strip()


class ListOutputParser:
    """Parses comma-separated output into a list."""

    def __init__(self, separator: str = ","):
        self.separator = separator

    def invoke(self, input_data: str) -> list[str]:
        return [item.strip() for item in str(input_data).split(self.separator)]


class JsonOutputParser:
    """Parses JSON string output into a dictionary."""

    def invoke(self, input_data: str) -> dict:
        try:
            return json.loads(input_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


# =============================================================================
# TASK 4: Implement RunnablePassthrough
# =============================================================================
class RunnablePassthrough:
    """A runnable that passes input through unchanged."""

    def invoke(self, input_data: Any) -> Any:
        return input_data

    @staticmethod
    def assign(**kwargs: Callable) -> "RunnableAssign":
        return RunnableAssign(kwargs)


class RunnableAssign:
    """Adds computed values to the input dictionary."""

    def __init__(self, assignments: dict[str, Callable]):
        self.assignments = assignments

    def invoke(self, input_dict: dict) -> dict:
        result = dict(input_dict)
        for key, func in self.assignments.items():
            result[key] = func(input_dict)
        return result


# =============================================================================
# TASK 5: Implement RunnableLambda
# =============================================================================
class RunnableLambda:
    """Wraps any function as a runnable."""

    def __init__(self, func: Callable):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

    def __or__(self, other) -> "ChainedRunnable":
        return ChainedRunnable([self, other])


class ChainedRunnable:
    """A chain of multiple runnables."""

    def __init__(self, runnables: list):
        self.runnables = list(runnables)

    def invoke(self, input_data: Any) -> Any:
        result = input_data
        for runnable in self.runnables:
            result = runnable.invoke(result)
        return result

    def __or__(self, other) -> "ChainedRunnable":
        return ChainedRunnable(self.runnables + [other])


# =============================================================================
# TASK 6: Implement RunnableParallel
# =============================================================================
class RunnableParallel:
    """Runs multiple runnables in parallel and combines results."""

    def __init__(self, branches: dict[str, Any]):
        self.branches = branches

    def invoke(self, input_data: Any) -> dict:
        results = {}
        for name, runnable in self.branches.items():
            if hasattr(runnable, "invoke"):
                results[name] = runnable.invoke(input_data)
            else:
                results[name] = runnable
        return results


# =============================================================================
# TASK 7: Implement Chain with Fallback
# =============================================================================
class ChainWithFallback:
    """A chain that tries alternatives if the main chain fails."""

    def __init__(self, main_chain, fallbacks: list):
        self.main_chain = main_chain
        self.fallbacks = fallbacks

    def invoke(self, input_data: Any) -> Any:
        last_error = None

        # Try main chain
        try:
            return self.main_chain.invoke(input_data)
        except Exception as e:
            last_error = e

        # Try fallbacks
        for fallback in self.fallbacks:
            try:
                return fallback.invoke(input_data)
            except Exception as e:
                last_error = e

        # All failed
        raise last_error


# =============================================================================
# TASK 8: Implement Batch Processing
# =============================================================================
class BatchableRunnable:
    """A runnable that supports batch processing."""

    def __init__(self, func: Callable):
        self.func = func

    def invoke(self, input_data: Any) -> Any:
        return self.func(input_data)

    def batch(self, inputs: list[Any], max_concurrency: int = None) -> list[Any]:
        return [self.invoke(x) for x in inputs]


# =============================================================================
# TASK 9: Implement a Caching Runnable
# =============================================================================
class CachingRunnable:
    """A runnable that caches results."""

    def __init__(self, runnable, cache_size: int = 100):
        self.runnable = runnable
        self.cache_size = cache_size
        self.cache = {}

    def _make_key(self, input_data: Any) -> str:
        if isinstance(input_data, dict):
            return json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, (list, tuple)):
            return json.dumps(list(input_data))
        else:
            return str(input_data)

    def invoke(self, input_data: Any) -> Any:
        key = self._make_key(input_data)

        if key in self.cache:
            return self.cache[key]

        result = self.runnable.invoke(input_data)

        # Simple cache eviction if over size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result
        return result

    def clear_cache(self):
        self.cache.clear()


# =============================================================================
# TASK 10: Build a Complete Prompt Chain
# =============================================================================
def create_prompt_chain(
    template: str,
    transform_fn: Callable[[str], str] = None,
    output_parser: str = "string",
) -> ChainedRunnable:
    """Create a complete chain: template -> transform -> parse."""

    runnables = []

    # Add prompt template
    prompt = PromptTemplate(template)
    runnables.append(RunnableLambda(lambda x: prompt.invoke(x)))

    # Add transform if provided
    if transform_fn:
        runnables.append(RunnableLambda(transform_fn))

    # Add output parser
    if output_parser == "string":
        runnables.append(RunnableLambda(lambda x: StringOutputParser().invoke(x)))
    elif output_parser == "list":
        runnables.append(RunnableLambda(lambda x: ListOutputParser().invoke(x)))
    elif output_parser == "json":
        runnables.append(RunnableLambda(lambda x: JsonOutputParser().invoke(x)))

    return ChainedRunnable(runnables)


# =============================================================================
# Test the solutions
# =============================================================================
if __name__ == "__main__":
    # Test Task 1: SimpleRunnable
    print("=== Testing SimpleRunnable ===")
    add_one = SimpleRunnable(lambda x: x + 1)
    double = SimpleRunnable(lambda x: x * 2)
    chain = add_one | double
    result = chain.invoke(5)
    print(f"(5 + 1) * 2 = {result}")
    assert result == 12

    # Test Task 2: PromptTemplate
    print("\n=== Testing PromptTemplate ===")
    template = PromptTemplate("Hello, {name}! You are {age} years old.")
    print(template.format(name="Alice", age=30))

    # Test Task 3: Output Parsers
    print("\n=== Testing Output Parsers ===")
    str_parser = StringOutputParser()
    print(str_parser.invoke("  hello world  "))

    list_parser = ListOutputParser()
    print(list_parser.invoke("a, b, c, d"))

    json_parser = JsonOutputParser()
    print(json_parser.invoke('{"name": "test", "value": 42}'))

    # Test Task 6: RunnableParallel
    print("\n=== Testing RunnableParallel ===")
    parallel = RunnableParallel(
        {
            "upper": RunnableLambda(lambda x: x["text"].upper()),
            "lower": RunnableLambda(lambda x: x["text"].lower()),
            "original": RunnablePassthrough(),
        }
    )
    result = parallel.invoke({"text": "Hello World"})
    print(result)
    assert result["upper"] == "HELLO WORLD"

    # Test Task 7: Fallback
    print("\n=== Testing Fallback ===")
    main = RunnableLambda(lambda x: 1 / 0)  # Will fail
    backup = RunnableLambda(lambda x: "backup")
    chain = ChainWithFallback(main, [backup])
    print(chain.invoke(5))

    # Test Task 9: Caching
    print("\n=== Testing Caching ===")
    call_count = 0

    def tracked(x):
        global call_count
        call_count += 1
        return x * 2

    cached = CachingRunnable(RunnableLambda(tracked))
    print(f"First call: {cached.invoke(5)}, calls: {call_count}")
    print(f"Second call: {cached.invoke(5)}, calls: {call_count}")
    assert call_count == 1  # Should only be called once

    # Test Task 10: Complete Chain
    print("\n=== Testing Complete Chain ===")
    chain = create_prompt_chain(
        template="Process: {input}",
        transform_fn=lambda x: f"PROCESSED: {x}",
        output_parser="string",
    )
    print(chain.invoke({"input": "test data"}))

    print("\n✅ All solutions verified!")

"""
Week 9 - Exercise Basic 1: LangChain Chains and LCEL
====================================================

Learn the fundamentals of LangChain Expression Language (LCEL)
and how to build composable chains.

Topics covered:
- LCEL pipe operator syntax
- Prompt templates
- Output parsers
- RunnablePassthrough and RunnableLambda
- Chain composition
"""

from typing import Any, Callable


# =============================================================================
# TASK 1: Implement a simple Runnable class
# =============================================================================
class SimpleRunnable:
    """
    A basic runnable that applies a function to input.

    This mimics LangChain's Runnable interface.
    """

    def __init__(self, func: Callable[[Any], Any]):
        """Initialize with a function to apply.

        Args:
            func: Function that takes input and returns output
        """
        # TODO: Store the function
        pass

    def invoke(self, input_data: Any) -> Any:
        """Run the function on input data.

        Args:
            input_data: The input to process

        Returns:
            The processed output
        """
        # TODO: Apply the function to input_data
        pass

    def __or__(self, other: "SimpleRunnable") -> "SimpleRunnable":
        """Enable pipe operator for chaining.

        Example: runnable1 | runnable2

        Args:
            other: Another runnable to chain after this one

        Returns:
            A new runnable that runs both in sequence
        """
        # TODO: Return a new SimpleRunnable that chains self and other
        # The new function should: run self.invoke, then other.invoke on result
        pass


# =============================================================================
# TASK 2: Implement a Prompt Template
# =============================================================================
class PromptTemplate:
    """
    A template for creating prompts with variable substitution.

    Similar to LangChain's PromptTemplate.
    """

    def __init__(self, template: str):
        """Initialize with a template string.

        Args:
            template: Template with {variable} placeholders
        """
        # TODO: Store the template
        pass

    def format(self, **kwargs) -> str:
        """Format the template with provided values.

        Args:
            **kwargs: Variables to substitute in template

        Returns:
            Formatted string
        """
        # TODO: Use string format to substitute variables
        pass

    def invoke(self, input_dict: dict) -> str:
        """Make template callable like a runnable.

        Args:
            input_dict: Dictionary of variables

        Returns:
            Formatted string
        """
        # TODO: Call format with the dictionary values
        pass


# =============================================================================
# TASK 3: Implement Output Parsers
# =============================================================================
class StringOutputParser:
    """Parses output as a string (identity parser)."""

    def invoke(self, input_data: Any) -> str:
        """Convert input to string.

        Args:
            input_data: Any input

        Returns:
            String representation
        """
        # TODO: Return str(input_data).strip()
        pass


class ListOutputParser:
    """Parses comma-separated output into a list."""

    def __init__(self, separator: str = ","):
        """Initialize with separator.

        Args:
            separator: Character to split on
        """
        # TODO: Store the separator
        pass

    def invoke(self, input_data: str) -> list[str]:
        """Parse string into list.

        Args:
            input_data: Comma-separated string

        Returns:
            List of stripped items
        """
        # TODO: Split by separator and strip each item
        pass


class JsonOutputParser:
    """Parses JSON string output into a dictionary."""

    def invoke(self, input_data: str) -> dict:
        """Parse JSON string.

        Args:
            input_data: JSON string

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If JSON is invalid
        """
        import json

        # TODO: Parse JSON, handle errors appropriately
        pass


# =============================================================================
# TASK 4: Implement RunnablePassthrough
# =============================================================================
class RunnablePassthrough:
    """
    A runnable that passes input through unchanged.

    Useful in LCEL for forwarding values.
    """

    def invoke(self, input_data: Any) -> Any:
        """Pass input through unchanged.

        Args:
            input_data: Any input

        Returns:
            Same input unchanged
        """
        # TODO: Return input_data unchanged
        pass

    @staticmethod
    def assign(**kwargs: Callable) -> "RunnableAssign":
        """Create an assign runnable.

        Args:
            **kwargs: Key-function pairs to add to output

        Returns:
            RunnableAssign instance
        """
        # TODO: Return a RunnableAssign with the kwargs
        pass


class RunnableAssign:
    """Adds computed values to the input dictionary."""

    def __init__(self, assignments: dict[str, Callable]):
        """Initialize with assignments.

        Args:
            assignments: Dict mapping keys to functions
        """
        # TODO: Store assignments
        pass

    def invoke(self, input_dict: dict) -> dict:
        """Add computed values to input.

        Args:
            input_dict: Input dictionary

        Returns:
            Dictionary with original + new values
        """
        # TODO: Start with copy of input_dict
        # For each key, func in assignments, add key: func(input_dict)
        pass


# =============================================================================
# TASK 5: Implement RunnableLambda
# =============================================================================
class RunnableLambda:
    """
    Wraps any function as a runnable.

    Enables using arbitrary functions in LCEL chains.
    """

    def __init__(self, func: Callable):
        """Initialize with a function.

        Args:
            func: Any callable
        """
        # TODO: Store the function
        pass

    def invoke(self, input_data: Any) -> Any:
        """Run the wrapped function.

        Args:
            input_data: Input to pass to function

        Returns:
            Function output
        """
        # TODO: Call and return self.func(input_data)
        pass

    def __or__(self, other) -> "ChainedRunnable":
        """Enable piping to another runnable.

        Args:
            other: Next runnable in chain

        Returns:
            ChainedRunnable combining both
        """
        # TODO: Return a ChainedRunnable with self and other
        pass


class ChainedRunnable:
    """A chain of multiple runnables."""

    def __init__(self, runnables: list):
        """Initialize with list of runnables.

        Args:
            runnables: List of runnable objects
        """
        # TODO: Store the list
        pass

    def invoke(self, input_data: Any) -> Any:
        """Run all runnables in sequence.

        Args:
            input_data: Initial input

        Returns:
            Final output after all runnables
        """
        # TODO: Pass input through each runnable in sequence
        pass

    def __or__(self, other) -> "ChainedRunnable":
        """Add another runnable to the chain.

        Args:
            other: Runnable to add

        Returns:
            New ChainedRunnable with added runnable
        """
        # TODO: Return new ChainedRunnable with other added to list
        pass


# =============================================================================
# TASK 6: Implement RunnableParallel
# =============================================================================
class RunnableParallel:
    """
    Runs multiple runnables in parallel and combines results.

    In LangChain: {"key1": runnable1, "key2": runnable2}
    """

    def __init__(self, branches: dict[str, Any]):
        """Initialize with named branches.

        Args:
            branches: Dict mapping names to runnables
        """
        # TODO: Store the branches
        pass

    def invoke(self, input_data: Any) -> dict:
        """Run all branches and combine results.

        Args:
            input_data: Input passed to each branch

        Returns:
            Dict mapping branch names to their outputs
        """
        # TODO: For each branch, call invoke if it has one, else return value
        # Return dict of results
        pass


# =============================================================================
# TASK 7: Implement Chain with Fallback
# =============================================================================
class ChainWithFallback:
    """
    A chain that tries alternatives if the main chain fails.

    Critical for production reliability.
    """

    def __init__(self, main_chain, fallbacks: list):
        """Initialize with main chain and fallbacks.

        Args:
            main_chain: Primary runnable to try
            fallbacks: List of backup runnables
        """
        # TODO: Store main_chain and fallbacks
        pass

    def invoke(self, input_data: Any) -> Any:
        """Try main chain, then fallbacks if it fails.

        Args:
            input_data: Input to process

        Returns:
            Output from first successful chain

        Raises:
            Exception: If all chains fail
        """
        # TODO: Try main_chain.invoke
        # If it fails, try each fallback in order
        # If all fail, raise the last exception
        pass


# =============================================================================
# TASK 8: Implement Batch Processing
# =============================================================================
class BatchableRunnable:
    """
    A runnable that supports batch processing.

    Efficiently processes multiple inputs.
    """

    def __init__(self, func: Callable):
        """Initialize with processing function.

        Args:
            func: Function to apply to each input
        """
        # TODO: Store the function
        pass

    def invoke(self, input_data: Any) -> Any:
        """Process single input.

        Args:
            input_data: Single input

        Returns:
            Processed output
        """
        # TODO: Apply function to single input
        pass

    def batch(self, inputs: list[Any], max_concurrency: int = None) -> list[Any]:
        """Process multiple inputs.

        Args:
            inputs: List of inputs
            max_concurrency: Max parallel operations (unused in basic implementation)

        Returns:
            List of outputs
        """
        # TODO: Apply invoke to each input and return list of results
        pass


# =============================================================================
# TASK 9: Implement a Caching Runnable
# =============================================================================
class CachingRunnable:
    """
    A runnable that caches results.

    Useful for expensive operations like LLM calls.
    """

    def __init__(self, runnable, cache_size: int = 100):
        """Initialize with a runnable to cache.

        Args:
            runnable: The runnable to wrap
            cache_size: Maximum cache entries
        """
        # TODO: Store runnable, initialize cache dict, store cache_size
        pass

    def _make_key(self, input_data: Any) -> str:
        """Create cache key from input.

        Args:
            input_data: Input to hash

        Returns:
            String key for cache
        """
        import json

        # TODO: Convert input to string key (handle dicts, lists, etc.)
        pass

    def invoke(self, input_data: Any) -> Any:
        """Get cached result or compute and cache.

        Args:
            input_data: Input to process

        Returns:
            Cached or computed result
        """
        # TODO: Check cache, return if hit
        # Otherwise call runnable.invoke, cache result, return
        pass

    def clear_cache(self):
        """Clear all cached results."""
        # TODO: Empty the cache
        pass


# =============================================================================
# TASK 10: Build a Complete Prompt Chain
# =============================================================================
def create_prompt_chain(
    template: str,
    transform_fn: Callable[[str], str] = None,
    output_parser: str = "string",
) -> ChainedRunnable:
    """
    Create a complete chain: template -> transform -> parse.

    This simulates a real LangChain chain (without actual LLM).

    Args:
        template: Prompt template string
        transform_fn: Optional function to transform formatted prompt
                     (simulates LLM call)
        output_parser: Type of parser ("string", "list", or "json")

    Returns:
        A ChainedRunnable that processes inputs through the chain

    Example:
        chain = create_prompt_chain(
            template="Translate: {text}",
            transform_fn=lambda x: x.upper(),  # Mock LLM
            output_parser="string"
        )
        result = chain.invoke({"text": "hello"})
        # Result: "TRANSLATE: HELLO"
    """
    # TODO: Create and chain together:
    # 1. PromptTemplate
    # 2. RunnableLambda for transform (if provided)
    # 3. Appropriate output parser
    pass


# =============================================================================
# Test your implementations
# =============================================================================
if __name__ == "__main__":
    # Test Task 1: SimpleRunnable
    print("=== Testing SimpleRunnable ===")
    add_one = SimpleRunnable(lambda x: x + 1)
    double = SimpleRunnable(lambda x: x * 2)
    chain = add_one | double
    result = chain.invoke(5)
    print(f"(5 + 1) * 2 = {result}")  # Should be 12

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
    print(parallel.invoke({"text": "Hello World"}))

    # Test Task 10: Complete Chain
    print("\n=== Testing Complete Chain ===")
    chain = create_prompt_chain(
        template="Process: {input}",
        transform_fn=lambda x: f"PROCESSED: {x}",
        output_parser="string",
    )
    if chain:
        print(chain.invoke({"input": "test data"}))

    print("\n✅ Basic exercises completed!")

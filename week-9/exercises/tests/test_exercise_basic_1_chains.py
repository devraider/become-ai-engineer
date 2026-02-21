"""
Tests for Week 9 - Exercise Basic 1: LangChain Chains and LCEL
"""

import pytest
import json

from exercise_basic_1_chains import (
    SimpleRunnable,
    PromptTemplate,
    StringOutputParser,
    ListOutputParser,
    JsonOutputParser,
    RunnablePassthrough,
    RunnableAssign,
    RunnableLambda,
    ChainedRunnable,
    RunnableParallel,
    ChainWithFallback,
    BatchableRunnable,
    CachingRunnable,
    create_prompt_chain,
)


class TestSimpleRunnable:
    """Tests for SimpleRunnable class."""

    def test_invoke(self):
        """Test basic invoke."""
        runnable = SimpleRunnable(lambda x: x + 1)
        assert runnable.invoke(5) == 6

    def test_pipe_operator(self):
        """Test chaining with pipe operator."""
        add_one = SimpleRunnable(lambda x: x + 1)
        double = SimpleRunnable(lambda x: x * 2)
        chain = add_one | double
        assert chain.invoke(5) == 12  # (5 + 1) * 2

    def test_multiple_pipes(self):
        """Test multiple chained operations."""
        add = SimpleRunnable(lambda x: x + 1)
        mult = SimpleRunnable(lambda x: x * 2)
        sub = SimpleRunnable(lambda x: x - 3)
        chain = add | mult | sub
        assert chain.invoke(5) == 9  # ((5 + 1) * 2) - 3


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_format(self):
        """Test basic format."""
        template = PromptTemplate("Hello, {name}!")
        result = template.format(name="Alice")
        assert result == "Hello, Alice!"

    def test_format_multiple_variables(self):
        """Test formatting with multiple variables."""
        template = PromptTemplate("{greeting}, {name}! You are {age} years old.")
        result = template.format(greeting="Hello", name="Bob", age=30)
        assert result == "Hello, Bob! You are 30 years old."

    def test_invoke(self):
        """Test invoke method."""
        template = PromptTemplate("Welcome {user}")
        result = template.invoke({"user": "Charlie"})
        assert result == "Welcome Charlie"


class TestOutputParsers:
    """Tests for output parser classes."""

    def test_string_parser(self):
        """Test StringOutputParser."""
        parser = StringOutputParser()
        assert parser.invoke("  hello  ") == "hello"
        assert parser.invoke(42) == "42"

    def test_list_parser(self):
        """Test ListOutputParser."""
        parser = ListOutputParser()
        result = parser.invoke("a, b, c")
        assert result == ["a", "b", "c"]

    def test_list_parser_custom_separator(self):
        """Test ListOutputParser with custom separator."""
        parser = ListOutputParser(separator="|")
        result = parser.invoke("a|b|c")
        assert result == ["a", "b", "c"]

    def test_json_parser(self):
        """Test JsonOutputParser."""
        parser = JsonOutputParser()
        result = parser.invoke('{"name": "test", "value": 42}')
        assert result == {"name": "test", "value": 42}

    def test_json_parser_invalid(self):
        """Test JsonOutputParser with invalid JSON."""
        parser = JsonOutputParser()
        with pytest.raises(ValueError):
            parser.invoke("not json")


class TestRunnablePassthrough:
    """Tests for RunnablePassthrough class."""

    def test_passthrough(self):
        """Test basic passthrough."""
        passthrough = RunnablePassthrough()
        data = {"key": "value"}
        assert passthrough.invoke(data) == data

    def test_assign(self):
        """Test RunnablePassthrough.assign."""
        assign = RunnablePassthrough.assign(upper=lambda x: x["text"].upper())
        result = assign.invoke({"text": "hello"})
        assert result["text"] == "hello"
        assert result["upper"] == "HELLO"


class TestRunnableLambda:
    """Tests for RunnableLambda class."""

    def test_invoke(self):
        """Test basic invoke."""
        runnable = RunnableLambda(lambda x: x.upper())
        assert runnable.invoke("hello") == "HELLO"

    def test_pipe(self):
        """Test piping RunnableLambda."""
        upper = RunnableLambda(lambda x: x.upper())
        add_ex = RunnableLambda(lambda x: x + "!")
        chain = upper | add_ex
        assert chain.invoke("hello") == "HELLO!"


class TestChainedRunnable:
    """Tests for ChainedRunnable class."""

    def test_invoke_sequence(self):
        """Test invoking a sequence."""
        runnables = [
            RunnableLambda(lambda x: x + 1),
            RunnableLambda(lambda x: x * 2),
        ]
        chain = ChainedRunnable(runnables)
        assert chain.invoke(5) == 12

    def test_extend_chain(self):
        """Test extending a chain."""
        chain = ChainedRunnable([RunnableLambda(lambda x: x + 1)])
        extended = chain | RunnableLambda(lambda x: x * 2)
        assert extended.invoke(5) == 12


class TestRunnableParallel:
    """Tests for RunnableParallel class."""

    def test_parallel_execution(self):
        """Test parallel branch execution."""
        parallel = RunnableParallel(
            {
                "upper": RunnableLambda(lambda x: x["text"].upper()),
                "lower": RunnableLambda(lambda x: x["text"].lower()),
            }
        )
        result = parallel.invoke({"text": "Hello"})
        assert result["upper"] == "HELLO"
        assert result["lower"] == "hello"

    def test_parallel_with_passthrough(self):
        """Test parallel with passthrough."""
        parallel = RunnableParallel(
            {
                "processed": RunnableLambda(lambda x: x["value"] * 2),
                "original": RunnablePassthrough(),
            }
        )
        result = parallel.invoke({"value": 5})
        assert result["processed"] == 10
        assert result["original"] == {"value": 5}


class TestChainWithFallback:
    """Tests for ChainWithFallback class."""

    def test_main_succeeds(self):
        """Test when main chain succeeds."""
        main = RunnableLambda(lambda x: x + 1)
        backup = RunnableLambda(lambda x: x + 100)
        chain = ChainWithFallback(main, [backup])
        assert chain.invoke(5) == 6

    def test_fallback_used(self):
        """Test when main fails and fallback is used."""

        def fail(x):
            raise ValueError("Main failed")

        main = RunnableLambda(fail)
        backup = RunnableLambda(lambda x: "backup result")
        chain = ChainWithFallback(main, [backup])
        assert chain.invoke(5) == "backup result"

    def test_all_fail(self):
        """Test when all chains fail."""

        def fail(x):
            raise ValueError("Failed")

        main = RunnableLambda(fail)
        fallbacks = [RunnableLambda(fail), RunnableLambda(fail)]
        chain = ChainWithFallback(main, fallbacks)

        with pytest.raises(Exception):
            chain.invoke(5)


class TestBatchableRunnable:
    """Tests for BatchableRunnable class."""

    def test_single_invoke(self):
        """Test single invocation."""
        runnable = BatchableRunnable(lambda x: x * 2)
        assert runnable.invoke(5) == 10

    def test_batch(self):
        """Test batch processing."""
        runnable = BatchableRunnable(lambda x: x * 2)
        results = runnable.batch([1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]


class TestCachingRunnable:
    """Tests for CachingRunnable class."""

    def test_caching(self):
        """Test that results are cached."""
        call_count = 0

        def tracked_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        inner = RunnableLambda(tracked_func)
        cached = CachingRunnable(inner)

        # First call
        result1 = cached.invoke(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same input - should use cache
        result2 = cached.invoke(5)
        assert result2 == 10
        assert call_count == 1  # No additional call

    def test_clear_cache(self):
        """Test cache clearing."""
        call_count = 0

        def tracked_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        inner = RunnableLambda(tracked_func)
        cached = CachingRunnable(inner)

        cached.invoke(5)
        assert call_count == 1

        cached.clear_cache()
        cached.invoke(5)
        assert call_count == 2  # Called again after cache clear


class TestCreatePromptChain:
    """Tests for create_prompt_chain function."""

    def test_basic_chain(self):
        """Test basic prompt chain."""
        chain = create_prompt_chain(
            template="Process: {input}",
            transform_fn=lambda x: x.upper(),
            output_parser="string",
        )
        result = chain.invoke({"input": "hello"})
        assert "PROCESS" in result
        assert "HELLO" in result

    def test_list_output(self):
        """Test chain with list output."""
        chain = create_prompt_chain(
            template="{items}", transform_fn=lambda x: x, output_parser="list"
        )
        result = chain.invoke({"items": "a, b, c"})
        assert isinstance(result, list)
        assert len(result) == 3

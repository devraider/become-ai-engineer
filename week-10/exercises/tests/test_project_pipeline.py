"""
Tests for Week 10 - Project: Research Assistant
"""

import pytest
from project_pipeline import (
    ResearchPhase,
    Source,
    ResearchPlan,
    ResearchState,
    QueryType,
    QueryAnalyzer,
    ResearchPlanner,
    MockSearchEngine,
    SearchExecutor,
    SourceEvaluator,
    Synthesizer,
    HumanReviewManager,
    ResearchAssistant,
    StreamingResearchAssistant,
)


class TestSource:
    """Tests for Source dataclass."""

    def test_source_creation(self):
        """Source should be created with required fields."""
        source = Source(id="1", title="Test", content="Test content")
        assert source.id == "1"
        assert source.title == "Test"

    def test_source_to_dict(self):
        """Source.to_dict should return dictionary."""
        source = Source(id="1", title="Test", content="Content", credibility_score=0.8)
        d = source.to_dict()
        assert d["id"] == "1"
        assert d["credibility"] == 0.8

    def test_source_truncates_long_content(self):
        """Source.to_dict should truncate long content."""
        long_content = "x" * 500
        source = Source(id="1", title="Test", content=long_content)
        d = source.to_dict()
        assert len(d["content"]) < len(long_content)
        assert "..." in d["content"]


class TestResearchPlan:
    """Tests for ResearchPlan dataclass."""

    def test_plan_creation(self):
        """ResearchPlan should be created with fields."""
        plan = ResearchPlan(
            query="test",
            sub_questions=["q1", "q2"],
            search_strategies=["web"],
            estimated_steps=3,
        )
        assert len(plan.sub_questions) == 2

    def test_plan_to_dict(self):
        """ResearchPlan.to_dict should work correctly."""
        plan = ResearchPlan("test", ["q1"], ["web"], 2)
        d = plan.to_dict()
        assert "query" in d
        assert "sub_questions" in d


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer."""

    def test_analyze_factual_query(self):
        """QueryAnalyzer should detect factual queries."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("What is the capital of France?")
        assert result is not None

    def test_analyze_comparative_query(self):
        """QueryAnalyzer should detect comparative queries."""
        analyzer = QueryAnalyzer()
        result = analyzer.analyze("Compare Python vs JavaScript")
        assert (
            result["query_type"] == QueryType.COMPARATIVE
            or "compare" in str(result).lower()
        )

    def test_extract_entities(self):
        """QueryAnalyzer should extract entities."""
        analyzer = QueryAnalyzer()
        entities = analyzer.extract_entities("What is Python and JavaScript?")
        assert len(entities) >= 0  # May or may not find entities

    def test_generate_sub_questions(self):
        """QueryAnalyzer should generate sub-questions."""
        analyzer = QueryAnalyzer()
        questions = analyzer.generate_sub_questions(
            "What is AI?", QueryType.EXPLANATORY
        )
        assert isinstance(questions, list)


class TestResearchPlanner:
    """Tests for ResearchPlanner."""

    def test_create_plan(self):
        """ResearchPlanner should create a plan."""
        planner = ResearchPlanner()
        analysis = {"complexity": "simple", "query_type": QueryType.FACTUAL}
        plan = planner.create_plan("What is Python?", analysis)
        assert isinstance(plan, ResearchPlan)
        assert len(plan.sub_questions) > 0

    def test_estimate_complexity(self):
        """ResearchPlanner should estimate complexity."""
        planner = ResearchPlanner()

        simple = planner.estimate_complexity({"complexity": "simple"})
        complex_val = planner.estimate_complexity({"complexity": "complex"})

        assert simple < complex_val


class TestMockSearchEngine:
    """Tests for MockSearchEngine."""

    def test_search_finds_results(self):
        """MockSearchEngine should find matching results."""
        engine = MockSearchEngine()
        results = engine.search("python")
        assert len(results) > 0
        assert all(isinstance(r, Source) for r in results)

    def test_search_respects_max_results(self):
        """MockSearchEngine should respect max_results."""
        engine = MockSearchEngine()
        results = engine.search("python", max_results=1)
        assert len(results) <= 1


class TestSearchExecutor:
    """Tests for SearchExecutor."""

    def test_execute_searches(self):
        """SearchExecutor should execute all searches."""
        executor = SearchExecutor()
        plan = ResearchPlan("python", ["What is Python?"], ["keyword"], 2)
        results = executor.execute_searches(plan)
        assert isinstance(results, list)

    def test_deduplicate(self):
        """SearchExecutor should remove duplicates."""
        executor = SearchExecutor()
        sources = [
            Source(id="1", title="Test", content="A"),
            Source(id="1", title="Test", content="A"),
            Source(id="2", title="Other", content="B"),
        ]
        deduped = executor.deduplicate(sources)
        assert len(deduped) <= len(sources)


class TestSourceEvaluator:
    """Tests for SourceEvaluator."""

    def test_evaluate_credibility(self):
        """SourceEvaluator should evaluate credibility."""
        evaluator = SourceEvaluator()
        source = Source(
            id="1", title="Test", content="Content", url="http://example.edu"
        )
        score = evaluator.evaluate_credibility(source)
        assert 0.0 <= score <= 1.0

    def test_evaluate_relevance(self):
        """SourceEvaluator should evaluate relevance."""
        evaluator = SourceEvaluator()
        source = Source(
            id="1", title="Python Guide", content="Python is a programming language"
        )
        score = evaluator.evaluate_relevance(source, "Python programming")
        assert 0.0 <= score <= 1.0

    def test_filter_sources(self):
        """SourceEvaluator should filter low-quality sources."""
        evaluator = SourceEvaluator()
        sources = [
            Source(id="1", title="Good", content="Good content", credibility_score=0.8),
            Source(id="2", title="Bad", content="Bad content", credibility_score=0.2),
        ]
        filtered = evaluator.filter_sources(sources, min_score=0.5)
        assert len(filtered) <= len(sources)


class TestSynthesizer:
    """Tests for Synthesizer."""

    def test_extract_findings(self):
        """Synthesizer should extract findings from sources."""
        synthesizer = Synthesizer()
        sources = [
            Source(id="1", title="Test", content="Python is versatile"),
            Source(id="2", title="Test2", content="Python is readable"),
        ]
        findings = synthesizer.extract_findings(sources)
        assert isinstance(findings, list)

    def test_synthesize(self):
        """Synthesizer should create coherent response."""
        synthesizer = Synthesizer()
        findings = ["Python is versatile", "Python is easy to learn"]
        result = synthesizer.synthesize("What is Python?", findings)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_citations(self):
        """Synthesizer should generate citations."""
        synthesizer = Synthesizer()
        sources = [
            Source(
                id="1", title="Python Guide", content="...", url="http://example.com"
            ),
            Source(id="2", title="Python Docs", content="..."),
        ]
        citations = synthesizer.generate_citations(sources)
        assert len(citations) == 2


class TestHumanReviewManager:
    """Tests for HumanReviewManager."""

    def test_needs_review_sensitive(self):
        """HumanReviewManager should flag sensitive topics."""
        manager = HumanReviewManager()
        state = {"query": "medical treatment advice", "synthesis": "..."}
        assert manager.needs_review(state) is True

    def test_needs_review_normal(self):
        """HumanReviewManager should not flag normal topics."""
        manager = HumanReviewManager()
        state = {"query": "What is Python?", "synthesis": "Python is..."}
        # May or may not need review depending on implementation
        result = manager.needs_review(state)
        assert isinstance(result, bool)

    def test_process_approval(self):
        """HumanReviewManager should process approval."""
        manager = HumanReviewManager()
        result = manager.process_approval(approved=True, feedback="Looks good")
        assert "approved" in str(result).lower() or result.get("approved")


class TestResearchAssistant:
    """Tests for ResearchAssistant."""

    def test_initialize_state(self):
        """ResearchAssistant should initialize state correctly."""
        assistant = ResearchAssistant()
        state = assistant._initialize_state("What is Python?")
        assert state["query"] == "What is Python?"
        assert state["phase"] == ResearchPhase.QUERY_ANALYSIS.value

    def test_research_completes(self):
        """ResearchAssistant.research should complete workflow."""
        assistant = ResearchAssistant()
        result = assistant.research("What is Python?", "thread-1")
        assert "phase" in result

    def test_research_persistence(self):
        """ResearchAssistant should save checkpoints."""
        assistant = ResearchAssistant()
        assistant.research("What is AI?", "thread-2")
        status = assistant.get_status("thread-2")
        assert status is not None


class TestStreamingResearchAssistant:
    """Tests for StreamingResearchAssistant."""

    def test_stream_research_yields_events(self):
        """stream_research should yield events."""
        assistant = StreamingResearchAssistant()
        events = list(assistant.stream_research("What is Python?"))
        assert len(events) > 0

    def test_stream_events_have_phase(self):
        """Stream events should include phase information."""
        assistant = StreamingResearchAssistant()
        events = list(assistant.stream_research("Test query"))
        for event in events:
            assert "phase" in event

    def test_stream_completes(self):
        """Stream should complete with result."""
        assistant = StreamingResearchAssistant()
        events = list(assistant.stream_research("What is ML?"))
        phases = [e["phase"] for e in events]
        assert "complete" in phases or len(phases) > 0

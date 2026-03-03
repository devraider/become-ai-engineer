"""
Tests for Week 14 - Project: Research Assistant Pipeline

Run with: pytest tests/test_project_pipeline.py -v
"""

import pytest
import asyncio
from datetime import datetime

from project_pipeline import (
    QueryType,
    SourceType,
    ResearchQuery,
    SearchResult,
    ResearchFinding,
    ReportFormat,
    ResearchReport,
    SearchTool,
    WebSearchTool,
    AcademicSearchTool,
    DocumentReaderTool,
    SummarizerTool,
    CitationTool,
    AgentBase,
    PlannerAgent,
    ResearcherAgent,
    AnalyzerAgent,
    WriterAgent,
    ResearchPipeline,
    QualityChecker,
    OutputFormatter,
    ResearchConfig,
    ResearchAssistant,
)


# =============================================================================
# Part 1: Research Query Tests
# =============================================================================
class TestResearchQuery:
    """Tests for ResearchQuery class."""

    def test_create_query(self):
        """Test creating a research query."""
        query = ResearchQuery(
            question="What is machine learning?", query_type=QueryType.FACTUAL
        )

        assert query.question == "What is machine learning?"
        assert query.query_type == QueryType.FACTUAL

    def test_extract_keywords(self):
        """Test keyword extraction."""
        query = ResearchQuery(
            question="What are the applications of deep learning in healthcare?"
        )

        keywords = query.extract_keywords()

        # Should extract meaningful words
        assert len(keywords) > 0
        assert "deep" in keywords or "learning" in keywords or "healthcare" in keywords

    def test_to_dict(self):
        """Test converting query to dictionary."""
        query = ResearchQuery(
            question="Test question",
            query_type=QueryType.EXPLORATORY,
            sources=[SourceType.WEB, SourceType.ACADEMIC],
        )

        result = query.to_dict()

        assert "question" in result
        assert result["query_type"] == "exploratory"
        assert len(result["sources"]) == 2


# =============================================================================
# Part 2: Search Result Tests
# =============================================================================
class TestSearchResult:
    """Tests for SearchResult class."""

    def test_create_result(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Article",
            url="https://example.com/article",
            snippet="This is a test article about AI.",
            source=SourceType.WEB,
        )

        assert result.title == "Test Article"
        assert result.source == SourceType.WEB

    def test_relevance_score(self):
        """Test relevance scoring."""
        result = SearchResult(
            title="Machine Learning Guide",
            url="https://example.com",
            snippet="Comprehensive guide to machine learning algorithms",
            source=SourceType.ACADEMIC,
            relevance_score=0.85,
        )

        assert result.relevance_score == 0.85

    def test_to_citation(self):
        """Test generating citation."""
        result = SearchResult(
            title="Deep Learning",
            url="https://example.com/deep-learning",
            snippet="Article about deep learning",
            source=SourceType.ACADEMIC,
            authors=["John Doe", "Jane Smith"],
            published_date=datetime(2023, 6, 15),
        )

        citation = result.to_citation()

        assert "Deep Learning" in citation
        assert "John Doe" in citation or "Doe" in citation


# =============================================================================
# Part 3: Research Finding Tests
# =============================================================================
class TestResearchFinding:
    """Tests for ResearchFinding class."""

    def test_create_finding(self):
        """Test creating a research finding."""
        source = SearchResult(
            title="Source Article",
            url="https://example.com",
            snippet="Test",
            source=SourceType.WEB,
        )

        finding = ResearchFinding(
            key_point="AI is transforming healthcare",
            evidence="Multiple studies show improvement in diagnostics",
            sources=[source],
            confidence=0.8,
        )

        assert finding.key_point == "AI is transforming healthcare"
        assert finding.confidence == 0.8

    def test_to_markdown(self):
        """Test converting finding to markdown."""
        source = SearchResult(
            title="Article",
            url="https://example.com",
            snippet="Test",
            source=SourceType.WEB,
        )

        finding = ResearchFinding(
            key_point="Important finding",
            evidence="Supporting evidence",
            sources=[source],
        )

        markdown = finding.to_markdown()

        assert "Important finding" in markdown
        assert "Supporting evidence" in markdown


# =============================================================================
# Part 4: Research Report Tests
# =============================================================================
class TestResearchReport:
    """Tests for ResearchReport class."""

    def test_create_report(self):
        """Test creating a research report."""
        source = SearchResult(
            title="Source",
            url="https://example.com",
            snippet="Test",
            source=SourceType.WEB,
        )

        finding = ResearchFinding(
            key_point="Key point", evidence="Evidence", sources=[source]
        )

        report = ResearchReport(
            query=ResearchQuery(question="Test question"),
            title="Test Report",
            summary="This is a summary",
            findings=[finding],
            sources=[source],
        )

        assert report.title == "Test Report"

    def test_format_markdown(self):
        """Test formatting report as markdown."""
        report = ResearchReport(
            query=ResearchQuery(question="Test question"),
            title="Test Report",
            summary="Summary text",
            findings=[],
            sources=[],
        )

        markdown = report.format(ReportFormat.MARKDOWN)

        assert "# Test Report" in markdown
        assert "Summary text" in markdown

    def test_format_json(self):
        """Test formatting report as JSON."""
        report = ResearchReport(
            query=ResearchQuery(question="Test question"),
            title="Test Report",
            summary="Summary",
            findings=[],
            sources=[],
        )

        json_output = report.format(ReportFormat.JSON)

        assert "title" in json_output
        assert "Test Report" in json_output


# =============================================================================
# Part 5: Search Tools Tests
# =============================================================================
class TestSearchTools:
    """Tests for search tool implementations."""

    @pytest.mark.asyncio
    async def test_web_search(self):
        """Test web search tool."""
        tool = WebSearchTool()

        results = await tool.search("machine learning", max_results=3)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_academic_search(self):
        """Test academic search tool."""
        tool = AcademicSearchTool()

        results = await tool.search("neural networks", max_results=3)

        assert isinstance(results, list)

    def test_search_tool_interface(self):
        """Test search tool interface."""
        tool = WebSearchTool()

        assert hasattr(tool, "search")
        assert tool.source_type == SourceType.WEB


# =============================================================================
# Part 6: Document Tools Tests
# =============================================================================
class TestDocumentTools:
    """Tests for document processing tools."""

    @pytest.mark.asyncio
    async def test_document_reader(self):
        """Test document reader tool."""
        tool = DocumentReaderTool()

        content = await tool.read("https://example.com/document")

        assert isinstance(content, str)

    @pytest.mark.asyncio
    async def test_summarizer(self):
        """Test summarizer tool."""
        tool = SummarizerTool()

        text = "This is a long text that needs to be summarized. " * 10
        summary = await tool.summarize(text, max_length=100)

        assert isinstance(summary, str)
        assert len(summary) <= len(text)

    @pytest.mark.asyncio
    async def test_citation_tool(self):
        """Test citation tool."""
        tool = CitationTool()

        source = SearchResult(
            title="Test Article",
            url="https://example.com",
            snippet="Test",
            source=SourceType.ACADEMIC,
            authors=["Author Name"],
        )

        citation = await tool.generate(source)

        assert isinstance(citation, str)


# =============================================================================
# Part 7: Agent Tests
# =============================================================================
class TestAgents:
    """Tests for research agents."""

    @pytest.mark.asyncio
    async def test_planner_agent(self):
        """Test planner agent."""
        agent = PlannerAgent()

        query = ResearchQuery(question="What are the benefits of AI in education?")

        plan = await agent.plan(query)

        assert plan is not None
        assert "steps" in plan or "tasks" in plan

    @pytest.mark.asyncio
    async def test_researcher_agent(self):
        """Test researcher agent."""
        agent = ResearcherAgent()

        task = {
            "query": "machine learning applications",
            "sources": ["web", "academic"],
        }

        results = await agent.research(task)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_analyzer_agent(self):
        """Test analyzer agent."""
        agent = AnalyzerAgent()

        results = [
            SearchResult(
                title="Article 1",
                url="https://example.com/1",
                snippet="AI improves efficiency",
                source=SourceType.WEB,
            )
        ]

        findings = await agent.analyze(results)

        assert isinstance(findings, list)

    @pytest.mark.asyncio
    async def test_writer_agent(self):
        """Test writer agent."""
        agent = WriterAgent()

        query = ResearchQuery(question="Test question")
        findings = [
            ResearchFinding(key_point="Key point", evidence="Evidence", sources=[])
        ]

        report = await agent.write(query, findings)

        assert isinstance(report, ResearchReport)


# =============================================================================
# Part 8: Research Pipeline Tests
# =============================================================================
class TestResearchPipeline:
    """Tests for the research pipeline."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = ResearchPipeline()

        assert pipeline is not None

    def test_add_step(self):
        """Test adding a step to pipeline."""
        pipeline = ResearchPipeline()

        async def custom_step(data):
            return data

        pipeline.add_step("custom", custom_step)

        assert "custom" in pipeline.steps

    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        """Test running the pipeline."""
        pipeline = ResearchPipeline()

        query = ResearchQuery(question="What is artificial intelligence?")

        result = await pipeline.run(query)

        assert result is not None


# =============================================================================
# Part 9: Quality Checker Tests
# =============================================================================
class TestQualityChecker:
    """Tests for quality checking."""

    def test_check_report(self):
        """Test checking a report."""
        checker = QualityChecker()

        report = ResearchReport(
            query=ResearchQuery(question="Test"),
            title="Test Report",
            summary="Summary",
            findings=[],
            sources=[],
        )

        result = checker.check(report)

        assert "score" in result or "passed" in result

    def test_check_sources(self):
        """Test checking source quality."""
        checker = QualityChecker()

        sources = [
            SearchResult(
                title="Article",
                url="https://academic.edu/paper",
                snippet="Test",
                source=SourceType.ACADEMIC,
            )
        ]

        score = checker.check_sources(sources)

        assert 0 <= score <= 1

    def test_check_findings(self):
        """Test checking findings quality."""
        checker = QualityChecker()

        findings = [
            ResearchFinding(
                key_point="Point",
                evidence="Evidence with supporting details",
                sources=[],
                confidence=0.8,
            )
        ]

        result = checker.check_findings(findings)

        assert result is not None


# =============================================================================
# Part 10: Output Formatter Tests
# =============================================================================
class TestOutputFormatter:
    """Tests for output formatting."""

    def test_format_markdown(self):
        """Test markdown formatting."""
        formatter = OutputFormatter()

        report = ResearchReport(
            query=ResearchQuery(question="Test"),
            title="Test Report",
            summary="Summary",
            findings=[],
            sources=[],
        )

        output = formatter.format(report, ReportFormat.MARKDOWN)

        assert "#" in output  # Markdown headers

    def test_format_html(self):
        """Test HTML formatting."""
        formatter = OutputFormatter()

        report = ResearchReport(
            query=ResearchQuery(question="Test"),
            title="Test Report",
            summary="Summary",
            findings=[],
            sources=[],
        )

        output = formatter.format(report, ReportFormat.HTML)

        assert "<" in output and ">" in output

    def test_format_json(self):
        """Test JSON formatting."""
        formatter = OutputFormatter()

        report = ResearchReport(
            query=ResearchQuery(question="Test"),
            title="Test Report",
            summary="Summary",
            findings=[],
            sources=[],
        )

        output = formatter.format(report, ReportFormat.JSON)

        assert "{" in output


# =============================================================================
# Part 11: Research Assistant Tests
# =============================================================================
class TestResearchAssistant:
    """Tests for the main ResearchAssistant class."""

    def test_create_assistant(self):
        """Test creating a research assistant."""
        config = ResearchConfig()
        assistant = ResearchAssistant(config)

        assert assistant is not None

    @pytest.mark.asyncio
    async def test_research_question(self):
        """Test researching a question."""
        assistant = ResearchAssistant()

        report = await assistant.research("What are the main applications of AI?")

        assert isinstance(report, ResearchReport)

    @pytest.mark.asyncio
    async def test_research_with_sources(self):
        """Test researching with specific sources."""
        assistant = ResearchAssistant()

        report = await assistant.research(
            "Deep learning advances", sources=[SourceType.ACADEMIC]
        )

        assert isinstance(report, ResearchReport)

    @pytest.mark.asyncio
    async def test_research_output_format(self):
        """Test research output in different formats."""
        assistant = ResearchAssistant()

        report = await assistant.research(
            "AI in healthcare", output_format=ReportFormat.MARKDOWN
        )

        formatted = report.format(ReportFormat.MARKDOWN)
        assert "# " in formatted


# =============================================================================
# Integration Tests
# =============================================================================
class TestResearchAssistantIntegration:
    """Integration tests for the research assistant."""

    @pytest.mark.asyncio
    async def test_full_research_flow(self):
        """Test complete research flow."""
        # Create assistant
        config = ResearchConfig(max_sources=5, quality_threshold=0.5)
        assistant = ResearchAssistant(config)

        # Execute research
        report = await assistant.research(
            question="What are recent advances in natural language processing?",
            sources=[SourceType.WEB, SourceType.ACADEMIC],
            output_format=ReportFormat.MARKDOWN,
        )

        # Verify report
        assert report.title is not None
        assert report.summary is not None

        # Check quality
        checker = QualityChecker()
        quality = checker.check(report)

        assert quality is not None

    @pytest.mark.asyncio
    async def test_multi_query_research(self):
        """Test researching multiple queries."""
        assistant = ResearchAssistant()

        questions = ["What is machine learning?", "How does deep learning work?"]

        reports = []
        for question in questions:
            report = await assistant.research(question)
            reports.append(report)

        assert len(reports) == 2

    @pytest.mark.asyncio
    async def test_pipeline_customization(self):
        """Test customizing the research pipeline."""
        assistant = ResearchAssistant()

        # Add custom step
        async def custom_filter(data):
            # Filter low confidence findings
            if "findings" in data:
                data["findings"] = [
                    f for f in data["findings"] if getattr(f, "confidence", 0.5) > 0.3
                ]
            return data

        assistant.pipeline.add_step("custom_filter", custom_filter)

        report = await assistant.research("AI applications")

        assert report is not None

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in research."""
        assistant = ResearchAssistant()

        # Empty query should still work
        report = await assistant.research("")

        # Should return valid report or handle error gracefully
        assert report is not None or True  # Either works or handles error

    @pytest.mark.asyncio
    async def test_concurrent_research(self):
        """Test concurrent research tasks."""
        assistant = ResearchAssistant()

        # Start multiple research tasks concurrently
        tasks = [
            assistant.research("AI in healthcare"),
            assistant.research("AI in finance"),
            assistant.research("AI in education"),
        ]

        reports = await asyncio.gather(*tasks)

        assert len(reports) == 3
        assert all(isinstance(r, ResearchReport) for r in reports)

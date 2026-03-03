"""
Solution for Week 14 - Project: Research Assistant Pipeline

Complete implementation of an AI-powered research assistant.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Callable, AsyncIterator
from enum import Enum
from datetime import datetime
import json
import asyncio
import uuid
from abc import ABC, abstractmethod
import re


# =============================================================================
# Part 1: Research Query - SOLUTION
# =============================================================================
class QueryType(Enum):
    """Types of research queries."""

    FACTUAL = "factual"
    EXPLORATORY = "exploratory"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"


class SourceType(Enum):
    """Types of sources."""

    WEB = "web"
    ACADEMIC = "academic"
    NEWS = "news"
    BOOK = "book"
    DATABASE = "database"


@dataclass
class ResearchQuery:
    """
    Represents a research query.

    Solution implements:
    - Query parsing
    - Keyword extraction
    - Query classification
    """

    question: str
    query_type: QueryType = QueryType.EXPLORATORY
    sources: List[SourceType] = field(default_factory=lambda: [SourceType.WEB])
    max_results: int = 10
    date_range: Optional[tuple[datetime, datetime]] = None
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Stop words for keyword extraction
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "same",
        "than",
        "too",
        "very",
        "just",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "how",
        "why",
        "when",
        "where",
    }

    def extract_keywords(self) -> List[str]:
        """Extract keywords from the question."""
        # Tokenize
        words = re.findall(r"\b[a-zA-Z]+\b", self.question.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "query_type": self.query_type.value,
            "sources": [s.value for s in self.sources],
            "max_results": self.max_results,
            "language": self.language,
            "keywords": self.extract_keywords(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchQuery":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            question=data["question"],
            query_type=QueryType(data.get("query_type", "exploratory")),
            sources=[SourceType(s) for s in data.get("sources", ["web"])],
            max_results=data.get("max_results", 10),
            language=data.get("language", "en"),
        )


# =============================================================================
# Part 2: Search Result - SOLUTION
# =============================================================================
@dataclass
class SearchResult:
    """
    A single search result.

    Solution implements:
    - Source metadata
    - Relevance scoring
    - Citation generation
    """

    title: str
    url: str
    snippet: str
    source: SourceType
    relevance_score: float = 0.0
    authors: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_at: datetime = field(default_factory=datetime.now)

    def to_citation(self, style: str = "apa") -> str:
        """Generate a citation."""
        if style == "apa":
            # APA style
            authors_str = ", ".join(self.authors) if self.authors else "Unknown"
            year = self.published_date.year if self.published_date else "n.d."
            return f"{authors_str} ({year}). {self.title}. Retrieved from {self.url}"
        elif style == "mla":
            # MLA style
            authors_str = ", ".join(self.authors) if self.authors else "Unknown"
            return f'{authors_str}. "{self.title}." Web. {self.retrieved_at.strftime("%d %b %Y")}.'
        else:
            # Simple
            return f"{self.title} - {self.url}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source.value,
            "relevance_score": self.relevance_score,
            "authors": self.authors,
            "published_date": (
                self.published_date.isoformat() if self.published_date else None
            ),
            "retrieved_at": self.retrieved_at.isoformat(),
        }


# =============================================================================
# Part 3: Research Finding - SOLUTION
# =============================================================================
@dataclass
class ResearchFinding:
    """
    A research finding with supporting evidence.

    Solution implements:
    - Evidence tracking
    - Source attribution
    - Confidence scoring
    """

    key_point: str
    evidence: str
    sources: List[SearchResult]
    confidence: float = 0.5
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_markdown(self) -> str:
        """Convert finding to markdown."""
        md = f"### {self.key_point}\n\n"
        md += f"{self.evidence}\n\n"

        if self.sources:
            md += "**Sources:**\n"
            for source in self.sources:
                md += f"- [{source.title}]({source.url})\n"

        md += f"\n*Confidence: {self.confidence:.0%}*\n"

        return md

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "key_point": self.key_point,
            "evidence": self.evidence,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "category": self.category,
            "tags": self.tags,
        }


# =============================================================================
# Part 4: Research Report - SOLUTION
# =============================================================================
class ReportFormat(Enum):
    """Report output formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


@dataclass
class ResearchReport:
    """
    Complete research report.

    Solution implements:
    - Multiple output formats
    - Source aggregation
    - Quality metrics
    """

    query: ResearchQuery
    title: str
    summary: str
    findings: List[ResearchFinding]
    sources: List[SearchResult]
    methodology: str = ""
    limitations: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, output_format: ReportFormat) -> str:
        """Format report for output."""
        if output_format == ReportFormat.MARKDOWN:
            return self._to_markdown()
        elif output_format == ReportFormat.HTML:
            return self._to_html()
        elif output_format == ReportFormat.JSON:
            return self._to_json()
        else:
            return self._to_text()

    def _to_markdown(self) -> str:
        """Convert to markdown."""
        md = f"# {self.title}\n\n"
        md += f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}*\n\n"

        md += "## Summary\n\n"
        md += f"{self.summary}\n\n"

        if self.findings:
            md += "## Key Findings\n\n"
            for i, finding in enumerate(self.findings, 1):
                md += f"### {i}. {finding.key_point}\n\n"
                md += f"{finding.evidence}\n\n"
                md += f"*Confidence: {finding.confidence:.0%}*\n\n"

        if self.sources:
            md += "## Sources\n\n"
            for source in self.sources:
                md += f"- [{source.title}]({source.url})\n"

        return md

    def _to_html(self) -> str:
        """Convert to HTML."""
        html = f"<html><head><title>{self.title}</title></head><body>\n"
        html += f"<h1>{self.title}</h1>\n"
        html += (
            f"<p><em>Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}</em></p>\n"
        )

        html += "<h2>Summary</h2>\n"
        html += f"<p>{self.summary}</p>\n"

        if self.findings:
            html += "<h2>Key Findings</h2>\n"
            for i, finding in enumerate(self.findings, 1):
                html += f"<h3>{i}. {finding.key_point}</h3>\n"
                html += f"<p>{finding.evidence}</p>\n"
                html += f"<p><em>Confidence: {finding.confidence:.0%}</em></p>\n"

        if self.sources:
            html += "<h2>Sources</h2>\n<ul>\n"
            for source in self.sources:
                html += f'<li><a href="{source.url}">{source.title}</a></li>\n'
            html += "</ul>\n"

        html += "</body></html>"
        return html

    def _to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(
            {
                "id": self.id,
                "title": self.title,
                "summary": self.summary,
                "query": self.query.to_dict(),
                "findings": [f.to_dict() for f in self.findings],
                "sources": [s.to_dict() for s in self.sources],
                "created_at": self.created_at.isoformat(),
            },
            indent=2,
        )

    def _to_text(self) -> str:
        """Convert to plain text."""
        text = f"{self.title}\n{'=' * len(self.title)}\n\n"
        text += f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"

        text += "SUMMARY\n-------\n"
        text += f"{self.summary}\n\n"

        if self.findings:
            text += "KEY FINDINGS\n------------\n"
            for i, finding in enumerate(self.findings, 1):
                text += f"\n{i}. {finding.key_point}\n"
                text += f"   {finding.evidence}\n"

        return text


# =============================================================================
# Part 5: Search Tools - SOLUTION
# =============================================================================
class SearchTool(ABC):
    """Abstract base class for search tools."""

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """Get the source type this tool searches."""
        pass

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Execute a search."""
        pass


class WebSearchTool(SearchTool):
    """
    Web search tool implementation.

    Solution implements:
    - Web search (mock/real)
    - Result parsing
    - Rate limiting
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._last_search = None
        self._rate_limit = 1.0  # seconds between searches

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Execute web search."""
        # Rate limiting
        if self._last_search:
            elapsed = (datetime.now() - self._last_search).total_seconds()
            if elapsed < self._rate_limit:
                await asyncio.sleep(self._rate_limit - elapsed)

        self._last_search = datetime.now()

        # In production, call actual search API
        # Mock results for demonstration
        results = []
        for i in range(min(max_results, 5)):
            results.append(
                SearchResult(
                    title=f"Web result {i+1} for: {query}",
                    url=f"https://example.com/result/{i+1}",
                    snippet=f"This is a search result about {query}...",
                    source=SourceType.WEB,
                    relevance_score=0.9 - (i * 0.1),
                )
            )

        return results


class AcademicSearchTool(SearchTool):
    """
    Academic search tool implementation.

    Solution implements:
    - Academic database search
    - Paper metadata extraction
    - Citation handling
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def source_type(self) -> SourceType:
        return SourceType.ACADEMIC

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Execute academic search."""
        # Mock results
        results = []
        for i in range(min(max_results, 3)):
            results.append(
                SearchResult(
                    title=f"Academic paper: {query}",
                    url=f"https://academic.example.com/paper/{i+1}",
                    snippet=f"Research findings on {query}...",
                    source=SourceType.ACADEMIC,
                    relevance_score=0.95 - (i * 0.05),
                    authors=[f"Author {j+1}" for j in range(2)],
                    published_date=datetime.now(),
                )
            )

        return results


# =============================================================================
# Part 6: Document Tools - SOLUTION
# =============================================================================
class DocumentReaderTool:
    """
    Reads and extracts content from documents.

    Solution implements:
    - URL fetching
    - Content extraction
    - Format handling
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    async def read(self, url: str) -> str:
        """Read content from a URL."""
        if url in self._cache:
            return self._cache[url]

        # In production, fetch and parse actual content
        # Mock for demonstration
        content = f"Content from {url}\n\n"
        content += "This is the main content of the document. "
        content += "It contains important information about the topic. "
        content += "The document discusses various aspects in detail."

        self._cache[url] = content
        return content

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self._cache = {}


class SummarizerTool:
    """
    Summarizes text content.

    Solution implements:
    - Text summarization
    - Length control
    - Key point extraction
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text."""
        # In production, use LLM for summarization
        # Simple extractive summary for demonstration
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        summary = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                summary.append(sentence)
                current_length += len(sentence)
            else:
                break

        return ". ".join(summary) + "." if summary else ""

    async def extract_key_points(self, text: str, count: int = 5) -> List[str]:
        """Extract key points from text."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

        # Simple scoring based on sentence length and position
        scored = []
        for i, s in enumerate(sentences):
            score = len(s.split()) * (1 - i / len(sentences))
            scored.append((s, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:count]]


class CitationTool:
    """
    Generates and formats citations.

    Solution implements:
    - Multiple citation styles
    - Source formatting
    - Bibliography generation
    """

    STYLES = ["apa", "mla", "chicago", "harvard"]

    async def generate(self, source: SearchResult, style: str = "apa") -> str:
        """Generate a citation for a source."""
        return source.to_citation(style)

    async def generate_bibliography(
        self, sources: List[SearchResult], style: str = "apa"
    ) -> str:
        """Generate a complete bibliography."""
        citations = []

        for source in sources:
            citation = await self.generate(source, style)
            citations.append(citation)

        # Sort alphabetically
        citations.sort()

        return "\n\n".join(citations)


# =============================================================================
# Part 7: Agents - SOLUTION
# =============================================================================
class AgentBase(ABC):
    """Abstract base class for research agents."""

    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """Execute the agent's task."""
        pass


class PlannerAgent(AgentBase):
    """
    Plans research strategy.

    Solution implements:
    - Query analysis
    - Task breakdown
    - Resource allocation
    """

    def __init__(self):
        super().__init__("planner")

    async def plan(self, query: ResearchQuery) -> Dict[str, Any]:
        """Create a research plan."""
        keywords = query.extract_keywords()

        # Determine search strategy
        search_tasks = []
        for source in query.sources:
            search_tasks.append(
                {
                    "type": "search",
                    "source": source.value,
                    "query": query.question,
                    "keywords": keywords,
                }
            )

        # Analysis tasks
        analysis_tasks = [
            {"type": "analyze", "focus": "key_findings"},
            {"type": "analyze", "focus": "compare_sources"},
        ]

        # Writing tasks
        writing_tasks = [
            {"type": "write", "section": "summary"},
            {"type": "write", "section": "findings"},
            {"type": "write", "section": "conclusion"},
        ]

        return {
            "query_id": query.id,
            "keywords": keywords,
            "steps": [
                {"phase": "search", "tasks": search_tasks},
                {"phase": "analyze", "tasks": analysis_tasks},
                {"phase": "write", "tasks": writing_tasks},
            ],
            "estimated_time": len(search_tasks) * 2
            + len(analysis_tasks)
            + len(writing_tasks),
        }

    async def execute(self, input_data: Any) -> Any:
        """Execute planning."""
        if isinstance(input_data, ResearchQuery):
            return await self.plan(input_data)
        return await self.plan(ResearchQuery(question=str(input_data)))


class ResearcherAgent(AgentBase):
    """
    Performs research tasks.

    Solution implements:
    - Search execution
    - Result aggregation
    - Relevance filtering
    """

    def __init__(self):
        super().__init__("researcher")
        self._tools: Dict[SourceType, SearchTool] = {
            SourceType.WEB: WebSearchTool(),
            SourceType.ACADEMIC: AcademicSearchTool(),
        }

    async def research(self, task: Dict[str, Any]) -> List[SearchResult]:
        """Execute research task."""
        query = task.get("query", "")
        sources = task.get("sources", ["web"])

        all_results = []

        for source in sources:
            source_type = SourceType(source) if isinstance(source, str) else source
            tool = self._tools.get(source_type)

            if tool:
                results = await tool.search(query, max_results=5)
                all_results.extend(results)

        # Sort by relevance
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return all_results

    async def execute(self, input_data: Any) -> Any:
        """Execute research."""
        if isinstance(input_data, dict):
            return await self.research(input_data)
        return await self.research({"query": str(input_data)})


class AnalyzerAgent(AgentBase):
    """
    Analyzes research results.

    Solution implements:
    - Pattern recognition
    - Finding extraction
    - Confidence scoring
    """

    def __init__(self):
        super().__init__("analyzer")
        self._summarizer = SummarizerTool()

    async def analyze(self, results: List[SearchResult]) -> List[ResearchFinding]:
        """Analyze search results to extract findings."""
        findings = []

        # Group by source type
        by_source: Dict[SourceType, List[SearchResult]] = {}
        for result in results:
            if result.source not in by_source:
                by_source[result.source] = []
            by_source[result.source].append(result)

        # Extract findings from each group
        for source_type, source_results in by_source.items():
            if not source_results:
                continue

            # Create finding from top results
            top = source_results[0]

            finding = ResearchFinding(
                key_point=f"Finding from {source_type.value} sources",
                evidence=top.snippet,
                sources=source_results[:3],
                confidence=sum(r.relevance_score for r in source_results[:3])
                / min(3, len(source_results)),
                category=source_type.value,
            )

            findings.append(finding)

        return findings

    async def execute(self, input_data: Any) -> Any:
        """Execute analysis."""
        if isinstance(input_data, list):
            return await self.analyze(input_data)
        return []


class WriterAgent(AgentBase):
    """
    Writes research reports.

    Solution implements:
    - Report generation
    - Section writing
    - Formatting
    """

    def __init__(self):
        super().__init__("writer")

    async def write(
        self, query: ResearchQuery, findings: List[ResearchFinding]
    ) -> ResearchReport:
        """Write a research report."""
        # Generate title
        keywords = query.extract_keywords()
        title = f"Research Report: {' '.join(keywords[:3]).title()}"

        # Generate summary
        if findings:
            summary_points = [f.key_point for f in findings[:3]]
            summary = f"This report examines: {', '.join(summary_points)}. "
            summary += f"Based on analysis of {len(findings)} key findings from multiple sources."
        else:
            summary = f"Research conducted on: {query.question}"

        # Collect all sources
        all_sources = []
        for finding in findings:
            all_sources.extend(finding.sources)

        # Remove duplicates
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        return ResearchReport(
            query=query,
            title=title,
            summary=summary,
            findings=findings,
            sources=unique_sources,
        )

    async def execute(self, input_data: Any) -> Any:
        """Execute writing."""
        if isinstance(input_data, dict):
            query = input_data.get("query", ResearchQuery(question=""))
            findings = input_data.get("findings", [])
            return await self.write(query, findings)
        return None


# =============================================================================
# Part 8: Research Pipeline - SOLUTION
# =============================================================================
class ResearchPipeline:
    """
    Orchestrates the research process.

    Solution implements:
    - Agent coordination
    - Pipeline execution
    - Error handling
    """

    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.analyzer = AnalyzerAgent()
        self.writer = WriterAgent()
        self.steps: Dict[str, Callable] = {}
        self._history: List[Dict[str, Any]] = []

    def add_step(self, name: str, handler: Callable) -> None:
        """Add a custom step to the pipeline."""
        self.steps[name] = handler

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Run the full research pipeline."""
        start_time = datetime.now()

        try:
            # Step 1: Plan
            plan = await self.planner.plan(query)
            self._log_step("plan", plan)

            # Step 2: Research
            research_task = {
                "query": query.question,
                "sources": [s.value for s in query.sources],
            }
            results = await self.researcher.research(research_task)
            self._log_step("research", {"result_count": len(results)})

            # Step 3: Analyze
            findings = await self.analyzer.analyze(results)
            self._log_step("analyze", {"finding_count": len(findings)})

            # Step 4: Run custom steps
            data = {"query": query, "results": results, "findings": findings}

            for step_name, handler in self.steps.items():
                if asyncio.iscoroutinefunction(handler):
                    data = await handler(data)
                else:
                    data = handler(data)
                self._log_step(step_name, {"custom": True})

            # Step 5: Write
            report = await self.writer.write(query, data.get("findings", findings))
            self._log_step("write", {"completed": True})

            execution_time = (datetime.now() - start_time).total_seconds()
            report.metadata["execution_time"] = execution_time

            return report

        except Exception as e:
            self._log_step("error", {"error": str(e)})
            # Return minimal report on error
            return ResearchReport(
                query=query,
                title="Research Error",
                summary=f"An error occurred during research: {str(e)}",
                findings=[],
                sources=[],
            )

    def _log_step(self, step: str, data: Dict[str, Any]) -> None:
        """Log a pipeline step."""
        self._history.append(
            {"step": step, "timestamp": datetime.now().isoformat(), "data": data}
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history."""
        return self._history.copy()


# =============================================================================
# Part 9: Quality Checker - SOLUTION
# =============================================================================
class QualityChecker:
    """
    Checks research quality.

    Solution implements:
    - Source validation
    - Finding verification
    - Report scoring
    """

    def __init__(self, min_sources: int = 2, min_confidence: float = 0.5):
        self.min_sources = min_sources
        self.min_confidence = min_confidence

    def check(self, report: ResearchReport) -> Dict[str, Any]:
        """Check report quality."""
        scores = {
            "source_score": self.check_sources(report.sources),
            "finding_score": self.check_findings(report.findings),
            "content_score": self._check_content(report),
            "structure_score": self._check_structure(report),
        }

        overall = sum(scores.values()) / len(scores)

        return {
            "passed": overall >= 0.5,
            "score": overall,
            "scores": scores,
            "issues": self._identify_issues(report),
        }

    def check_sources(self, sources: List[SearchResult]) -> float:
        """Check source quality."""
        if not sources:
            return 0.0

        score = 0.0

        # Check source count
        if len(sources) >= self.min_sources:
            score += 0.3

        # Check source diversity
        source_types = set(s.source for s in sources)
        score += min(0.3, len(source_types) * 0.1)

        # Check relevance scores
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        score += avg_relevance * 0.4

        return min(1.0, score)

    def check_findings(self, findings: List[ResearchFinding]) -> float:
        """Check finding quality."""
        if not findings:
            return 0.0

        score = 0.0

        # Check finding count
        score += min(0.3, len(findings) * 0.1)

        # Check confidence levels
        high_confidence = sum(
            1 for f in findings if f.confidence >= self.min_confidence
        )
        score += (high_confidence / len(findings)) * 0.4

        # Check source backing
        backed = sum(1 for f in findings if len(f.sources) > 0)
        score += (backed / len(findings)) * 0.3

        return min(1.0, score)

    def _check_content(self, report: ResearchReport) -> float:
        """Check content quality."""
        score = 0.0

        # Title present and meaningful
        if report.title and len(report.title) > 10:
            score += 0.3

        # Summary present and substantial
        if report.summary and len(report.summary) > 50:
            score += 0.4

        # Has findings
        if report.findings:
            score += 0.3

        return min(1.0, score)

    def _check_structure(self, report: ResearchReport) -> float:
        """Check report structure."""
        score = 0.0

        # Has basic components
        if report.title:
            score += 0.25
        if report.summary:
            score += 0.25
        if report.findings:
            score += 0.25
        if report.sources:
            score += 0.25

        return score

    def _identify_issues(self, report: ResearchReport) -> List[str]:
        """Identify quality issues."""
        issues = []

        if not report.sources:
            issues.append("No sources provided")
        elif len(report.sources) < self.min_sources:
            issues.append(f"Insufficient sources (need at least {self.min_sources})")

        if not report.findings:
            issues.append("No findings extracted")

        low_confidence = [
            f for f in report.findings if f.confidence < self.min_confidence
        ]
        if low_confidence:
            issues.append(f"{len(low_confidence)} findings have low confidence")

        if not report.summary:
            issues.append("Missing summary")

        return issues


# =============================================================================
# Part 10: Output Formatter - SOLUTION
# =============================================================================
class OutputFormatter:
    """
    Formats research output.

    Solution implements:
    - Multiple formats
    - Customization options
    - Export functionality
    """

    def __init__(self, default_format: ReportFormat = ReportFormat.MARKDOWN):
        self.default_format = default_format
        self._templates: Dict[str, str] = {}

    def format(
        self, report: ResearchReport, output_format: Optional[ReportFormat] = None
    ) -> str:
        """Format a report."""
        fmt = output_format or self.default_format
        return report.format(fmt)

    def set_template(self, name: str, template: str) -> None:
        """Set a custom template."""
        self._templates[name] = template

    def format_with_template(self, report: ResearchReport, template_name: str) -> str:
        """Format using a custom template."""
        template = self._templates.get(template_name, "{title}\n\n{summary}")

        return template.format(
            title=report.title,
            summary=report.summary,
            findings="\n".join(f.key_point for f in report.findings),
            source_count=len(report.sources),
            finding_count=len(report.findings),
            date=report.created_at.strftime("%Y-%m-%d"),
        )

    def export(
        self, report: ResearchReport, path: str, output_format: ReportFormat
    ) -> str:
        """Export report to file."""
        content = self.format(report, output_format)

        with open(path, "w") as f:
            f.write(content)

        return path


# =============================================================================
# Part 11: Research Assistant - SOLUTION
# =============================================================================
@dataclass
class ResearchConfig:
    """Configuration for research assistant."""

    max_sources: int = 10
    quality_threshold: float = 0.5
    default_sources: List[SourceType] = field(
        default_factory=lambda: [SourceType.WEB, SourceType.ACADEMIC]
    )
    timeout: float = 120.0


class ResearchAssistant:
    """
    Complete research assistant.

    Solution implements:
    - Full research workflow
    - Quality control
    - Multiple output formats
    """

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.pipeline = ResearchPipeline()
        self.quality_checker = QualityChecker()
        self.formatter = OutputFormatter()
        self._research_history: List[ResearchReport] = []

    async def research(
        self,
        question: str,
        sources: Optional[List[SourceType]] = None,
        output_format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ResearchReport:
        """Conduct research on a question."""
        # Create query
        query = ResearchQuery(
            question=question,
            sources=sources or self.config.default_sources,
            max_results=self.config.max_sources,
        )

        # Run pipeline
        report = await self.pipeline.run(query)

        # Check quality
        quality = self.quality_checker.check(report)
        report.metadata["quality"] = quality

        # Store in history
        self._research_history.append(report)

        return report

    def get_formatted_output(
        self, report: ResearchReport, output_format: ReportFormat
    ) -> str:
        """Get formatted output."""
        return self.formatter.format(report, output_format)

    def get_history(self) -> List[ResearchReport]:
        """Get research history."""
        return self._research_history.copy()

    def clear_history(self) -> None:
        """Clear research history."""
        self._research_history = []

    async def batch_research(
        self, questions: List[str], sources: Optional[List[SourceType]] = None
    ) -> List[ResearchReport]:
        """Research multiple questions."""
        reports = []

        for question in questions:
            report = await self.research(question, sources)
            reports.append(report)

        return reports

    async def compare(
        self, questions: List[str], sources: Optional[List[SourceType]] = None
    ) -> ResearchReport:
        """Comparative research on multiple questions."""
        # Research each question
        individual_reports = await self.batch_research(questions, sources)

        # Combine findings
        all_findings = []
        all_sources = []

        for report in individual_reports:
            all_findings.extend(report.findings)
            all_sources.extend(report.sources)

        # Create comparative report
        query = ResearchQuery(
            question=f"Comparison: {', '.join(questions)}",
            query_type=QueryType.COMPARATIVE,
            sources=sources or self.config.default_sources,
        )

        return ResearchReport(
            query=query,
            title=f"Comparative Analysis: {len(questions)} Topics",
            summary=f"Comparative analysis of {len(questions)} research questions, "
            f"yielding {len(all_findings)} findings from {len(all_sources)} sources.",
            findings=all_findings,
            sources=list({s.url: s for s in all_sources}.values()),  # Deduplicate
        )


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":

    async def main():
        # Create assistant
        config = ResearchConfig(max_sources=5, quality_threshold=0.4)
        assistant = ResearchAssistant(config)

        # Conduct research
        report = await assistant.research(
            "What are the latest advances in AI and machine learning?",
            sources=[SourceType.WEB, SourceType.ACADEMIC],
        )

        # Print report
        print(assistant.get_formatted_output(report, ReportFormat.MARKDOWN))

        # Check quality
        print("\n--- Quality Report ---")
        print(json.dumps(report.metadata.get("quality", {}), indent=2))

        # Comparative research
        print("\n--- Comparative Research ---")
        comparison = await assistant.compare(
            ["Benefits of AI in healthcare", "Benefits of AI in education"]
        )
        print(assistant.get_formatted_output(comparison, ReportFormat.TEXT))

    asyncio.run(main())

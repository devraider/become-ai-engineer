"""
Solutions for Week 10 - Project: Research Assistant
===================================================
"""

from typing import TypedDict, Annotated, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid
import re


# =============================================================================
# PART 1: State Definitions
# =============================================================================
class ResearchPhase(Enum):
    QUERY_ANALYSIS = "query_analysis"
    PLANNING = "planning"
    SEARCHING = "searching"
    EVALUATING = "evaluating"
    SYNTHESIZING = "synthesizing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


@dataclass
class Source:
    """Represents an information source."""

    id: str
    title: str
    content: str
    url: str | None = None
    credibility_score: float = 0.5
    relevance_score: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": (
                self.content[:200] + "..." if len(self.content) > 200 else self.content
            ),
            "url": self.url,
            "credibility": self.credibility_score,
            "relevance": self.relevance_score,
        }


@dataclass
class ResearchPlan:
    """A plan for conducting research."""

    query: str
    sub_questions: list[str]
    search_strategies: list[str]
    estimated_steps: int

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "sub_questions": self.sub_questions,
            "search_strategies": self.search_strategies,
            "estimated_steps": self.estimated_steps,
        }


def append_list(existing: list, new: list) -> list:
    return existing + new


class ResearchState(TypedDict):
    """Complete research state."""

    query: str
    phase: str
    plan: dict | None
    sources: list[dict]
    findings: list[str]
    synthesis: str
    citations: list[str]
    needs_review: bool
    approved: bool | None
    error: str | None
    metadata: dict


# =============================================================================
# PART 2: Query Analysis
# =============================================================================
class QueryType(Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    EXPLANATORY = "explanatory"
    INVESTIGATIVE = "investigative"


class QueryAnalyzer:
    """Analyzes user queries."""

    COMPARATIVE_KEYWORDS = ["compare", "versus", "vs", "difference", "better"]
    EXPLANATORY_KEYWORDS = ["explain", "how", "why", "what is", "describe"]
    INVESTIGATIVE_KEYWORDS = ["investigate", "research", "analyze", "deep dive"]

    def analyze(self, query: str) -> dict:
        """Analyze the query."""
        query_lower = query.lower()

        # Determine query type
        if any(kw in query_lower for kw in self.COMPARATIVE_KEYWORDS):
            query_type = QueryType.COMPARATIVE
        elif any(kw in query_lower for kw in self.INVESTIGATIVE_KEYWORDS):
            query_type = QueryType.INVESTIGATIVE
        elif any(kw in query_lower for kw in self.EXPLANATORY_KEYWORDS):
            query_type = QueryType.EXPLANATORY
        else:
            query_type = QueryType.FACTUAL

        # Determine complexity
        word_count = len(query.split())
        if word_count > 15 or query_type == QueryType.INVESTIGATIVE:
            complexity = "complex"
        elif word_count > 8 or query_type == QueryType.COMPARATIVE:
            complexity = "medium"
        else:
            complexity = "simple"

        entities = self.extract_entities(query)
        sub_questions = self.generate_sub_questions(query, query_type)

        return {
            "query_type": query_type,
            "complexity": complexity,
            "entities": entities,
            "sub_questions": sub_questions,
        }

    def extract_entities(self, query: str) -> list[str]:
        """Extract key entities."""
        # Extract capitalized words (potential proper nouns)
        caps = re.findall(r"\b[A-Z][a-z]+\b", query)
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        return list(set(caps + quoted))

    def generate_sub_questions(self, query: str, query_type: QueryType) -> list[str]:
        """Generate sub-questions."""
        if query_type == QueryType.FACTUAL:
            return [query]
        elif query_type == QueryType.COMPARATIVE:
            return [
                f"What are the key features of each item in: {query}",
                f"What are the pros and cons related to: {query}",
            ]
        elif query_type == QueryType.EXPLANATORY:
            return [
                f"What is the definition related to: {query}",
                f"How does it work: {query}",
                f"Why is this important: {query}",
            ]
        else:  # INVESTIGATIVE
            return [
                f"Background on: {query}",
                f"Current state of: {query}",
                f"Different perspectives on: {query}",
                f"Future implications of: {query}",
            ]


def analyze_query_node(state: ResearchState) -> dict:
    """Node that analyzes the query."""
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(state["query"])
    return {
        "phase": ResearchPhase.PLANNING.value,
        "metadata": {**state.get("metadata", {}), "analysis": analysis},
    }


# =============================================================================
# PART 3: Research Planning
# =============================================================================
class ResearchPlanner:
    """Creates research plans."""

    def create_plan(self, query: str, analysis: dict) -> ResearchPlan:
        """Create a research plan."""
        sub_questions = analysis.get("sub_questions", [query])

        strategies = ["keyword_search"]
        if analysis.get("complexity") in ["medium", "complex"]:
            strategies.append("semantic_search")
        if analysis.get("query_type") == QueryType.INVESTIGATIVE:
            strategies.append("deep_search")

        return ResearchPlan(
            query=query,
            sub_questions=sub_questions,
            search_strategies=strategies,
            estimated_steps=self.estimate_complexity(analysis),
        )

    def estimate_complexity(self, analysis: dict) -> int:
        """Estimate research steps."""
        complexity = analysis.get("complexity", "simple")
        if complexity == "simple":
            return 3
        elif complexity == "medium":
            return 5
        return 8


def create_plan_node(state: ResearchState) -> dict:
    """Node that creates the research plan."""
    planner = ResearchPlanner()
    analysis = state.get("metadata", {}).get("analysis", {})
    plan = planner.create_plan(state["query"], analysis)
    return {"plan": plan.to_dict(), "phase": ResearchPhase.SEARCHING.value}


# =============================================================================
# PART 4: Search Execution
# =============================================================================
class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        pass


class MockSearchEngine(SearchEngine):
    """Mock search engine."""

    def __init__(self):
        self.knowledge_base = {
            "python": [
                Source(
                    id="1",
                    title="Python Overview",
                    content="Python is a high-level, interpreted programming language known for its readability and versatility.",
                    url="https://python.org",
                    credibility_score=0.95,
                ),
                Source(
                    id="2",
                    title="Python Applications",
                    content="Python is widely used in web development, data science, AI/ML, automation, and scientific computing.",
                    credibility_score=0.9,
                ),
            ],
            "machine learning": [
                Source(
                    id="3",
                    title="ML Fundamentals",
                    content="Machine learning is a subset of AI that enables systems to learn and improve from experience.",
                    credibility_score=0.92,
                ),
                Source(
                    id="4",
                    title="ML vs Deep Learning",
                    content="ML uses algorithms to parse data, while deep learning structures algorithms in layers.",
                    credibility_score=0.88,
                ),
            ],
            "langgraph": [
                Source(
                    id="5",
                    title="LangGraph Introduction",
                    content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
                    url="https://langchain-ai.github.io/langgraph/",
                    credibility_score=0.95,
                )
            ],
            "ai": [
                Source(
                    id="6",
                    title="AI Overview",
                    content="Artificial Intelligence refers to systems designed to perform tasks requiring human intelligence.",
                    credibility_score=0.9,
                )
            ],
        }

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """Search the mock knowledge base."""
        results = []
        query_lower = query.lower()

        for key, sources in self.knowledge_base.items():
            if key in query_lower or any(word in key for word in query_lower.split()):
                results.extend(sources)

        # Assign relevance scores
        for source in results:
            source.relevance_score = 0.7 + (
                0.3 if query_lower in source.content.lower() else 0
            )

        return results[:max_results]


class SearchExecutor:
    """Executes searches across engines."""

    def __init__(self, engines: list[SearchEngine] = None):
        self.engines = engines or [MockSearchEngine()]

    def execute_searches(self, plan: ResearchPlan) -> list[Source]:
        """Execute all searches in the plan."""
        all_sources = []

        for question in plan.sub_questions:
            for engine in self.engines:
                sources = engine.search(question)
                all_sources.extend(sources)

        return self.deduplicate(all_sources)

    def deduplicate(self, sources: list[Source]) -> list[Source]:
        """Remove duplicate sources."""
        seen_ids = set()
        unique = []
        for source in sources:
            if source.id not in seen_ids:
                seen_ids.add(source.id)
                unique.append(source)
        return unique


def search_node(state: ResearchState) -> dict:
    """Node that executes searches."""
    executor = SearchExecutor()
    plan_dict = state.get("plan", {})
    plan = ResearchPlan(
        query=plan_dict.get("query", state["query"]),
        sub_questions=plan_dict.get("sub_questions", [state["query"]]),
        search_strategies=plan_dict.get("search_strategies", ["keyword"]),
        estimated_steps=plan_dict.get("estimated_steps", 3),
    )

    sources = executor.execute_searches(plan)
    return {
        "sources": [s.to_dict() for s in sources],
        "phase": ResearchPhase.EVALUATING.value,
    }


# =============================================================================
# PART 5: Source Evaluation
# =============================================================================
class SourceEvaluator:
    """Evaluates source credibility and relevance."""

    TRUSTED_DOMAINS = [".edu", ".gov", ".org"]

    def evaluate_credibility(self, source: Source) -> float:
        """Evaluate source credibility."""
        score = source.credibility_score

        if source.url:
            for domain in self.TRUSTED_DOMAINS:
                if domain in source.url:
                    score = min(1.0, score + 0.1)
                    break

        # Content quality indicators
        if len(source.content) > 100:
            score = min(1.0, score + 0.05)

        return score

    def evaluate_relevance(self, source: Source, query: str) -> float:
        """Evaluate source relevance."""
        query_words = set(query.lower().split())
        content_words = set(source.content.lower().split())
        title_words = set(source.title.lower().split())

        # Calculate overlap
        content_overlap = len(query_words & content_words) / max(len(query_words), 1)
        title_overlap = len(query_words & title_words) / max(len(query_words), 1)

        return min(
            1.0,
            (content_overlap * 0.7 + title_overlap * 0.3)
            + source.relevance_score * 0.5,
        )

    def filter_sources(
        self, sources: list[Source], min_score: float = 0.5
    ) -> list[Source]:
        """Filter sources below minimum score."""
        return [s for s in sources if s.credibility_score >= min_score]


def evaluate_sources_node(state: ResearchState) -> dict:
    """Node that evaluates and filters sources."""
    evaluator = SourceEvaluator()

    # Reconstruct Source objects
    sources = [
        Source(
            id=s["id"],
            title=s["title"],
            content=s["content"],
            url=s.get("url"),
            credibility_score=s.get("credibility", 0.5),
            relevance_score=s.get("relevance", 0.5),
        )
        for s in state.get("sources", [])
    ]

    # Evaluate
    for source in sources:
        source.credibility_score = evaluator.evaluate_credibility(source)
        source.relevance_score = evaluator.evaluate_relevance(source, state["query"])

    # Filter
    filtered = evaluator.filter_sources(sources, min_score=0.4)

    return {
        "sources": [s.to_dict() for s in filtered],
        "phase": ResearchPhase.SYNTHESIZING.value,
    }


# =============================================================================
# PART 6: Synthesis & Citation
# =============================================================================
class Synthesizer:
    """Synthesizes findings into coherent response."""

    def extract_findings(self, sources: list[Source]) -> list[str]:
        """Extract key findings from sources."""
        findings = []
        for source in sources:
            # Extract first sentence as key finding
            content = source.content
            if ". " in content:
                finding = content.split(". ")[0] + "."
            else:
                finding = content[:100] + "..."
            findings.append(finding)
        return findings

    def synthesize(self, query: str, findings: list[str]) -> str:
        """Synthesize findings into coherent response."""
        if not findings:
            return f"I couldn't find specific information about: {query}"

        response_parts = [f"Regarding your query about '{query}':\n"]

        for i, finding in enumerate(findings, 1):
            response_parts.append(f"• {finding}")

        response_parts.append(
            f"\nIn summary, based on {len(findings)} sources, the research provides comprehensive insights into your query."
        )

        return "\n".join(response_parts)

    def generate_citations(self, sources: list[Source]) -> list[str]:
        """Generate citation list."""
        citations = []
        for source in sources:
            url_part = source.url if source.url else "No URL"
            citations.append(f"[{source.id}] {source.title}. {url_part}")
        return citations


def synthesize_node(state: ResearchState) -> dict:
    """Node that synthesizes response."""
    synthesizer = Synthesizer()

    sources = [
        Source(
            id=s["id"],
            title=s["title"],
            content=s["content"],
            url=s.get("url"),
            credibility_score=s.get("credibility", 0.5),
        )
        for s in state.get("sources", [])
    ]

    findings = synthesizer.extract_findings(sources)
    synthesis = synthesizer.synthesize(state["query"], findings)
    citations = synthesizer.generate_citations(sources)

    return {
        "findings": findings,
        "synthesis": synthesis,
        "citations": citations,
        "phase": ResearchPhase.REVIEWING.value,
    }


# =============================================================================
# PART 7: Human Review
# =============================================================================
class HumanReviewManager:
    """Manages human review workflow."""

    REVIEW_TRIGGERS = ["sensitive", "confidential", "legal", "medical", "financial"]

    def needs_review(self, state: dict) -> bool:
        """Determine if human review is needed."""
        query = state.get("query", "").lower()
        synthesis = state.get("synthesis", "").lower()

        # Check for sensitive topics
        for trigger in self.REVIEW_TRIGGERS:
            if trigger in query or trigger in synthesis:
                return True

        # Check for low source count
        sources = state.get("sources", [])
        if len(sources) < 2:
            return True

        return False

    def request_review(self, state: dict) -> dict:
        """Prepare review request."""
        return {
            "query": state.get("query"),
            "synthesis": state.get("synthesis"),
            "source_count": len(state.get("sources", [])),
            "citations": state.get("citations", []),
        }

    def process_approval(self, approved: bool, feedback: str = "") -> dict:
        """Process human approval/rejection."""
        return {
            "approved": approved,
            "feedback": feedback,
            "phase": (
                ResearchPhase.COMPLETE.value
                if approved
                else ResearchPhase.REVIEWING.value
            ),
        }


def check_review_needed(state: ResearchState) -> str:
    """Conditional edge function."""
    manager = HumanReviewManager()
    if state.get("approved") is not None:
        return "complete"
    if manager.needs_review(state):
        return "review"
    return "complete"


def request_review_node(state: ResearchState) -> dict:
    """Node that requests human review."""
    return {"needs_review": True}


# =============================================================================
# PART 8: Complete Research Graph
# =============================================================================
class ResearchAssistant:
    """Complete research assistant."""

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer if checkpointer is not None else {}
        self.query_analyzer = QueryAnalyzer()
        self.planner = ResearchPlanner()
        self.search_executor = SearchExecutor()
        self.source_evaluator = SourceEvaluator()
        self.synthesizer = Synthesizer()
        self.review_manager = HumanReviewManager()

    def _initialize_state(self, query: str) -> ResearchState:
        """Initialize state for new research."""
        return {
            "query": query,
            "phase": ResearchPhase.QUERY_ANALYSIS.value,
            "plan": None,
            "sources": [],
            "findings": [],
            "synthesis": "",
            "citations": [],
            "needs_review": False,
            "approved": None,
            "error": None,
            "metadata": {},
        }

    def _save_checkpoint(self, thread_id: str, state: dict) -> None:
        """Save checkpoint for thread."""
        self.checkpointer[thread_id] = state

    def _load_checkpoint(self, thread_id: str) -> dict | None:
        """Load checkpoint for thread."""
        return self.checkpointer.get(thread_id)

    def research(self, query: str, thread_id: str = None) -> dict:
        """Execute research workflow."""
        thread_id = thread_id or str(uuid.uuid4())

        # Load or initialize state
        state = self._load_checkpoint(thread_id)
        if state is None:
            state = self._initialize_state(query)

        # Run through phases
        while state["phase"] != ResearchPhase.COMPLETE.value:
            phase = state["phase"]

            if phase == ResearchPhase.QUERY_ANALYSIS.value:
                update = analyze_query_node(state)
            elif phase == ResearchPhase.PLANNING.value:
                update = create_plan_node(state)
            elif phase == ResearchPhase.SEARCHING.value:
                update = search_node(state)
            elif phase == ResearchPhase.EVALUATING.value:
                update = evaluate_sources_node(state)
            elif phase == ResearchPhase.SYNTHESIZING.value:
                update = synthesize_node(state)
            elif phase == ResearchPhase.REVIEWING.value:
                decision = check_review_needed(state)
                if decision == "review" and state.get("approved") is None:
                    state["needs_review"] = True
                    break  # Pause for human review
                state["phase"] = ResearchPhase.COMPLETE.value
                break
            else:
                break

            state = {**state, **update}
            self._save_checkpoint(thread_id, state)

        return state

    def provide_feedback(
        self, thread_id: str, approved: bool, feedback: str = ""
    ) -> dict:
        """Process human feedback and continue."""
        state = self._load_checkpoint(thread_id)
        if not state:
            return {"error": "Thread not found"}

        update = self.review_manager.process_approval(approved, feedback)
        state = {**state, **update, "needs_review": False}
        self._save_checkpoint(thread_id, state)
        return state

    def get_status(self, thread_id: str) -> dict:
        """Get current status of research thread."""
        state = self._load_checkpoint(thread_id)
        if not state:
            return {"error": "Thread not found"}
        return {
            "phase": state["phase"],
            "needs_review": state.get("needs_review", False),
            "has_synthesis": bool(state.get("synthesis")),
        }


# =============================================================================
# PART 9: Streaming Support
# =============================================================================
class StreamingResearchAssistant(ResearchAssistant):
    """Research assistant with streaming support."""

    def stream_research(self, query: str, thread_id: str = None):
        """Stream research execution."""
        thread_id = thread_id or str(uuid.uuid4())
        state = self._initialize_state(query)

        yield {"phase": "analyzing", "message": f"Analyzing query: {query}"}
        update = analyze_query_node(state)
        state = {**state, **update}

        yield {
            "phase": "planning",
            "message": "Creating research plan...",
            "data": state.get("metadata", {}).get("analysis"),
        }
        update = create_plan_node(state)
        state = {**state, **update}

        yield {"phase": "searching", "message": "Searching sources...", "progress": 0.3}
        update = search_node(state)
        state = {**state, **update}
        yield {
            "phase": "searching",
            "message": f"Found {len(state['sources'])} sources",
            "progress": 0.6,
        }

        yield {
            "phase": "evaluating",
            "message": "Evaluating sources...",
            "sources": len(state["sources"]),
        }
        update = evaluate_sources_node(state)
        state = {**state, **update}

        yield {"phase": "synthesizing", "message": "Synthesizing findings..."}
        update = synthesize_node(state)
        state = {**state, **update}

        self._save_checkpoint(thread_id, state)

        yield {
            "phase": "complete",
            "result": {
                "synthesis": state["synthesis"],
                "citations": state["citations"],
                "source_count": len(state["sources"]),
            },
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Week 10 - Project Solution: Research Assistant")
    print("=" * 70)

    # Test Query Analysis
    print("\n--- Query Analysis ---")
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("Compare Python and JavaScript for web development")
    print(f"Analysis: {analysis}")

    # Test Research Planning
    print("\n--- Research Planning ---")
    planner = ResearchPlanner()
    plan = planner.create_plan(
        "What is machine learning?",
        {"complexity": "simple", "query_type": QueryType.EXPLANATORY},
    )
    print(f"Plan: {plan.to_dict()}")

    # Test Full Research Flow
    print("\n--- Full Research Assistant ---")
    assistant = ResearchAssistant()
    result = assistant.research("Explain how LangGraph works", "thread-1")
    print(f"Phase: {result['phase']}")
    print(f"Sources: {len(result['sources'])}")
    print(f"Synthesis preview: {result['synthesis'][:200]}...")

    # Test Streaming
    print("\n--- Streaming Research ---")
    stream_assistant = StreamingResearchAssistant()
    for event in stream_assistant.stream_research("What is Python?"):
        print(f"  {event['phase']}: {event.get('message', event.get('result', ''))}")

    print("\n✅ Project solution complete!")

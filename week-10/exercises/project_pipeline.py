"""
Week 10 - Project: Research Assistant with Multi-Step Reasoning
================================================================
Build a comprehensive research assistant using LangGraph patterns.

The assistant should:
1. Parse and classify user queries
2. Create multi-step research plans
3. Execute searches and gather information
4. Evaluate source credibility
5. Synthesize findings into coherent responses
6. Track and cite sources
7. Support human review at key points

This project integrates all concepts from Week 10:
- State management
- Conditional routing
- Cycles for iterative refinement
- Tool integration
- Persistence
- Human-in-the-loop
"""

from typing import TypedDict, Annotated, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid


# =============================================================================
# PART 1: State Definitions
# =============================================================================
class ResearchPhase(Enum):
    """Phases of the research process."""

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
    """Reducer for appending lists."""
    return existing + new


class ResearchState(TypedDict):
    """
    TODO: Complete the research state definition.

    Fields needed:
    - query: str - Original user query
    - phase: str - Current research phase
    - plan: dict | None - Research plan
    - sources: list[dict] - Collected sources (use append_list reducer)
    - findings: list[str] - Key findings (use append_list reducer)
    - synthesis: str - Synthesized response
    - citations: list[str] - Citation list
    - needs_review: bool - Whether human review is needed
    - approved: bool | None - Human approval status
    - error: str | None - Any error message
    - metadata: dict - Additional metadata
    """

    pass


# =============================================================================
# PART 2: Query Analysis
# =============================================================================
class QueryType(Enum):
    """Types of research queries."""

    FACTUAL = "factual"  # Simple fact lookup
    COMPARATIVE = "comparative"  # Compare things
    EXPLANATORY = "explanatory"  # Explain concepts
    INVESTIGATIVE = "investigative"  # Deep research needed


class QueryAnalyzer:
    """
    TODO: Analyzes user queries to determine research approach.

    Methods:
    - analyze(query: str) -> dict: Analyze query type and complexity
    - extract_entities(query: str) -> list[str]: Extract key entities
    - generate_sub_questions(query: str, query_type: QueryType) -> list[str]
    """

    COMPARATIVE_KEYWORDS = ["compare", "versus", "vs", "difference", "better"]
    EXPLANATORY_KEYWORDS = ["explain", "how", "why", "what is", "describe"]
    INVESTIGATIVE_KEYWORDS = ["investigate", "research", "analyze", "deep dive"]

    def analyze(self, query: str) -> dict:
        """
        TODO: Analyze the query and return:
        - query_type: QueryType
        - complexity: "simple" | "medium" | "complex"
        - entities: list of key entities
        - sub_questions: list of sub-questions to research
        """
        pass

    def extract_entities(self, query: str) -> list[str]:
        """
        TODO: Extract key entities from the query.
        Simple approach: Extract capitalized words and quoted phrases.
        """
        pass

    def generate_sub_questions(self, query: str, query_type: QueryType) -> list[str]:
        """
        TODO: Generate sub-questions based on query type.

        For FACTUAL: 1-2 direct questions
        For COMPARATIVE: Questions about each item being compared
        For EXPLANATORY: What, why, how questions
        For INVESTIGATIVE: Multiple angles and perspectives
        """
        pass


def analyze_query_node(state: ResearchState) -> dict:
    """
    TODO: Node that analyzes the query.

    Returns updates to state including phase change to PLANNING.
    """
    pass


# =============================================================================
# PART 3: Research Planning
# =============================================================================
class ResearchPlanner:
    """
    TODO: Creates research plans based on query analysis.

    Methods:
    - create_plan(query: str, analysis: dict) -> ResearchPlan
    - estimate_complexity(analysis: dict) -> int
    """

    def create_plan(self, query: str, analysis: dict) -> ResearchPlan:
        """
        TODO: Create a research plan.

        Include:
        - Sub-questions to answer
        - Search strategies to use
        - Estimated number of steps
        """
        pass

    def estimate_complexity(self, analysis: dict) -> int:
        """
        TODO: Estimate research steps needed.

        Simple: 2-3 steps
        Medium: 4-6 steps
        Complex: 7-10 steps
        """
        pass


def create_plan_node(state: ResearchState) -> dict:
    """
    TODO: Node that creates the research plan.

    Returns plan and moves to SEARCHING phase.
    """
    pass


# =============================================================================
# PART 4: Search Execution
# =============================================================================
class SearchEngine(ABC):
    """Abstract base for search engines."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        pass


class MockSearchEngine(SearchEngine):
    """
    TODO: Mock search engine for testing.

    Returns predefined results based on query keywords.
    """

    def __init__(self):
        self.knowledge_base = {
            "python": [
                Source(
                    id="1",
                    title="Python Overview",
                    content="Python is a high-level programming language...",
                    credibility_score=0.9,
                ),
                Source(
                    id="2",
                    title="Python Applications",
                    content="Python is used in web development, AI, data science...",
                    credibility_score=0.85,
                ),
            ],
            "machine learning": [
                Source(
                    id="3",
                    title="ML Fundamentals",
                    content="Machine learning enables computers to learn from data...",
                    credibility_score=0.9,
                ),
                Source(
                    id="4",
                    title="ML vs AI",
                    content="ML is a subset of AI focused on learning algorithms...",
                    credibility_score=0.8,
                ),
            ],
            "langgraph": [
                Source(
                    id="5",
                    title="LangGraph Introduction",
                    content="LangGraph is a library for building stateful agents...",
                    credibility_score=0.95,
                )
            ],
        }

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """
        TODO: Search the mock knowledge base.

        Look for keyword matches and return relevant sources.
        """
        pass


class SearchExecutor:
    """
    TODO: Executes searches across multiple engines.

    Methods:
    - execute_searches(plan: ResearchPlan) -> list[Source]
    - deduplicate(sources: list[Source]) -> list[Source]
    """

    def __init__(self, engines: list[SearchEngine] = None):
        self.engines = engines or [MockSearchEngine()]

    def execute_searches(self, plan: ResearchPlan) -> list[Source]:
        """
        TODO: Execute all searches in the plan.

        For each sub-question, search all engines and collect results.
        """
        pass

    def deduplicate(self, sources: list[Source]) -> list[Source]:
        """
        TODO: Remove duplicate sources.
        """
        pass


def search_node(state: ResearchState) -> dict:
    """
    TODO: Node that executes searches.

    Returns sources and moves to EVALUATING phase.
    """
    pass


# =============================================================================
# PART 5: Source Evaluation
# =============================================================================
class SourceEvaluator:
    """
    TODO: Evaluates source credibility and relevance.

    Methods:
    - evaluate_credibility(source: Source) -> float
    - evaluate_relevance(source: Source, query: str) -> float
    - filter_sources(sources: list[Source], min_score: float) -> list[Source]
    """

    TRUSTED_DOMAINS = [".edu", ".gov", ".org"]

    def evaluate_credibility(self, source: Source) -> float:
        """
        TODO: Evaluate source credibility (0-1).

        Factors:
        - Domain trustworthiness
        - Content quality indicators
        - Source reputation
        """
        pass

    def evaluate_relevance(self, source: Source, query: str) -> float:
        """
        TODO: Evaluate source relevance to query (0-1).

        Factors:
        - Keyword overlap
        - Topic alignment
        - Recency
        """
        pass

    def filter_sources(
        self, sources: list[Source], min_score: float = 0.5
    ) -> list[Source]:
        """
        TODO: Filter sources below minimum score.
        """
        pass


def evaluate_sources_node(state: ResearchState) -> dict:
    """
    TODO: Node that evaluates and filters sources.

    Returns filtered sources and moves to SYNTHESIZING phase.
    """
    pass


# =============================================================================
# PART 6: Synthesis & Citation
# =============================================================================
class Synthesizer:
    """
    TODO: Synthesizes findings into coherent response.

    Methods:
    - extract_findings(sources: list[Source]) -> list[str]
    - synthesize(query: str, findings: list[str]) -> str
    - generate_citations(sources: list[Source]) -> list[str]
    """

    def extract_findings(self, sources: list[Source]) -> list[str]:
        """
        TODO: Extract key findings from sources.
        """
        pass

    def synthesize(self, query: str, findings: list[str]) -> str:
        """
        TODO: Synthesize findings into a coherent response.

        Structure:
        - Opening that addresses the query
        - Key points from findings
        - Conclusion
        """
        pass

    def generate_citations(self, sources: list[Source]) -> list[str]:
        """
        TODO: Generate citation list.

        Format: "[{id}] {title}. {url or 'No URL'}"
        """
        pass


def synthesize_node(state: ResearchState) -> dict:
    """
    TODO: Node that synthesizes response.

    Returns synthesis, citations, and moves to REVIEWING phase.
    """
    pass


# =============================================================================
# PART 7: Human Review
# =============================================================================
class HumanReviewManager:
    """
    TODO: Manages human review workflow.

    Methods:
    - needs_review(state: ResearchState) -> bool
    - request_review(state: ResearchState) -> dict
    - process_approval(state: ResearchState, approved: bool) -> dict
    """

    REVIEW_TRIGGERS = ["sensitive", "confidential", "legal", "medical", "financial"]

    def needs_review(self, state: ResearchState) -> bool:
        """
        TODO: Determine if human review is needed.

        Triggers:
        - Query contains sensitive topics
        - Low confidence in synthesis
        - Conflicting sources
        """
        pass

    def request_review(self, state: ResearchState) -> dict:
        """
        TODO: Prepare review request.

        Return summary for human reviewer.
        """
        pass

    def process_approval(self, approved: bool, feedback: str = "") -> dict:
        """
        TODO: Process human approval/rejection.
        """
        pass


def check_review_needed(state: ResearchState) -> str:
    """
    TODO: Conditional edge function.

    Returns:
    - "review" if needs_review is True
    - "complete" if no review needed or already approved
    """
    pass


def request_review_node(state: ResearchState) -> dict:
    """
    TODO: Node that requests human review.

    Pauses execution until human provides input.
    """
    pass


# =============================================================================
# PART 8: Complete Research Graph
# =============================================================================
class ResearchAssistant:
    """
    TODO: Complete research assistant using graph-based workflow.

    Graph structure:
    START -> analyze -> plan -> search -> evaluate -> synthesize -> check_review
           -> [review: request_review -> END (pause) | complete: END]

    With persistence for resuming after human review.
    """

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer or {}  # Simple dict for demo
        self.query_analyzer = QueryAnalyzer()
        self.planner = ResearchPlanner()
        self.search_executor = SearchExecutor()
        self.source_evaluator = SourceEvaluator()
        self.synthesizer = Synthesizer()
        self.review_manager = HumanReviewManager()

    def _initialize_state(self, query: str) -> ResearchState:
        """Initialize state for new research."""
        pass

    def _save_checkpoint(self, thread_id: str, state: dict) -> None:
        """Save checkpoint for thread."""
        pass

    def _load_checkpoint(self, thread_id: str) -> dict | None:
        """Load checkpoint for thread."""
        pass

    def research(self, query: str, thread_id: str = None) -> dict:
        """
        TODO: Execute research workflow.

        Steps:
        1. Initialize or load state
        2. Run through graph nodes
        3. Pause if review needed
        4. Save checkpoint
        5. Return current state
        """
        pass

    def provide_feedback(
        self, thread_id: str, approved: bool, feedback: str = ""
    ) -> dict:
        """
        TODO: Process human feedback and continue.

        Load checkpoint, update approval, and continue to completion.
        """
        pass

    def get_status(self, thread_id: str) -> dict:
        """Get current status of research thread."""
        pass


# =============================================================================
# PART 9: Streaming Support
# =============================================================================
class StreamingResearchAssistant(ResearchAssistant):
    """
    TODO: Research assistant with streaming support.

    Yields events during execution for real-time UI updates.
    """

    def stream_research(self, query: str, thread_id: str = None):
        """
        TODO: Stream research execution.

        Yields events:
        {"phase": "analyzing", "message": "Analyzing query..."}
        {"phase": "planning", "message": "Creating research plan...", "data": plan}
        {"phase": "searching", "message": "Searching sources...", "progress": 0.5}
        {"phase": "evaluating", "message": "Evaluating sources...", "sources": [...]}
        {"phase": "synthesizing", "message": "Synthesizing findings..."}
        {"phase": "complete", "result": {...}}
        """
        pass


# =============================================================================
# MAIN - Test your implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Week 10 - Project: Research Assistant with Multi-Step Reasoning")
    print("=" * 70)

    # Test Query Analysis
    print("\n--- Testing Query Analysis ---")
    # analyzer = QueryAnalyzer()
    # analysis = analyzer.analyze("Compare Python and JavaScript for web development")
    # print(f"Analysis: {analysis}")

    # Test Research Planning
    print("\n--- Testing Research Planning ---")
    # planner = ResearchPlanner()
    # plan = planner.create_plan("What is machine learning?", {"complexity": "simple"})
    # print(f"Plan: {plan.to_dict()}")

    # Test Search Execution
    print("\n--- Testing Search ---")
    # executor = SearchExecutor()
    # plan = ResearchPlan("python", ["What is Python?"], ["keyword"], 2)
    # sources = executor.execute_searches(plan)
    # print(f"Found {len(sources)} sources")

    # Test Full Research Flow
    print("\n--- Testing Full Research Assistant ---")
    # assistant = ResearchAssistant()
    # result = assistant.research("Explain how LangGraph works", "thread-1")
    # print(f"Research status: {result.get('phase')}")

    # Test Streaming
    print("\n--- Testing Streaming ---")
    # stream_assistant = StreamingResearchAssistant()
    # for event in stream_assistant.stream_research("What is Python?"):
    #     print(f"Event: {event}")

    print("\n✅ Uncomment tests as you implement each part!")
    print("\nThis project integrates:")
    print("  - State management with TypedDict")
    print("  - Multi-node graph workflows")
    print("  - Conditional routing")
    print("  - Persistence with checkpointing")
    print("  - Human-in-the-loop review")
    print("  - Streaming support")

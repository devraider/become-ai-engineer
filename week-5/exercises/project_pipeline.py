"""
Week 5 Project: AI Code Review Assistant
========================================

Build a code review tool that analyzes code and provides feedback using LLM APIs.

Before running:
    1. Create a .env file with your GOOGLE_API_KEY
    2. Run: uv add google-generativeai python-dotenv pydantic

Run this file:
    python project_pipeline.py

Run tests:
    python -m pytest tests/test_project_pipeline.py -v
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    import google.generativeai as genai
    from dotenv import load_dotenv

    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    load_dotenv = None
    GENAI_AVAILABLE = False


# =============================================================================
# SETUP
# =============================================================================


def setup_gemini():
    """Set up Gemini API client."""
    if not GENAI_AVAILABLE:
        return None

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class Severity(Enum):
    """Issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Category(Enum):
    """Issue categories."""

    STYLE = "style"
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BEST_PRACTICE = "best_practice"


@dataclass
class CodeIssue:
    """
    Represents a single code issue found during review.
    """

    severity: Severity
    category: Category
    line: Optional[int]
    message: str
    suggestion: str
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        # TODO: Implement
        pass


@dataclass
class ReviewResult:
    """
    Complete review result for a piece of code.
    """

    language: str
    issues: List[CodeIssue] = field(default_factory=list)
    summary: str = ""
    overall_score: int = 0  # 1-10

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        # TODO: Implement
        pass

    def to_json(self) -> str:
        """Convert to JSON string."""
        # TODO: Implement
        pass


# =============================================================================
# PART 1: Code Analysis
# =============================================================================


def detect_language(code: str) -> str:
    """
    TASK 1: Detect the programming language of the code.

    Args:
        code: Source code string

    Returns:
        Language name (e.g., "python", "javascript", "java")

    Hint: Look for language-specific keywords, syntax patterns, etc.
    """
    # TODO: Implement
    pass


def extract_code_structure(code: str, language: str) -> Dict:
    """
    TASK 2: Extract basic structure from code.

    Args:
        code: Source code
        language: Programming language

    Returns:
        Dictionary with:
        - 'functions': List of function names
        - 'classes': List of class names
        - 'imports': List of imports
        - 'line_count': Number of lines
    """
    # TODO: Implement
    pass


# =============================================================================
# PART 2: Prompt Engineering for Code Review
# =============================================================================


def create_review_prompt(
    code: str, language: str, focus_areas: List[str] = None
) -> str:
    """
    TASK 3: Create a prompt for code review.

    Args:
        code: Source code to review
        language: Programming language
        focus_areas: Optional specific areas to focus on

    Returns:
        Well-structured prompt for code review

    Requirements:
        - Ask for structured JSON output
        - Include severity levels
        - Request line numbers
        - Ask for suggestions
    """
    # TODO: Implement
    pass


def create_explanation_prompt(issue: CodeIssue, code: str) -> str:
    """
    TASK 4: Create a prompt to explain an issue in detail.

    Args:
        issue: The code issue to explain
        code: The original code

    Returns:
        Prompt asking for detailed explanation
    """
    # TODO: Implement
    pass


def create_fix_prompt(issue: CodeIssue, code: str) -> str:
    """
    TASK 5: Create a prompt to get a code fix suggestion.

    Args:
        issue: The code issue to fix
        code: The original code

    Returns:
        Prompt asking for corrected code
    """
    # TODO: Implement
    pass


# =============================================================================
# PART 3: Response Parsing
# =============================================================================


def parse_review_response(response_text: str) -> List[CodeIssue]:
    """
    TASK 6: Parse LLM response into structured issues.

    Args:
        response_text: Raw text response from LLM

    Returns:
        List of CodeIssue objects

    Note: Handle various response formats and edge cases
    """
    # TODO: Implement
    pass


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    TASK 7: Extract JSON from potentially messy LLM response.

    Args:
        text: Response text that may contain JSON

    Returns:
        Parsed JSON dict or None if not found
    """
    # TODO: Implement
    pass


# =============================================================================
# PART 4: Main Code Review Class
# =============================================================================


class CodeReviewAssistant:
    """
    TASK 8: Implement the main code review assistant.

    This class orchestrates the entire code review process.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the assistant.

        Args:
            model_name: LLM model to use
        """
        # TODO: Initialize
        pass

    def analyze(
        self,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
    ) -> ReviewResult:
        """
        Analyze code and return review results.

        Args:
            code: Source code to review
            language: Programming language (auto-detect if None)
            focus_areas: Specific areas to focus on

        Returns:
            ReviewResult with all issues found
        """
        # TODO: Implement
        pass

    def explain(self, issue: CodeIssue, code: str) -> str:
        """
        Get detailed explanation for an issue.

        Args:
            issue: The issue to explain
            code: Original code context

        Returns:
            Detailed explanation string
        """
        # TODO: Implement
        pass

    def suggest_fix(self, issue: CodeIssue, code: str) -> str:
        """
        Get a code fix suggestion.

        Args:
            issue: The issue to fix
            code: Original code

        Returns:
            Suggested fixed code
        """
        # TODO: Implement
        pass

    def interactive_chat(self, code: str) -> None:
        """
        Start an interactive review session.

        Args:
            code: Code to review
        """
        # TODO: Implement interactive mode
        pass


# =============================================================================
# PART 5: Utilities
# =============================================================================


def format_review_report(result: ReviewResult) -> str:
    """
    TASK 9: Format review results as a readable report.

    Args:
        result: Review result to format

    Returns:
        Formatted string report
    """
    # TODO: Implement
    pass


def compare_reviews(review1: ReviewResult, review2: ReviewResult) -> Dict:
    """
    TASK 10: Compare two reviews (e.g., before/after fix).

    Args:
        review1: First review
        review2: Second review

    Returns:
        Comparison summary
    """
    # TODO: Implement
    pass


# =============================================================================
# DEMONSTRATION
# =============================================================================

# Sample code for testing
SAMPLE_CODE_WITH_ISSUES = """
def get_user(id):
    # Get user from database
    query = f"SELECT * FROM users WHERE id = {id}"
    result = db.execute(query)
    return result

def process_data(data):
    for i in range(len(data)):
        print(data[i])
    
    x = []
    for item in data:
        x.append(item * 2)
    return x

password = "admin123"

def calculate(a, b):
    return a / b
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Project: AI Code Review Assistant")
    print("=" * 60)

    print("\n1. Testing language detection...")
    detected = detect_language(SAMPLE_CODE_WITH_ISSUES)
    if detected:
        print(f"   Detected language: {detected}")

    print("\n2. Testing code structure extraction...")
    structure = extract_code_structure(SAMPLE_CODE_WITH_ISSUES, "python")
    if structure:
        print(f"   Functions: {structure.get('functions', [])}")
        print(f"   Line count: {structure.get('line_count', 0)}")

    print("\n3. Testing review prompt creation...")
    prompt = create_review_prompt(SAMPLE_CODE_WITH_ISSUES, "python")
    if prompt:
        print(f"   Prompt preview: {prompt[:200]}...")

    print("\n4. Testing CodeReviewAssistant...")
    assistant = CodeReviewAssistant()

    # Check if API is available
    model = setup_gemini()
    if model:
        print("   Gemini API available - running full review...")
        try:
            result = assistant.analyze(SAMPLE_CODE_WITH_ISSUES)
            if result:
                print(f"\n   Review Summary: {result.summary}")
                print(f"   Issues found: {len(result.issues)}")
                for issue in result.issues[:3]:  # Show first 3
                    print(f"   - [{issue.severity.value}] {issue.message}")
        except Exception as e:
            print(f"   API error: {e}")
    else:
        print("   Gemini API not available - testing without API...")
        # Test components that don't need API

    print("\n5. Testing report formatting...")
    # Create mock result for testing
    mock_result = ReviewResult(
        language="python",
        issues=[
            CodeIssue(
                severity=Severity.ERROR,
                category=Category.SECURITY,
                line=3,
                message="SQL injection vulnerability",
                suggestion="Use parameterized queries",
            ),
            CodeIssue(
                severity=Severity.WARNING,
                category=Category.STYLE,
                line=8,
                message="Use enumerate instead of range(len())",
                suggestion="for i, item in enumerate(data):",
            ),
        ],
        summary="Found 2 issues",
        overall_score=6,
    )

    report = format_review_report(mock_result)
    if report:
        print(f"   Report:\n{report[:500]}...")

    print("\nComplete all TODOs to build the full code review assistant!")

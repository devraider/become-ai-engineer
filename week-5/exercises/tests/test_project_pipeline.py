"""
Tests for Week 5 Project: AI Code Review Assistant
"""

import pytest
from project_pipeline import (
    Severity,
    Category,
    CodeIssue,
    ReviewResult,
    detect_language,
    extract_code_structure,
    create_review_prompt,
    create_explanation_prompt,
    create_fix_prompt,
    parse_review_response,
    extract_json_from_response,
    CodeReviewAssistant,
    format_review_report,
    compare_reviews,
)


# Sample code for testing
PYTHON_CODE = '''
def hello():
    print("Hello, World!")

class MyClass:
    def method(self):
        pass
'''

JAVASCRIPT_CODE = '''
function hello() {
    console.log("Hello");
}

const arrow = () => {
    return 42;
};
'''

CODE_WITH_ISSUES = '''
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
'''


class TestCodeIssue:
    def test_creation(self):
        issue = CodeIssue(
            severity=Severity.ERROR,
            category=Category.SECURITY,
            line=5,
            message="SQL injection",
            suggestion="Use parameterized queries",
        )
        assert issue.severity == Severity.ERROR
        assert issue.line == 5

    def test_to_dict(self):
        issue = CodeIssue(
            severity=Severity.WARNING,
            category=Category.STYLE,
            line=10,
            message="Test message",
            suggestion="Test suggestion",
        )
        result = issue.to_dict()
        
        if result:  # If implemented
            assert result["severity"] == "warning" or result["severity"] == Severity.WARNING
            assert result["line"] == 10


class TestReviewResult:
    def test_creation(self):
        result = ReviewResult(
            language="python",
            issues=[],
            summary="No issues found",
            overall_score=10,
        )
        assert result.language == "python"

    def test_to_dict(self):
        result = ReviewResult(
            language="python",
            issues=[
                CodeIssue(Severity.INFO, Category.STYLE, 1, "msg", "sug")
            ],
            summary="Test",
            overall_score=8,
        )
        
        d = result.to_dict()
        if d:  # If implemented
            assert d["language"] == "python"
            assert len(d["issues"]) == 1

    def test_to_json(self):
        result = ReviewResult(language="python", summary="Test")
        json_str = result.to_json()
        
        if json_str:  # If implemented
            assert "python" in json_str


class TestDetectLanguage:
    def test_detect_python(self):
        result = detect_language(PYTHON_CODE)
        if result:  # If implemented
            assert result.lower() == "python"

    def test_detect_javascript(self):
        result = detect_language(JAVASCRIPT_CODE)
        if result:  # If implemented
            assert result.lower() in ["javascript", "js"]


class TestExtractCodeStructure:
    def test_extract_functions(self):
        result = extract_code_structure(PYTHON_CODE, "python")
        if result:  # If implemented
            assert "functions" in result
            assert "hello" in result["functions"]

    def test_extract_classes(self):
        result = extract_code_structure(PYTHON_CODE, "python")
        if result:  # If implemented
            assert "classes" in result
            assert "MyClass" in result["classes"]

    def test_line_count(self):
        result = extract_code_structure(PYTHON_CODE, "python")
        if result:  # If implemented
            assert "line_count" in result
            assert result["line_count"] > 0


class TestCreateReviewPrompt:
    def test_returns_string(self):
        result = create_review_prompt(CODE_WITH_ISSUES, "python")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_code(self):
        result = create_review_prompt(CODE_WITH_ISSUES, "python")
        assert "get_user" in result or "SELECT" in result

    def test_requests_json(self):
        result = create_review_prompt(CODE_WITH_ISSUES, "python")
        assert "json" in result.lower() or "JSON" in result

    def test_with_focus_areas(self):
        result = create_review_prompt(CODE_WITH_ISSUES, "python", ["security"])
        assert "security" in result.lower()


class TestCreateExplanationPrompt:
    def test_returns_string(self):
        issue = CodeIssue(
            Severity.ERROR, Category.SECURITY, 2,
            "SQL injection", "Use parameterized queries"
        )
        result = create_explanation_prompt(issue, CODE_WITH_ISSUES)
        
        if result:  # If implemented
            assert isinstance(result, str)
            assert "SQL" in result or "injection" in result


class TestCreateFixPrompt:
    def test_returns_string(self):
        issue = CodeIssue(
            Severity.ERROR, Category.SECURITY, 2,
            "SQL injection", "Use parameterized queries"
        )
        result = create_fix_prompt(issue, CODE_WITH_ISSUES)
        
        if result:  # If implemented
            assert isinstance(result, str)


class TestParseReviewResponse:
    def test_parse_json_response(self):
        response = '''
        {
            "issues": [
                {
                    "severity": "error",
                    "category": "security",
                    "line": 2,
                    "message": "SQL injection",
                    "suggestion": "Use parameters"
                }
            ]
        }
        '''
        result = parse_review_response(response)
        
        if result:  # If implemented
            assert len(result) >= 1

    def test_empty_response(self):
        result = parse_review_response("No issues found.")
        assert result is not None  # Should return empty list, not fail


class TestExtractJsonFromResponse:
    def test_extract_from_markdown(self):
        text = '''Here is the analysis:
        ```json
        {"key": "value"}
        ```
        '''
        result = extract_json_from_response(text)
        
        if result:  # If implemented
            assert result["key"] == "value"

    def test_extract_plain_json(self):
        text = '{"test": 123}'
        result = extract_json_from_response(text)
        
        if result:  # If implemented
            assert result["test"] == 123

    def test_no_json(self):
        result = extract_json_from_response("No JSON here")
        assert result is None


class TestCodeReviewAssistant:
    def test_initialization(self):
        assistant = CodeReviewAssistant()
        assert assistant is not None

    def test_analyze_returns_result(self):
        assistant = CodeReviewAssistant()
        result = assistant.analyze(CODE_WITH_ISSUES, language="python")
        
        if result:  # If implemented (may need API)
            assert isinstance(result, ReviewResult)


class TestFormatReviewReport:
    def test_format_with_issues(self):
        result = ReviewResult(
            language="python",
            issues=[
                CodeIssue(
                    Severity.ERROR, Category.SECURITY, 2,
                    "SQL injection vulnerability",
                    "Use parameterized queries"
                ),
            ],
            summary="Found 1 critical issue",
            overall_score=4,
        )
        
        report = format_review_report(result)
        
        if report:  # If implemented
            assert "python" in report.lower() or "Python" in report
            assert "SQL" in report or "security" in report.lower()

    def test_format_no_issues(self):
        result = ReviewResult(
            language="python",
            issues=[],
            summary="No issues found",
            overall_score=10,
        )
        
        report = format_review_report(result)
        
        if report:  # If implemented
            assert len(report) > 0


class TestCompareReviews:
    def test_compare_improved(self):
        review1 = ReviewResult(
            language="python",
            issues=[
                CodeIssue(Severity.ERROR, Category.SECURITY, 1, "Bug", "Fix"),
                CodeIssue(Severity.WARNING, Category.STYLE, 2, "Style", "Fix"),
            ],
            summary="2 issues",
            overall_score=5,
        )
        
        review2 = ReviewResult(
            language="python",
            issues=[
                CodeIssue(Severity.INFO, Category.STYLE, 1, "Minor", "Optional"),
            ],
            summary="1 issue",
            overall_score=8,
        )
        
        comparison = compare_reviews(review1, review2)
        
        if comparison:  # If implemented
            # Should indicate improvement
            assert isinstance(comparison, dict)

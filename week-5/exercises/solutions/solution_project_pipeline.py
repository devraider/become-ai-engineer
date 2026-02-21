"""
Week 5 Project: AI Code Review Assistant - SOLUTIONS
====================================================
"""

import os
import json
import re
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
    """Represents a single code issue found during review."""
    severity: Severity
    category: Category
    line: Optional[int]
    message: str
    suggestion: str
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "line": self.line,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ReviewResult:
    """Complete review result for a piece of code."""
    language: str
    issues: List[CodeIssue] = field(default_factory=list)
    summary: str = ""
    overall_score: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary,
            "overall_score": self.overall_score,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def detect_language(code: str) -> str:
    """Detect the programming language of the code."""
    indicators = {
        "python": ["def ", "import ", "print(", "class ", "elif ", "self.", "__init__"],
        "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>", "async "],
        "java": ["public class", "public static void", "System.out", "import java"],
        "go": ["func ", "package ", "import (", "fmt."],
        "rust": ["fn ", "let mut", "impl ", "pub fn", "::"],
    }
    
    code_lower = code.lower()
    scores = {lang: 0 for lang in indicators}
    
    for lang, keywords in indicators.items():
        for keyword in keywords:
            if keyword.lower() in code_lower:
                scores[lang] += 1
    
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else "unknown"


def extract_code_structure(code: str, language: str) -> Dict:
    """Extract basic structure from code."""
    lines = code.strip().split("\n")
    structure = {
        "functions": [],
        "classes": [],
        "imports": [],
        "line_count": len(lines),
    }
    
    if language == "python":
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                match = re.match(r"def\s+(\w+)", stripped)
                if match:
                    structure["functions"].append(match.group(1))
            elif stripped.startswith("class "):
                match = re.match(r"class\s+(\w+)", stripped)
                if match:
                    structure["classes"].append(match.group(1))
            elif stripped.startswith("import ") or stripped.startswith("from "):
                structure["imports"].append(stripped)
    
    return structure


def create_review_prompt(code: str, language: str, focus_areas: List[str] = None) -> str:
    """Create a prompt for code review."""
    focus = ""
    if focus_areas:
        focus = f"\nFocus especially on: {', '.join(focus_areas)}"
    
    return f"""You are an expert code reviewer. Review the following {language} code.
{focus}

Analyze the code for:
1. Style and formatting issues
2. Potential bugs
3. Security vulnerabilities
4. Performance problems
5. Best practice violations

CODE TO REVIEW:
```{language}
{code}
```

Respond with a JSON object in this exact format:
{{
    "issues": [
        {{
            "severity": "error" | "warning" | "info",
            "category": "style" | "bug" | "security" | "performance" | "best_practice",
            "line": <line_number or null>,
            "message": "<description of the issue>",
            "suggestion": "<how to fix it>"
        }}
    ],
    "summary": "<brief summary of the review>",
    "overall_score": <1-10>
}}

Return ONLY valid JSON, no additional text."""


def create_explanation_prompt(issue: CodeIssue, code: str) -> str:
    """Create a prompt to explain an issue in detail."""
    return f"""Explain the following code issue in detail:

Issue: {issue.message}
Category: {issue.category.value}
Severity: {issue.severity.value}
Line: {issue.line}
Suggestion: {issue.suggestion}

Code context:
```
{code}
```

Please explain:
1. Why this is a problem
2. What could go wrong if not fixed
3. The correct way to handle this
4. Any relevant best practices"""


def create_fix_prompt(issue: CodeIssue, code: str) -> str:
    """Create a prompt to get a code fix suggestion."""
    return f"""Fix the following issue in the code:

Issue: {issue.message}
Suggestion: {issue.suggestion}

Original code:
```
{code}
```

Please provide the corrected code with:
1. The fix applied
2. Comments explaining the changes
3. Any additional improvements that should be made"""


def parse_review_response(response_text: str) -> List[CodeIssue]:
    """Parse LLM response into structured issues."""
    issues = []
    
    json_data = extract_json_from_response(response_text)
    if not json_data:
        return issues
    
    for item in json_data.get("issues", []):
        try:
            severity = Severity(item.get("severity", "info"))
            category = Category(item.get("category", "style"))
            
            issue = CodeIssue(
                severity=severity,
                category=category,
                line=item.get("line"),
                message=item.get("message", ""),
                suggestion=item.get("suggestion", ""),
            )
            issues.append(issue)
        except (ValueError, KeyError):
            continue
    
    return issues


def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from potentially messy LLM response."""
    # Try to find JSON in code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"(\{[\s\S]*\})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class CodeReviewAssistant:
    """Main code review assistant."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = setup_gemini()
        self.chat_history = []
    
    def analyze(
        self,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
    ) -> ReviewResult:
        """Analyze code and return review results."""
        if language is None:
            language = detect_language(code)
        
        result = ReviewResult(language=language)
        
        if self.model:
            prompt = create_review_prompt(code, language, focus_areas)
            
            try:
                response = self.model.generate_content(prompt)
                json_data = extract_json_from_response(response.text)
                
                if json_data:
                    result.issues = parse_review_response(response.text)
                    result.summary = json_data.get("summary", "")
                    result.overall_score = json_data.get("overall_score", 5)
            except Exception as e:
                result.summary = f"Error during analysis: {e}"
        else:
            result.summary = "API not available - mock analysis"
        
        return result
    
    def explain(self, issue: CodeIssue, code: str) -> str:
        """Get detailed explanation for an issue."""
        if not self.model:
            return "API not available"
        
        prompt = create_explanation_prompt(issue, code)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def suggest_fix(self, issue: CodeIssue, code: str) -> str:
        """Get a code fix suggestion."""
        if not self.model:
            return "API not available"
        
        prompt = create_fix_prompt(issue, code)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def interactive_chat(self, code: str) -> None:
        """Start an interactive review session."""
        print("Starting interactive code review session...")
        print("Type 'exit' to end the session.\n")
        
        result = self.analyze(code)
        print(format_review_report(result))
        
        while True:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() == "exit":
                break
            
            if self.model:
                prompt = f"""Based on this code review, answer the following question:

Code:
```
{code}
```

Review Summary: {result.summary}

Question: {user_input}"""
                
                try:
                    response = self.model.generate_content(prompt)
                    print(f"\nAssistant: {response.text}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("API not available")


def format_review_report(result: ReviewResult) -> str:
    """Format review results as a readable report."""
    lines = [
        "=" * 50,
        "CODE REVIEW REPORT",
        "=" * 50,
        f"\nLanguage: {result.language}",
        f"Overall Score: {result.overall_score}/10",
        f"\nSummary: {result.summary}",
        f"\nIssues Found: {len(result.issues)}",
    ]
    
    if result.issues:
        lines.append("\n" + "-" * 30)
        lines.append("ISSUES:")
        lines.append("-" * 30)
        
        for i, issue in enumerate(result.issues, 1):
            lines.append(f"\n{i}. [{issue.severity.value.upper()}] {issue.category.value}")
            if issue.line:
                lines.append(f"   Line: {issue.line}")
            lines.append(f"   Message: {issue.message}")
            lines.append(f"   Suggestion: {issue.suggestion}")
    
    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def compare_reviews(review1: ReviewResult, review2: ReviewResult) -> Dict:
    """Compare two reviews (e.g., before/after fix)."""
    return {
        "score_change": review2.overall_score - review1.overall_score,
        "issues_before": len(review1.issues),
        "issues_after": len(review2.issues),
        "issues_fixed": len(review1.issues) - len(review2.issues),
        "improved": review2.overall_score > review1.overall_score,
        "categories_improved": list(set(
            i.category.value for i in review1.issues
        ) - set(
            i.category.value for i in review2.issues
        )),
    }

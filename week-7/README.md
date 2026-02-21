# Week 7 - Advanced Prompt Engineering

> **Master the art of communicating with LLMs to get exactly what you need**

This week focuses on advanced prompt engineering techniques that separate beginners from professionals. You'll learn patterns used by top AI engineers to get consistent, high-quality outputs from any LLM.

---

## References

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Google Prompt Engineering](https://ai.google.dev/docs/prompt_best_practices)
- [Prompting Guide](https://www.promptingguide.ai/)

---

## Installation

```bash
cd week-7

# We'll continue using Gemini (free tier)
uv add google-generativeai python-dotenv

# For structured outputs
uv add pydantic instructor
```

Create `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## Concepts

### 1. The Prompt Engineering Mindset 🧠

Prompt engineering is not about "magic words" - it's about **clear communication**:

```python
# ❌ BAD: Vague, ambiguous
prompt = "Write something about Python"

# ✅ GOOD: Specific, structured, contextual
prompt = """You are a senior Python developer writing documentation.

Task: Explain Python list comprehensions for intermediate developers.

Requirements:
- Start with a one-sentence definition
- Show 3 progressive examples (simple → complex)
- Include common pitfalls to avoid
- Keep total response under 200 words

Format: Use markdown with code blocks."""
```

**Key Principle**: If a human would need clarification, so will the LLM.

---

### 2. The CRAFT Framework

Use **CRAFT** to structure any prompt:

| Letter | Meaning | Purpose                       |
| ------ | ------- | ----------------------------- |
| **C**  | Context | Background info the LLM needs |
| **R**  | Role    | Who the LLM should act as     |
| **A**  | Action  | Specific task to accomplish   |
| **F**  | Format  | How to structure the output   |
| **T**  | Tone    | Communication style           |

```python
def craft_prompt(context: str, role: str, action: str, format: str, tone: str) -> str:
    """Build a CRAFT-structured prompt."""
    return f"""Context: {context}

Role: {role}

Task: {action}

Output Format: {format}

Tone: {tone}"""


# Example usage
prompt = craft_prompt(
    context="I'm building a REST API for an e-commerce platform",
    role="You are a senior backend engineer with 10 years of experience",
    action="Review this endpoint design and suggest improvements",
    format="Numbered list of suggestions with code examples",
    tone="Direct and technical, focus on practical improvements"
)
```

**Exercise 1** - See `exercises/exercise_basic_1_fundamentals.py`

---

### 3. Chain-of-Thought (CoT) Prompting 🔗

Make the LLM "show its work" for better reasoning:

```python
# ❌ Without CoT - might give wrong answer
prompt = "Is 17 * 24 greater than 400?"

# ✅ With CoT - forces step-by-step reasoning
prompt = """Is 17 * 24 greater than 400?

Think through this step by step:
1. First, calculate 17 * 24
2. Then, compare the result to 400
3. Finally, state your conclusion

Show your work."""
```

**When to use CoT:**

- Math problems
- Logic puzzles
- Multi-step tasks
- Complex comparisons
- Code debugging

**CoT Variations:**

```python
# Zero-shot CoT (simplest)
prompt = "Solve this problem. Think step by step."

# Few-shot CoT (with examples)
prompt = """Example:
Q: Is 15 * 20 greater than 250?
A: Let me work through this:
   1. Calculate 15 * 20 = 300
   2. Compare: 300 > 250
   3. Answer: Yes, 15 * 20 is greater than 250

Now solve:
Q: Is 17 * 24 greater than 400?
A:"""

# Structured CoT
prompt = """Analyze this code for bugs.

For each potential issue:
- OBSERVE: What do you see?
- REASON: Why might this be problematic?
- CONCLUDE: Is it a bug? How severe?
- FIX: Suggest a correction."""
```

---

### 4. Few-Shot Learning Patterns 📚

Teach by example - the most powerful technique:

```python
def create_few_shot_prompt(examples: list[dict], query: str) -> str:
    """Create a few-shot prompt from examples."""
    prompt_parts = ["Learn from these examples:\n"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['output']}")
        prompt_parts.append("")

    prompt_parts.append("Now process this:")
    prompt_parts.append(f"Input: {query}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)


# Example: Sentiment classification
examples = [
    {"input": "This product is amazing!", "output": "positive"},
    {"input": "Terrible experience, never again", "output": "negative"},
    {"input": "It's okay, nothing special", "output": "neutral"},
]

prompt = create_few_shot_prompt(examples, "Best purchase I've ever made!")
```

**Best Practices for Few-Shot:**

- Use 3-5 examples (more isn't always better)
- Cover edge cases in examples
- Keep examples consistent in format
- Order matters - put best examples last

**Exercise 2** - See `exercises/exercise_intermediate_2_techniques.py`

---

### 5. Output Structuring & JSON Mode 📋

Force consistent, parseable outputs:

```python
import google.generativeai as genai
import json

def get_structured_response(prompt: str, schema: dict) -> dict:
    """Get structured JSON response from LLM."""

    structured_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

No explanations, no markdown, just the JSON object."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(structured_prompt)

    # Parse and validate
    return json.loads(response.text)


# Example usage
schema = {
    "sentiment": "positive | negative | neutral",
    "confidence": "float between 0 and 1",
    "key_phrases": ["list", "of", "phrases"],
    "summary": "one sentence summary"
}

result = get_structured_response(
    "Analyze: 'This product exceeded my expectations!'",
    schema
)
```

**Using Pydantic for Validation:**

```python
from pydantic import BaseModel, Field
from typing import Literal
import json

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    key_phrases: list[str]
    summary: str

def analyze_with_validation(text: str) -> SentimentAnalysis:
    """Get validated structured output."""
    prompt = f"""Analyze the sentiment of this text:
"{text}"

Respond with JSON:
{{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": float 0-1,
    "key_phrases": ["phrase1", "phrase2"],
    "summary": "one sentence"
}}"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Validate with Pydantic
    data = json.loads(response.text)
    return SentimentAnalysis(**data)
```

---

### 6. Self-Consistency & Verification 🔍

Make outputs more reliable by asking the LLM to check itself:

```python
def generate_with_verification(task: str) -> dict:
    """Generate content then verify it."""

    # Step 1: Generate
    generation_prompt = f"""Task: {task}

Provide your response."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    initial = model.generate_content(generation_prompt).text

    # Step 2: Verify
    verification_prompt = f"""You generated this response:

---
{initial}
---

Now verify your work:
1. Check for factual errors
2. Check for logical inconsistencies
3. Check if it fully addresses the task: "{task}"

Respond with JSON:
{{
    "issues_found": ["list of issues or empty"],
    "confidence_score": 0.0-1.0,
    "revised_response": "corrected version if needed, or 'NO_CHANGES'"
}}"""

    verification = model.generate_content(verification_prompt).text
    return json.loads(verification)
```

**Multi-Path Verification:**

```python
def consensus_response(prompt: str, num_samples: int = 3) -> str:
    """Generate multiple responses and find consensus."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Generate multiple responses
    responses = []
    for _ in range(num_samples):
        response = model.generate_content(prompt)
        responses.append(response.text)

    # Ask for consensus
    consensus_prompt = f"""I asked: "{prompt}"

I got these {num_samples} different responses:

{chr(10).join(f'Response {i+1}: {r}' for i, r in enumerate(responses))}

Analyze these responses and provide:
1. The most accurate/common answer
2. Confidence level (how much they agree)
3. Any important differences to note"""

    return model.generate_content(consensus_prompt).text
```

---

### 7. Prompt Templates & Reusability 🎯

Build a library of proven prompts:

````python
from string import Template
from dataclasses import dataclass
from typing import Optional

@dataclass
class PromptTemplate:
    """Reusable prompt template."""
    name: str
    template: str
    description: str
    required_vars: list[str]

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        # Check required vars
        missing = set(self.required_vars) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return Template(self.template).safe_substitute(**kwargs)


# Create a template library
TEMPLATES = {
    "code_review": PromptTemplate(
        name="code_review",
        template="""You are a senior $language developer reviewing code.

Code to review:
```$language
$code
````

Focus on:

- Bugs and potential errors
- Performance issues
- Best practices
- Security concerns

Provide specific, actionable feedback with code examples.""",
description="Reviews code for bugs, performance, and best practices",
required_vars=["language", "code"]
),

    "explain_concept": PromptTemplate(
        name="explain_concept",
        template="""Explain $concept to someone with $experience_level experience.

Requirements:

- Use analogies from $domain if helpful
- Include a practical example
- Mention common misconceptions
- Keep it under $word_limit words""",
  description="Explains technical concepts at appropriate level",
  required_vars=["concept", "experience_level", "domain", "word_limit"]
  ),
  "debug_error": PromptTemplate(
  name="debug_error",
  template="""I'm getting this error in my $language code:

Error message:

```
$error_message
```

Relevant code:

```$language
$code
```

Help me:

1. Understand what's causing this error
2. Fix the code
3. Prevent similar errors in the future""",
   description="Helps debug code errors",
   required_vars=["language", "error_message", "code"]
   ),
   }

# Usage

prompt = TEMPLATES["code_review"].render(
language="python",
code="""
def calculate_average(numbers):
return sum(numbers) / len(numbers)
"""
)

````

**Exercise 3** - See `exercises/exercise_advanced_3_systems.py`

---

### 8. Meta-Prompting: Prompts That Generate Prompts 🔄

Use LLMs to improve your prompts:

```python
def improve_prompt(original_prompt: str, goal: str) -> str:
    """Use LLM to improve a prompt."""

    meta_prompt = f"""You are a prompt engineering expert.

I have this prompt:
---
{original_prompt}
---

Goal: {goal}

Improve this prompt by:
1. Making it more specific and unambiguous
2. Adding relevant context
3. Specifying the desired output format
4. Including examples if helpful

Provide ONLY the improved prompt, no explanations."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(meta_prompt).text


def generate_prompt(task_description: str) -> str:
    """Generate a prompt from a task description."""

    meta_prompt = f"""You are a prompt engineering expert.

Task: Create an effective prompt for this goal:
"{task_description}"

Requirements for the prompt you create:
- Clear role/persona for the AI
- Specific, actionable instructions
- Defined output format
- Appropriate constraints (length, style, etc.)

Return ONLY the prompt itself, ready to use."""

    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.generate_content(meta_prompt).text
````

---

## Weekly Project: Prompt Engineering Toolkit 🛠️

Build a comprehensive prompt engineering toolkit that you can use in real projects.

**Your toolkit will include:**

1. CRAFT prompt builder
2. Few-shot example manager
3. Chain-of-thought wrapper
4. Output validator with Pydantic
5. Prompt template library
6. Meta-prompt optimizer

See `exercises/project_pipeline.py` for the full project.

---

## Interview Questions

Be ready to answer these in AI engineering interviews:

1. **What is prompt engineering and why does it matter?**
   - The practice of designing inputs to get optimal outputs from LLMs
   - Matters because it can improve output quality 10x without additional cost
   - Key skill for anyone working with LLMs in production

2. **Explain Chain-of-Thought prompting and when to use it.**
   - Technique where you ask the LLM to "show its work"
   - Forces step-by-step reasoning, reducing errors
   - Best for: math, logic, multi-step tasks, debugging

3. **What's the difference between zero-shot and few-shot prompting?**
   - Zero-shot: Ask directly without examples
   - Few-shot: Provide examples to learn from
   - Few-shot typically more accurate but uses more tokens

4. **How do you ensure consistent, parseable output from LLMs?**
   - Specify exact format (JSON schema)
   - Use structured output modes when available
   - Validate with Pydantic or similar
   - Add format instructions and examples

5. **What makes a good prompt template?**
   - Clear variables with defaults
   - Includes role, context, task, format
   - Tested across different inputs
   - Documented with examples

6. **How do you debug a prompt that gives inconsistent results?**
   - Add more specific constraints
   - Include examples (few-shot)
   - Use chain-of-thought
   - Try temperature=0 for consistency
   - Break into smaller steps

---

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Structure prompts using the CRAFT framework
- [ ] Implement Chain-of-Thought prompting
- [ ] Create effective few-shot examples
- [ ] Get consistent JSON outputs from LLMs
- [ ] Build reusable prompt templates
- [ ] Use meta-prompting to improve prompts
- [ ] Debug and optimize underperforming prompts

---

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 8](../week-8/README.md)**

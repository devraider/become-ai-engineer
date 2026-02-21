# Week 5 - LLM APIs & Prompt Engineering

Master working with Large Language Models through APIs. Learn prompt engineering, structured outputs, and building AI applications.

## References

- [Google AI Studio](https://aistudio.google.com/) - Free Gemini API dashboard
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI Cookbook](https://cookbook.openai.com/) - Patterns apply to other LLMs

> 💡 **Note on API Costs**: This course uses **Google's Gemini API** which offers a **free tier** with generous limits. OpenAI, Anthropic, and other providers require paid API keys with per-token billing. The patterns you learn with Gemini transfer directly to other providers.

## Installation

```bash
# Navigate to main project folder
cd become-ai-engineer

# Install dependencies, it uses uv to install the packages
uv add google-generativeai python-dotenv pydantic requests

# Optional: For OpenAI integration (requires paid API key)
# uv add openai
```

### API Key Setup

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Create API Key"
4. Create a `.env` file in your project root:

```bash
# .env file (never commit this!)
GOOGLE_API_KEY=your_api_key_here

# Optional: For other providers
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Concepts

### 1. Introduction to LLM APIs 🌐

LLM APIs provide access to powerful language models without managing infrastructure:

```python
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# Generate a simple response
response = model.generate_content("Explain machine learning in one sentence.")
print(response.text)
```

**Why Use APIs Instead of Local Models?**

- 🚀 State-of-the-art models without GPU requirements
- 📈 Auto-scaling - handles any traffic load
- 🔄 Always up-to-date models
- 💰 Pay only for what you use (or free with Gemini!)

### 2. Understanding Tokens & Pricing

LLMs process text as **tokens** (roughly 4 characters or ¾ of a word):

```python
# Count tokens before making expensive calls
model = genai.GenerativeModel("gemini-1.5-flash")

# Count tokens
token_count = model.count_tokens("Hello, how are you today?")
print(f"Token count: {token_count.total_tokens}")

# Longer text = more tokens
long_text = "This is a much longer piece of text " * 100
token_count = model.count_tokens(long_text)
print(f"Long text tokens: {token_count.total_tokens}")
```

**Gemini Free Tier Limits (as of 2024):**

- 15 requests per minute
- 1 million tokens per day
- 1,500 requests per day

> Compare to OpenAI GPT-4: ~$0.03/1K input tokens, $0.06/1K output tokens

### 3. Prompt Engineering Fundamentals

The quality of your output depends heavily on your prompt:

```python
# ❌ Bad prompt - vague and unstructured
bad_prompt = "Tell me about Python"

# ✅ Good prompt - specific and structured
good_prompt = """
You are an expert Python instructor.

Task: Explain Python list comprehensions.

Requirements:
1. Start with a one-sentence definition
2. Provide 3 examples from simple to complex
3. Explain when to use vs regular loops
4. Keep total response under 200 words

Format: Use markdown with code blocks.
"""

response = model.generate_content(good_prompt)
print(response.text)
```

**Key Prompt Engineering Techniques:**

- **Role Setting**: "You are a [expert role]..."
- **Task Specification**: Clear, specific instructions
- **Format Control**: Specify output structure
- **Constraints**: Length limits, style guidelines
- **Examples**: Show desired input/output pairs

📝 **Exercise**: [exercise_basic_1_prompts.py](exercises/exercise_basic_1_prompts.py)

### 4. System Instructions & Context

Set consistent behavior across conversations:

```python
# Configure model with system instructions
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="""
    You are a helpful coding assistant specialized in Python.

    Guidelines:
    - Always provide working code examples
    - Explain your code with comments
    - Suggest best practices
    - If you're unsure, say so
    - Keep responses concise
    """
)

# Now all responses follow these guidelines
response = model.generate_content("How do I read a CSV file?")
print(response.text)
```

### 5. Multi-turn Conversations (Chat)

Maintain context across multiple messages:

```python
model = genai.GenerativeModel("gemini-1.5-flash")

# Start a chat session
chat = model.start_chat(history=[])

# First message
response1 = chat.send_message("My name is Alex and I'm learning Python.")
print("Bot:", response1.text)

# Follow-up message (model remembers context)
response2 = chat.send_message("What should I learn first?")
print("Bot:", response2.text)

# The model remembers your name!
response3 = chat.send_message("Can you remind me what my name is?")
print("Bot:", response3.text)

# View conversation history
for message in chat.history:
    print(f"{message.role}: {message.parts[0].text[:50]}...")
```

📝 **Exercise**: [exercise_intermediate_2_chat.py](exercises/exercise_intermediate_2_chat.py)

### 6. Structured Outputs with JSON

Get predictable, parseable responses:

````python
import json

prompt = """
Analyze this product review and return a JSON object.

Review: "This laptop is amazing! Fast processor, great battery life,
but the keyboard is a bit mushy. Worth every penny of the $999 price."

Return ONLY valid JSON with this structure:
{
    "sentiment": "positive" or "negative" or "mixed",
    "score": 1-10,
    "pros": ["list", "of", "pros"],
    "cons": ["list", "of", "cons"],
    "price_mentioned": number or null,
    "would_recommend": boolean
}
"""

response = model.generate_content(prompt)

# Parse the JSON response
try:
    # Clean up response (remove markdown code blocks if present)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    result = json.loads(text)
    print("Sentiment:", result["sentiment"])
    print("Pros:", result["pros"])
except json.JSONDecodeError:
    print("Failed to parse JSON:", response.text)
````

### 7. Advanced: Function Calling

Let the LLM decide when to call your functions:

```python
# Define tools/functions the model can use
def get_weather(location: str) -> dict:
    """Simulated weather API"""
    # In real app, call actual weather API
    return {"location": location, "temp": 72, "condition": "sunny"}

def search_web(query: str) -> list:
    """Simulated web search"""
    return [f"Result 1 for {query}", f"Result 2 for {query}"]

# Configure model with function declarations
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[get_weather, search_web]
)

chat = model.start_chat()

# The model will call functions when appropriate
response = chat.send_message("What's the weather in San Francisco?")

# Check if model wants to call a function
for part in response.parts:
    if fn := part.function_call:
        print(f"Function called: {fn.name}")
        print(f"Arguments: {fn.args}")

        # Execute the function
        if fn.name == "get_weather":
            result = get_weather(**fn.args)
            # Send result back to model
            response = chat.send_message(str(result))
            print("Final response:", response.text)
```

📝 **Exercise**: [exercise_advanced_3_tools.py](exercises/exercise_advanced_3_tools.py)

### 8. Error Handling & Best Practices

Production-ready LLM integration:

```python
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import time

def robust_generate(prompt: str, max_retries: int = 3) -> str:
    """Generate content with retry logic and error handling."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)

            # Check for blocked content
            if response.prompt_feedback.block_reason:
                raise ValueError(f"Prompt blocked: {response.prompt_feedback}")

            return response.text

        except google_exceptions.ResourceExhausted:
            # Rate limited - wait and retry
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except google_exceptions.InvalidArgument as e:
            # Bad request - don't retry
            raise ValueError(f"Invalid request: {e}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Error: {e}. Retrying...")
            time.sleep(1)

    raise RuntimeError("Max retries exceeded")

# Usage
try:
    response = robust_generate("Explain quantum computing")
    print(response)
except ValueError as e:
    print(f"Request error: {e}")
except RuntimeError as e:
    print(f"Service unavailable: {e}")
```

**Best Practices Checklist:**

- ✅ Never hardcode API keys
- ✅ Implement retry logic with exponential backoff
- ✅ Handle rate limits gracefully
- ✅ Validate and sanitize outputs
- ✅ Log requests for debugging
- ✅ Set reasonable timeouts
- ✅ Cache responses when appropriate

### 9. Using Other Providers (Paid)

The patterns you learn apply to other providers:

```python
# OpenAI (requires paid API key)
# from openai import OpenAI
# client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# print(response.choices[0].message.content)

# Anthropic Claude (requires paid API key)
# from anthropic import Anthropic
# client = Anthropic()
# response = client.messages.create(
#     model="claude-3-sonnet-20240229",
#     max_tokens=1024,
#     messages=[{"role": "user", "content": "Hello!"}]
# )
# print(response.content[0].text)

# The core patterns are the same:
# 1. Configure client with API key
# 2. Specify model
# 3. Send messages/prompts
# 4. Process response
```

## Weekly Project: AI Code Review Assistant 🤖

Build a code review tool that analyzes code and provides feedback:

**Project Requirements:**

1. **Code Analysis**
   - Accept code input (string or file)
   - Identify the programming language
   - Analyze for common issues

2. **Review Categories**
   - Code style and formatting
   - Potential bugs
   - Security vulnerabilities
   - Performance suggestions
   - Best practice recommendations

3. **Structured Output**
   - Return JSON with categorized feedback
   - Include severity levels (info, warning, error)
   - Provide line-specific comments where possible

4. **Interactive Mode**
   - Allow follow-up questions about feedback
   - Explain suggestions in detail
   - Suggest fixes

```python
# Example usage of your completed project:
reviewer = CodeReviewAssistant()

code = '''
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
'''

review = reviewer.analyze(code, language="python")
print(review)
# {
#   "language": "python",
#   "issues": [
#     {
#       "severity": "error",
#       "category": "security",
#       "line": 2,
#       "message": "SQL injection vulnerability",
#       "suggestion": "Use parameterized queries"
#     }
#   ],
#   "summary": "Found 1 critical security issue"
# }

# Ask follow-up
response = reviewer.explain("How do I fix the SQL injection?")
```

📝 **Project**: [project_pipeline.py](exercises/project_pipeline.py)

## Interview Questions

### Basic Level

1. **What is a token in the context of LLMs? Why does token count matter?**
2. **Explain the difference between zero-shot and few-shot prompting.**
3. **What is prompt injection and how do you prevent it?**

### Intermediate Level

4. **How would you implement conversation memory in a chatbot with limited context window?**
5. **Describe strategies for getting consistent structured (JSON) outputs from LLMs.**
6. **What is temperature in LLM APIs? When would you use high vs low temperature?**

### Advanced Level

7. **Design a system that uses function calling to let an LLM interact with external APIs. What are the security considerations?**
8. **How would you implement RAG (Retrieval Augmented Generation) to give an LLM access to private documents?**
9. **Compare different strategies for reducing LLM API costs while maintaining quality.**

## Takeaway Checklist

After completing this week, you should be able to:

- [ ] Set up and authenticate with LLM APIs (Gemini, OpenAI)
- [ ] Write effective prompts with proper structure
- [ ] Implement multi-turn conversations with context
- [ ] Extract structured data (JSON) from LLM responses
- [ ] Handle errors and rate limits gracefully
- [ ] Use function calling for tool integration
- [ ] Apply prompt engineering best practices
- [ ] Build production-ready LLM applications

**[→ View Full Roadmap](../ROADMAP.md)** | **[→ Begin Week 6](../week-6/README.md)**

# Python Basics for AI Engineering

> **Skip this if you already know Python.** This is a quick reference for complete beginners.

---

## Why Python for AI?

- Most AI/ML libraries (PyTorch, TensorFlow, HuggingFace) are Python-first
- Easy to read and write
- Huge ecosystem and community
- Fast prototyping (interpret, not compile)

---

## 1. Variables and Types

```python
# Basic types
name = "Alice"          # str (string)
age = 25                # int (integer)
score = 95.5            # float (decimal)
is_student = True       # bool (boolean)

# Type checking
print(type(name))       # <class 'str'>
print(type(age))        # <class 'int'>
```

---

## 2. Collections

### Lists (ordered, mutable)

```python
# Most common - you'll use these constantly
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]

# Access by index (0-based)
print(fruits[0])        # "apple"
print(fruits[-1])       # "cherry" (last item)

# Common operations
fruits.append("orange") # Add item
fruits.remove("banana") # Remove item
len(fruits)             # Count items

# List comprehension - VERY common in AI code
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
even = [x for x in numbers if x % 2 == 0]  # [2, 4]
```

### Dictionaries (key-value pairs)

```python
# Configuration and data storage
config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Access values
print(config["model"])          # "gpt-4"
print(config.get("missing", 0)) # 0 (default if key missing)

# Add/update
config["top_p"] = 0.9

# Loop through
for key, value in config.items():
    print(f"{key}: {value}")
```

### Tuples (ordered, immutable)

```python
# Fixed data that shouldn't change
point = (10, 20)
rgb = (255, 128, 0)

x, y = point  # Unpacking
```

---

## 3. Control Flow

### If/Else

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"
```

### For Loops

```python
# Loop through list
for fruit in fruits:
    print(fruit)

# Loop with index
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Loop through range
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)
```

### While Loops

```python
count = 0
while count < 5:
    print(count)
    count += 1
```

---

## 4. Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))  # "Hello, Alice!"

# With default parameter
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# With type hints (common in modern Python)
def add(a: int, b: int) -> int:
    return a + b

# Multiple returns
def get_stats(numbers: list) -> tuple:
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])
```

---

## 5. Classes (Brief)

You'll see classes everywhere in AI libraries:

```python
class TextProcessor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.processed_count = 0

    def process(self, text: str) -> str:
        self.processed_count += 1
        return text.lower().strip()

# Usage
processor = TextProcessor("bert-base")
result = processor.process("  Hello World  ")
print(result)  # "hello world"
print(processor.processed_count)  # 1
```

---

## 6. String Operations (Used Constantly in NLP)

```python
text = "  Hello, World!  "

# Basic operations
text.lower()        # "  hello, world!  "
text.upper()        # "  HELLO, WORLD!  "
text.strip()        # "Hello, World!"

# Splitting and joining
words = text.split(",")  # ["  Hello", " World!  "]
joined = " ".join(["a", "b", "c"])  # "a b c"

# Formatting (f-strings) - use this!
name = "Alice"
age = 25
message = f"{name} is {age} years old"

# Check content
text.startswith("Hello")  # True (after strip)
text.endswith("!")        # True (after strip)
"World" in text           # True
```

---

## 7. File Operations

```python
# Read file
with open("data.txt", "r") as f:
    content = f.read()
    # or
    lines = f.readlines()

# Write file
with open("output.txt", "w") as f:
    f.write("Hello, World!")

# JSON (you'll use this a lot)
import json

# Read JSON
with open("config.json", "r") as f:
    data = json.load(f)

# Write JSON
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)
```

---

## 8. Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("This always runs")

# Common in API calls
try:
    response = call_api(data)
except TimeoutError:
    print("Request timed out, retrying...")
except Exception as e:
    print(f"API error: {e}")
```

---

## 9. Imports and Modules

```python
# Import entire module
import numpy as np
import pandas as pd

# Import specific items
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

# Import with alias
import matplotlib.pyplot as plt
```

---

## 10. Common Patterns in AI Code

### Type Hints

```python
from typing import List, Dict, Optional, Tuple

def process_batch(
    texts: List[str],
    config: Dict[str, any],
    max_length: Optional[int] = None
) -> List[Dict]:
    pass
```

### Dataclasses (Modern Python)

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000

config = ModelConfig("gpt-4", temperature=0.5)
```

### Context Managers

```python
# Automatically cleanup resources
with open("file.txt") as f:
    data = f.read()
# File automatically closed after this block
```

---

## ✅ Quick Reference Card

```python
# Lists
items = [1, 2, 3]
items.append(4)
items[0]  # First
items[-1]  # Last

# Dict
d = {"key": "value"}
d.get("key", "default")
d.items(), d.keys(), d.values()

# String
s = "hello"
s.upper(), s.lower(), s.strip()
s.split(" "), " ".join(list)

# Loop
for item in items:
for i, item in enumerate(items):
for k, v in dict.items():

# Comprehension
[x*2 for x in items]
{k: v*2 for k, v in d.items()}

# Function
def func(arg: type) -> return_type:
    return result
```

---

## 📚 Resources to Learn More

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/) - Great tutorials
- [Python for Everybody](https://www.py4e.com/) - Free course

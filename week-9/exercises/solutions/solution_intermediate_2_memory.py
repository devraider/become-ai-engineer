"""
Solutions for Week 9 - Exercise Intermediate 2: Memory Systems
==============================================================
"""

from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import os


@dataclass
class Message:
    """Represents a chat message."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def add_user_message(self, content: str) -> None:
        self.add_message("human", content)

    def add_ai_message(self, content: str) -> None:
        self.add_message("ai", content)

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.add_user_message(inputs.get("input", ""))
        self.add_ai_message(outputs.get("output", ""))


class ConversationBufferMemory(BaseMemory):
    """Stores all messages without any limit."""

    def __init__(self, return_messages: bool = True):
        self.return_messages = return_messages
        self.messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        if self.return_messages:
            return {"history": self.get_messages()}
        return {"history": self._format_messages_as_string()}

    def _format_messages_as_string(self) -> str:
        lines = []
        for msg in self.messages:
            role_name = "Human" if msg.role == "human" else "AI"
            lines.append(f"{role_name}: {msg.content}")
        return "\n".join(lines)


class ConversationBufferWindowMemory(BaseMemory):
    """Keeps only the last k conversation turns."""

    def __init__(self, k: int = 5, return_messages: bool = True):
        self.k = k
        self.return_messages = return_messages
        self.messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> list[Message]:
        return list(self.messages[-self.k * 2 :])

    def clear(self) -> None:
        self.messages.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        if self.return_messages:
            return {"history": self.get_messages()}
        lines = [f"{m.role}: {m.content}" for m in self.get_messages()]
        return {"history": "\n".join(lines)}


class ConversationSummaryMemory(BaseMemory):
    """Maintains a running summary of the conversation."""

    def __init__(self, summarizer=None, max_messages_before_summary: int = 6):
        self.summarizer = summarizer
        self.max_messages_before_summary = max_messages_before_summary
        self.messages: list[Message] = []
        self.summary: str = ""

    def _default_summarizer(self, text: str) -> str:
        return text[:200] + "..." if len(text) > 200 else text

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) >= self.max_messages_before_summary:
            self._summarize_messages()

    def _summarize_messages(self) -> str:
        lines = [f"{m.role}: {m.content}" for m in self.messages]
        text = "\n".join(lines)
        summarizer = self.summarizer or self._default_summarizer
        new_summary = summarizer(text)
        self.summary = f"{self.summary}\n{new_summary}" if self.summary else new_summary
        self.messages.clear()
        return self.summary

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def get_summary(self) -> str:
        return self.summary

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""

    def load_memory_variables(self, inputs: dict = None) -> dict:
        return {"history": self.get_messages(), "summary": self.summary}


class ConversationTokenMemory(BaseMemory):
    """Keeps messages up to a token limit."""

    def __init__(self, max_tokens: int = 2000, token_counter=None):
        self.max_tokens = max_tokens
        self.token_counter = token_counter or (lambda t: len(t.split()))
        self.messages: list[Message] = []

    def _count_message_tokens(self, message: Message) -> int:
        return self.token_counter(f"{message.role}: {message.content}")

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        while self.get_token_count() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def get_token_count(self) -> int:
        return sum(self._count_message_tokens(m) for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        return {"history": self.get_messages(), "token_count": self.get_token_count()}


class EntityMemory(BaseMemory):
    """Tracks entities mentioned in conversation."""

    def __init__(self, entity_extractor=None):
        import re

        self.entity_extractor = entity_extractor or (
            lambda t: [
                w
                for w in re.findall(r"\b[A-Z][a-z]+\b", t)
                if w
                not in {
                    "The",
                    "This",
                    "That",
                    "What",
                    "When",
                    "Where",
                    "How",
                    "Why",
                    "I",
                    "We",
                }
            ]
        )
        self.messages: list[Message] = []
        self.entities: dict[str, list[str]] = {}

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        for entity in self.entity_extractor(content):
            if entity not in self.entities:
                self.entities[entity] = []
            self.entities[entity].append(content)

    def get_entity_info(self, entity: str) -> list[str]:
        return self.entities.get(entity, [])

    def get_all_entities(self) -> dict:
        return dict(self.entities)

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.entities.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        return {"history": self.get_messages(), "entities": self.get_all_entities()}


class PersistentMemory(BaseMemory):
    """Memory that persists to disk."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.messages: list[Message] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.messages = [
                        Message(
                            role=m["role"],
                            content=m["content"],
                            timestamp=datetime.fromisoformat(
                                m.get("timestamp", datetime.now().isoformat())
                            ),
                        )
                        for m in data
                    ]
            except (json.JSONDecodeError, KeyError):
                self.messages = []

    def _save(self) -> None:
        data = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in self.messages
        ]
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self._save()

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def load_memory_variables(self, inputs: dict = None) -> dict:
        return {"history": self.get_messages()}


class ConversationChain:
    """A chain that maintains conversation memory."""

    def __init__(
        self, processor, memory: BaseMemory = None, prompt_template: str = None
    ):
        self.processor = processor
        self.memory = memory or ConversationBufferMemory()
        self.prompt_template = (
            prompt_template or "History:\n{history}\n\nHuman: {input}\nAI:"
        )

    def _build_prompt(self, user_input: str) -> str:
        variables = self.memory.load_memory_variables({})
        history = variables.get("history", [])
        if isinstance(history, list):
            history = "\n".join(f"{m.role}: {m.content}" for m in history)
        return self.prompt_template.format(history=history, input=user_input)

    def run(self, user_input: str) -> str:
        prompt = self._build_prompt(user_input)
        response = self.processor(prompt)
        self.memory.save_context({"input": user_input}, {"output": response})
        return response

    def get_memory(self) -> BaseMemory:
        return self.memory


class SearchableMemory(BaseMemory):
    """Memory that supports semantic search over history."""

    def __init__(self, embedding_fn=None):
        self.embedding_fn = embedding_fn or self._default_embedding
        self.messages: list[Message] = []
        self.embeddings: list[dict] = []

    def _default_embedding(self, text: str) -> dict:
        words = text.lower().split()
        return {w: words.count(w) for w in set(words)}

    def _similarity(self, emb1: dict, emb2: dict) -> float:
        common = set(emb1.keys()) & set(emb2.keys())
        if not common:
            return 0.0
        dot = sum(emb1.get(k, 0) * emb2.get(k, 0) for k in common)
        norm1 = sum(v**2 for v in emb1.values()) ** 0.5
        norm2 = sum(v**2 for v in emb2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def add_message(self, role: str, content: str) -> None:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.embeddings.append(self.embedding_fn(content))

    def search(self, query: str, k: int = 3) -> list[Message]:
        query_emb = self.embedding_fn(query)
        scores = [
            (i, self._similarity(query_emb, emb))
            for i, emb in enumerate(self.embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.messages[i] for i, _ in scores[:k]]

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.embeddings.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        if inputs and "query" in inputs:
            return {"history": self.search(inputs["query"])}
        return {"history": self.get_messages()}


class CombinedMemory(BaseMemory):
    """Combines multiple memory types."""

    def __init__(self, memories: list[BaseMemory]):
        self.memories = memories

    def add_message(self, role: str, content: str) -> None:
        for memory in self.memories:
            memory.add_message(role, content)

    def get_messages(self) -> list[Message]:
        return self.memories[0].get_messages() if self.memories else []

    def clear(self) -> None:
        for memory in self.memories:
            memory.clear()

    def load_memory_variables(self, inputs: dict = None) -> dict:
        result = {}
        for i, memory in enumerate(self.memories):
            variables = memory.load_memory_variables(inputs)
            for key, value in variables.items():
                result_key = key if key not in result else f"{key}_{i}"
                result[result_key] = value
        return result


if __name__ == "__main__":
    print("=== Testing Buffer Memory ===")
    buffer = ConversationBufferMemory()
    buffer.add_user_message("Hello!")
    buffer.add_ai_message("Hi there!")
    print(f"Messages: {len(buffer.get_messages())}")

    print("\n=== Testing Window Memory ===")
    window = ConversationBufferWindowMemory(k=2)
    for i in range(5):
        window.add_user_message(f"Q{i}")
        window.add_ai_message(f"A{i}")
    print(f"Window size: {len(window.get_messages())}")

    print("\n=== Testing Conversation Chain ===")
    chain = ConversationChain(processor=lambda p: f"Response to: {p[-30:]}")
    r1 = chain.run("Hello!")
    r2 = chain.run("How are you?")
    print(f"Response: {r2}")

    print("\n✅ All solutions verified!")

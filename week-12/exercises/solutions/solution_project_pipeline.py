"""
Week 12 - Project: Knowledge Base MCP Server - SOLUTIONS

Complete implementation of a Knowledge Base MCP server.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import asyncio
import json
import uuid
import hashlib
import os
import re


# =============================================================================
# Part 1: Document Models
# =============================================================================
class DocumentStatus(str, Enum):
    """Document lifecycle status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class DocumentVersion:
    """A version of a document."""

    version: int
    content: str
    created_at: datetime
    author: Optional[str] = None
    changes: Optional[str] = None


@dataclass
class Document:
    """
    Represents a knowledge base document.

    Attributes:
        id: Unique document identifier
        title: Document title
        content: Document content (Markdown supported)
        tags: List of tags for categorization
        status: Document lifecycle status
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
        versions: Version history
    """

    id: str
    title: str
    content: str
    tags: list[str]
    status: DocumentStatus = DocumentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    versions: list[DocumentVersion] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "version_count": len(self.versions),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create from dictionary."""
        status = data.get("status", "draft")
        if isinstance(status, str):
            status = DocumentStatus(status)

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            tags=data.get("tags", []),
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {}),
        )

    def add_version(
        self, author: Optional[str] = None, changes: Optional[str] = None
    ) -> None:
        """Add current state as a new version."""
        version = DocumentVersion(
            version=len(self.versions) + 1,
            content=self.content,
            created_at=datetime.now(),
            author=author,
            changes=changes,
        )
        self.versions.append(version)


@dataclass
class SearchResult:
    """A search result with relevance score."""

    document: Document
    score: float
    matched_fields: list[str]
    highlight: Optional[str] = None


# =============================================================================
# Part 2: Storage Backend
# =============================================================================
class StorageBackend(ABC):
    """Abstract storage backend for documents."""

    @abstractmethod
    async def save(self, document: Document) -> None:
        """Save a document."""
        pass

    @abstractmethod
    async def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass

    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    async def list_all(self) -> list[Document]:
        """List all documents."""
        pass

    @abstractmethod
    async def exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        pass


class InMemoryStorage(StorageBackend):
    """In-memory storage implementation."""

    def __init__(self):
        self._documents: dict[str, Document] = {}

    async def save(self, document: Document) -> None:
        """Save a document."""
        self._documents[document.id] = document

    async def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    async def list_all(self) -> list[Document]:
        """List all documents."""
        return list(self._documents.values())

    async def exists(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self._documents


class FileStorage(StorageBackend):
    """File-based storage implementation."""

    def __init__(self, base_path: str):
        self._base_path = base_path

    async def initialize(self) -> None:
        """Initialize storage directory."""
        os.makedirs(self._base_path, exist_ok=True)

    def _doc_path(self, doc_id: str) -> str:
        """Get file path for document."""
        return os.path.join(self._base_path, f"{doc_id}.json")

    async def save(self, document: Document) -> None:
        """Save a document to file."""
        path = self._doc_path(document.id)
        data = document.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    async def get(self, doc_id: str) -> Optional[Document]:
        """Get a document from file."""
        path = self._doc_path(doc_id)

        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            data = json.load(f)

        return Document.from_dict(data)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document file."""
        path = self._doc_path(doc_id)

        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    async def list_all(self) -> list[Document]:
        """List all documents from files."""
        documents = []

        if not os.path.exists(self._base_path):
            return documents

        for filename in os.listdir(self._base_path):
            if filename.endswith(".json"):
                doc_id = filename[:-5]
                doc = await self.get(doc_id)
                if doc:
                    documents.append(doc)

        return documents

    async def exists(self, doc_id: str) -> bool:
        """Check if document file exists."""
        return os.path.exists(self._doc_path(doc_id))


# =============================================================================
# Part 3: Search Index
# =============================================================================
class SearchIndex:
    """Full-text search index for documents."""

    def __init__(self):
        self._documents: dict[str, Document] = {}
        self._title_index: dict[str, set[str]] = {}
        self._content_index: dict[str, set[str]] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into searchable terms."""
        text = text.lower()
        words = re.findall(r"\b\w+\b", text)
        return words

    async def index_document(self, document: Document) -> None:
        """Index a document for searching."""
        self._documents[document.id] = document

        # Index title
        for word in self._tokenize(document.title):
            if word not in self._title_index:
                self._title_index[word] = set()
            self._title_index[word].add(document.id)

        # Index content
        for word in self._tokenize(document.content):
            if word not in self._content_index:
                self._content_index[word] = set()
            self._content_index[word].add(document.id)

    async def remove_document(self, doc_id: str) -> None:
        """Remove document from index."""
        if doc_id not in self._documents:
            return

        document = self._documents[doc_id]

        # Remove from title index
        for word in self._tokenize(document.title):
            if word in self._title_index:
                self._title_index[word].discard(doc_id)

        # Remove from content index
        for word in self._tokenize(document.content):
            if word in self._content_index:
                self._content_index[word].discard(doc_id)

        del self._documents[doc_id]

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for documents matching query."""
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Score documents
        scores: dict[str, tuple[float, list[str]]] = {}

        for term in query_terms:
            # Title matches (higher weight)
            if term in self._title_index:
                for doc_id in self._title_index[term]:
                    if doc_id not in scores:
                        scores[doc_id] = (0.0, [])
                    score, fields = scores[doc_id]
                    scores[doc_id] = (
                        score + 2.0,
                        fields + (["title"] if "title" not in fields else []),
                    )

            # Content matches
            if term in self._content_index:
                for doc_id in self._content_index[term]:
                    if doc_id not in scores:
                        scores[doc_id] = (0.0, [])
                    score, fields = scores[doc_id]
                    scores[doc_id] = (
                        score + 1.0,
                        fields + (["content"] if "content" not in fields else []),
                    )

        # Build results
        results = []
        for doc_id, (score, fields) in sorted(scores.items(), key=lambda x: -x[1][0])[
            :limit
        ]:
            doc = self._documents.get(doc_id)
            if doc:
                results.append(
                    SearchResult(
                        document=doc,
                        score=score / len(query_terms),
                        matched_fields=list(set(fields)),
                    )
                )

        return results


# =============================================================================
# Part 4: Tag Index
# =============================================================================
class TagIndex:
    """Index for document tags."""

    def __init__(self):
        self._tag_to_docs: dict[str, set[str]] = {}
        self._doc_to_tags: dict[str, set[str]] = {}

    async def add_document(self, doc_id: str, tags: list[str]) -> None:
        """Add document tags to index."""
        self._doc_to_tags[doc_id] = set(tags)

        for tag in tags:
            if tag not in self._tag_to_docs:
                self._tag_to_docs[tag] = set()
            self._tag_to_docs[tag].add(doc_id)

    async def remove_document(self, doc_id: str) -> None:
        """Remove document from tag index."""
        if doc_id not in self._doc_to_tags:
            return

        for tag in self._doc_to_tags[doc_id]:
            if tag in self._tag_to_docs:
                self._tag_to_docs[tag].discard(doc_id)

        del self._doc_to_tags[doc_id]

    async def update_tags(self, doc_id: str, tags: list[str]) -> None:
        """Update document tags."""
        await self.remove_document(doc_id)
        await self.add_document(doc_id, tags)

    async def get_documents_by_tag(self, tag: str) -> set[str]:
        """Get document IDs with a specific tag."""
        return self._tag_to_docs.get(tag, set()).copy()

    async def get_tags_for_document(self, doc_id: str) -> set[str]:
        """Get tags for a document."""
        return self._doc_to_tags.get(doc_id, set()).copy()

    async def get_all_tags(self) -> set[str]:
        """Get all unique tags."""
        return set(self._tag_to_docs.keys())

    async def get_tag_counts(self) -> dict[str, int]:
        """Get document count per tag."""
        return {tag: len(docs) for tag, docs in self._tag_to_docs.items()}


# =============================================================================
# Part 5: Knowledge Base Manager
# =============================================================================
class KnowledgeBase:
    """Main knowledge base manager."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage
        self._search_index = SearchIndex()
        self._tag_index = TagIndex()

    async def initialize(self) -> None:
        """Initialize knowledge base with existing documents."""
        documents = await self._storage.list_all()

        for doc in documents:
            await self._search_index.index_document(doc)
            await self._tag_index.add_document(doc.id, doc.tags)

    async def create_document(
        self,
        title: str,
        content: str,
        tags: list[str],
        metadata: Optional[dict] = None,
        status: DocumentStatus = DocumentStatus.DRAFT,
    ) -> Document:
        """Create a new document."""
        doc_id = str(uuid.uuid4())

        document = Document(
            id=doc_id,
            title=title,
            content=content,
            tags=tags,
            status=status,
            metadata=metadata or {},
        )

        await self._storage.save(document)
        await self._search_index.index_document(document)
        await self._tag_index.add_document(doc_id, tags)

        return document

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return await self._storage.get(doc_id)

    async def update_document(
        self,
        doc_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: Optional[DocumentStatus] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Document]:
        """Update an existing document."""
        document = await self._storage.get(doc_id)

        if document is None:
            return None

        # Create version before update
        document.add_version()

        # Update fields
        if title is not None:
            document.title = title
        if content is not None:
            document.content = content
        if tags is not None:
            document.tags = tags
            await self._tag_index.update_tags(doc_id, tags)
        if status is not None:
            document.status = status
        if metadata is not None:
            document.metadata.update(metadata)

        document.updated_at = datetime.now()

        # Save and re-index
        await self._storage.save(document)
        await self._search_index.remove_document(doc_id)
        await self._search_index.index_document(document)

        return document

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        await self._search_index.remove_document(doc_id)
        await self._tag_index.remove_document(doc_id)
        return await self._storage.delete(doc_id)

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for documents."""
        return await self._search_index.search(query, limit)

    async def get_documents_by_tag(self, tag: str) -> list[Document]:
        """Get documents with a specific tag."""
        doc_ids = await self._tag_index.get_documents_by_tag(tag)
        documents = []

        for doc_id in doc_ids:
            doc = await self._storage.get(doc_id)
            if doc:
                documents.append(doc)

        return documents

    async def list_all(self) -> list[Document]:
        """List all documents."""
        return await self._storage.list_all()

    async def get_all_tags(self) -> set[str]:
        """Get all unique tags."""
        return await self._tag_index.get_all_tags()

    async def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        all_docs = await self._storage.list_all()
        all_tags = await self._tag_index.get_all_tags()
        tag_counts = await self._tag_index.get_tag_counts()

        status_counts = {}
        for doc in all_docs:
            status = doc.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_documents": len(all_docs),
            "total_tags": len(all_tags),
            "status_breakdown": status_counts,
            "tag_counts": tag_counts,
        }


# =============================================================================
# Part 6: MCP Resources
# =============================================================================
class KnowledgeBaseResources:
    """MCP resources for the knowledge base."""

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    async def get_resource_definitions(self) -> list[dict]:
        """Get all resource definitions."""
        resources = [
            {
                "uri": "kb://documents",
                "name": "All Documents",
                "description": "List of all documents in the knowledge base",
                "mimeType": "application/json",
            },
            {
                "uri": "kb://tags",
                "name": "All Tags",
                "description": "List of all tags",
                "mimeType": "application/json",
            },
            {
                "uri": "kb://stats",
                "name": "Knowledge Base Stats",
                "description": "Statistics about the knowledge base",
                "mimeType": "application/json",
            },
        ]

        # Add individual document resources
        all_docs = await self._kb.list_all()
        for doc in all_docs:
            resources.append(
                {
                    "uri": f"kb://documents/{doc.id}",
                    "name": doc.title,
                    "description": f"Document: {doc.title}",
                    "mimeType": "application/json",
                }
            )

        return resources

    async def read_resource(self, uri: str) -> Optional[dict]:
        """Read a resource by URI."""
        if uri == "kb://documents":
            docs = await self._kb.list_all()
            return {"documents": [d.to_dict() for d in docs]}

        if uri == "kb://tags":
            tags = await self._kb.get_all_tags()
            tag_counts = await self._kb._tag_index.get_tag_counts()
            return {"tags": list(tags), "counts": tag_counts}

        if uri == "kb://stats":
            return await self._kb.get_stats()

        if uri.startswith("kb://documents/"):
            doc_id = uri.replace("kb://documents/", "")
            doc = await self._kb.get_document(doc_id)
            if doc:
                return doc.to_dict()

        return None


# =============================================================================
# Part 7: MCP Tools
# =============================================================================
class KnowledgeBaseTools:
    """MCP tools for the knowledge base."""

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    async def get_tool_definitions(self) -> list[dict]:
        """Get all tool definitions."""
        return [
            {
                "name": "create_document",
                "description": "Create a new document in the knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Document title"},
                        "content": {
                            "type": "string",
                            "description": "Document content (Markdown)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                    },
                    "required": ["title", "content", "tags"],
                },
            },
            {
                "name": "update_document",
                "description": "Update an existing document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Document ID"},
                        "title": {"type": "string", "description": "New title"},
                        "content": {"type": "string", "description": "New content"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "delete_document",
                "description": "Delete a document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Document ID"}
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "search_documents",
                "description": "Search for documents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_by_tag",
                "description": "Get documents with a specific tag",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "description": "Tag to filter by"}
                    },
                    "required": ["tag"],
                },
            },
            {
                "name": "publish_document",
                "description": "Publish a draft document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Document ID"}
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "archive_document",
                "description": "Archive a document",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Document ID"}
                    },
                    "required": ["id"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool with arguments."""
        handlers = {
            "create_document": self._create_document,
            "update_document": self._update_document,
            "delete_document": self._delete_document,
            "search_documents": self._search_documents,
            "get_by_tag": self._get_by_tag,
            "publish_document": self._publish_document,
            "archive_document": self._archive_document,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        return await handler(arguments)

    async def _create_document(self, args: dict) -> dict:
        """Create document handler."""
        doc = await self._kb.create_document(
            title=args["title"], content=args["content"], tags=args["tags"]
        )
        return {"id": doc.id, "title": doc.title, "status": "created"}

    async def _update_document(self, args: dict) -> dict:
        """Update document handler."""
        doc = await self._kb.update_document(
            doc_id=args["id"],
            title=args.get("title"),
            content=args.get("content"),
            tags=args.get("tags"),
        )
        if doc:
            return {"id": doc.id, "status": "updated"}
        return {"error": "Document not found"}

    async def _delete_document(self, args: dict) -> dict:
        """Delete document handler."""
        success = await self._kb.delete_document(args["id"])
        return {"deleted": success}

    async def _search_documents(self, args: dict) -> dict:
        """Search documents handler."""
        results = await self._kb.search(
            query=args["query"], limit=args.get("limit", 10)
        )
        return {
            "results": [
                {
                    "id": r.document.id,
                    "title": r.document.title,
                    "score": r.score,
                    "matched_fields": r.matched_fields,
                }
                for r in results
            ]
        }

    async def _get_by_tag(self, args: dict) -> dict:
        """Get by tag handler."""
        docs = await self._kb.get_documents_by_tag(args["tag"])
        return {"documents": [{"id": d.id, "title": d.title} for d in docs]}

    async def _publish_document(self, args: dict) -> dict:
        """Publish document handler."""
        doc = await self._kb.update_document(
            doc_id=args["id"], status=DocumentStatus.PUBLISHED
        )
        if doc:
            return {"id": doc.id, "status": "published"}
        return {"error": "Document not found"}

    async def _archive_document(self, args: dict) -> dict:
        """Archive document handler."""
        doc = await self._kb.update_document(
            doc_id=args["id"], status=DocumentStatus.ARCHIVED
        )
        if doc:
            return {"id": doc.id, "status": "archived"}
        return {"error": "Document not found"}


# =============================================================================
# Part 8: MCP Prompts
# =============================================================================
class KnowledgeBasePrompts:
    """MCP prompts for the knowledge base."""

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb

    async def get_prompt_definitions(self) -> list[dict]:
        """Get all prompt definitions."""
        return [
            {
                "name": "summarize_document",
                "description": "Generate a prompt to summarize a document",
                "arguments": [
                    {"name": "doc_id", "description": "Document ID", "required": True}
                ],
            },
            {
                "name": "compare_documents",
                "description": "Generate a prompt to compare two documents",
                "arguments": [
                    {
                        "name": "doc_id_1",
                        "description": "First document ID",
                        "required": True,
                    },
                    {
                        "name": "doc_id_2",
                        "description": "Second document ID",
                        "required": True,
                    },
                ],
            },
            {
                "name": "suggest_tags",
                "description": "Generate a prompt to suggest tags for a document",
                "arguments": [
                    {"name": "doc_id", "description": "Document ID", "required": True}
                ],
            },
            {
                "name": "generate_outline",
                "description": "Generate an outline prompt for a new document",
                "arguments": [
                    {
                        "name": "topic",
                        "description": "Topic for the document",
                        "required": True,
                    }
                ],
            },
            {
                "name": "qa_from_kb",
                "description": "Generate a Q&A prompt using knowledge base context",
                "arguments": [
                    {
                        "name": "question",
                        "description": "Question to answer",
                        "required": True,
                    }
                ],
            },
        ]

    async def get_prompt(self, name: str, arguments: dict) -> dict:
        """Get a rendered prompt."""
        handlers = {
            "summarize_document": self._summarize_prompt,
            "compare_documents": self._compare_prompt,
            "suggest_tags": self._suggest_tags_prompt,
            "generate_outline": self._outline_prompt,
            "qa_from_kb": self._qa_prompt,
        }

        handler = handlers.get(name)
        if not handler:
            return {"messages": []}

        return await handler(arguments)

    async def _summarize_prompt(self, args: dict) -> dict:
        """Generate summarize prompt."""
        doc = await self._kb.get_document(args["doc_id"])
        if not doc:
            return {"messages": []}

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Please summarize the following document:\n\nTitle: {doc.title}\n\nContent:\n{doc.content}",
                    },
                }
            ]
        }

    async def _compare_prompt(self, args: dict) -> dict:
        """Generate compare prompt."""
        doc1 = await self._kb.get_document(args["doc_id_1"])
        doc2 = await self._kb.get_document(args["doc_id_2"])

        if not doc1 or not doc2:
            return {"messages": []}

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Please compare these two documents:\n\n"
                        f"Document 1: {doc1.title}\n{doc1.content}\n\n"
                        f"Document 2: {doc2.title}\n{doc2.content}",
                    },
                }
            ]
        }

    async def _suggest_tags_prompt(self, args: dict) -> dict:
        """Generate tag suggestion prompt."""
        doc = await self._kb.get_document(args["doc_id"])
        if not doc:
            return {"messages": []}

        existing_tags = await self._kb.get_all_tags()

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Suggest tags for this document. "
                        f"Current tags: {', '.join(doc.tags)}. "
                        f"Existing tags in KB: {', '.join(existing_tags)}.\n\n"
                        f"Title: {doc.title}\n\nContent:\n{doc.content}",
                    },
                }
            ]
        }

    async def _outline_prompt(self, args: dict) -> dict:
        """Generate outline prompt."""
        topic = args["topic"]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Generate a detailed outline for a knowledge base document about: {topic}",
                    },
                }
            ]
        }

    async def _qa_prompt(self, args: dict) -> dict:
        """Generate Q&A prompt with KB context."""
        question = args["question"]

        # Search for relevant documents
        results = await self._kb.search(question, limit=3)

        context = ""
        for r in results:
            context += f"\n--- {r.document.title} ---\n{r.document.content}\n"

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Using the following context from our knowledge base, "
                        f"please answer the question.\n\n"
                        f"Context:{context}\n\n"
                        f"Question: {question}",
                    },
                }
            ]
        }


# =============================================================================
# Part 9: Document Exporter
# =============================================================================
class ExportFormat(str, Enum):
    """Export format options."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class DocumentExporter:
    """Export documents in various formats."""

    def export(self, document: Document, format: ExportFormat) -> str:
        """Export a document to the specified format."""
        exporters = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.MARKDOWN: self._export_markdown,
            ExportFormat.HTML: self._export_html,
        }

        exporter = exporters.get(format)
        if not exporter:
            raise ValueError(f"Unknown format: {format}")

        return exporter(document)

    def _export_json(self, document: Document) -> str:
        """Export as JSON."""
        return json.dumps(document.to_dict(), indent=2)

    def _export_markdown(self, document: Document) -> str:
        """Export as Markdown."""
        tags_str = ", ".join(document.tags) if document.tags else "None"

        return f"""# {document.title}

**Status:** {document.status.value}
**Tags:** {tags_str}
**Created:** {document.created_at.isoformat()}
**Updated:** {document.updated_at.isoformat()}

---

{document.content}
"""

    def _export_html(self, document: Document) -> str:
        """Export as HTML."""
        tags_html = ", ".join(f"<span class='tag'>{t}</span>" for t in document.tags)

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{document.title}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }}
        .meta {{ color: #666; font-size: 0.9em; }}
        .tag {{ background: #eee; padding: 2px 8px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>{document.title}</h1>
    <div class="meta">
        <p>Status: {document.status.value}</p>
        <p>Tags: {tags_html}</p>
        <p>Created: {document.created_at.isoformat()}</p>
    </div>
    <hr>
    <div class="content">
        {document.content}
    </div>
</body>
</html>
"""

    async def export_all(
        self, kb: KnowledgeBase, format: ExportFormat
    ) -> dict[str, str]:
        """Export all documents."""
        documents = await kb.list_all()

        return {doc.id: self.export(doc, format) for doc in documents}


# =============================================================================
# Part 10: Complete MCP Server
# =============================================================================
class KnowledgeBaseMCPServer:
    """Complete MCP server for the knowledge base."""

    def __init__(self, storage: StorageBackend):
        self._kb = KnowledgeBase(storage)
        self._resources = KnowledgeBaseResources(self._kb)
        self._tools = KnowledgeBaseTools(self._kb)
        self._prompts = KnowledgeBasePrompts(self._kb)

    async def initialize(self) -> dict:
        """Initialize the server."""
        await self._kb.initialize()

        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "knowledge-base-server", "version": "1.0.0"},
            "capabilities": {"resources": {}, "tools": {}, "prompts": {}},
        }

    async def list_resources(self) -> list[dict]:
        """List all resources."""
        return await self._resources.get_resource_definitions()

    async def read_resource(self, uri: str) -> Optional[dict]:
        """Read a resource."""
        return await self._resources.read_resource(uri)

    async def list_tools(self) -> list[dict]:
        """List all tools."""
        return await self._tools.get_tool_definitions()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool."""
        return await self._tools.call_tool(name, arguments)

    async def list_prompts(self) -> list[dict]:
        """List all prompts."""
        return await self._prompts.get_prompt_definitions()

    async def get_prompt(self, name: str, arguments: dict) -> dict:
        """Get a rendered prompt."""
        return await self._prompts.get_prompt(name, arguments)

    async def handle_request(self, method: str, params: dict) -> dict:
        """Handle an MCP request."""
        handlers = {
            "initialize": lambda _: self.initialize(),
            "resources/list": lambda _: self.list_resources(),
            "resources/read": lambda p: self.read_resource(p.get("uri", "")),
            "tools/list": lambda _: self.list_tools(),
            "tools/call": lambda p: self.call_tool(
                p.get("name", ""), p.get("arguments", {})
            ),
            "prompts/list": lambda _: self.list_prompts(),
            "prompts/get": lambda p: self.get_prompt(
                p.get("name", ""), p.get("arguments", {})
            ),
        }

        handler = handlers.get(method)
        if not handler:
            return {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

        result = handler(params)
        if asyncio.iscoroutine(result):
            result = await result

        return result


# =============================================================================
# Factory
# =============================================================================
class KnowledgeBaseServerFactory:
    """Factory for creating knowledge base servers."""

    @staticmethod
    def create_with_memory_storage() -> KnowledgeBaseMCPServer:
        """Create server with in-memory storage."""
        return KnowledgeBaseMCPServer(InMemoryStorage())

    @staticmethod
    def create_with_file_storage(path: str) -> KnowledgeBaseMCPServer:
        """Create server with file storage."""
        return KnowledgeBaseMCPServer(FileStorage(path))


# =============================================================================
# Example Usage
# =============================================================================
async def main():
    """Demonstrate knowledge base server."""
    # Create server
    server = KnowledgeBaseServerFactory.create_with_memory_storage()

    # Initialize
    result = await server.handle_request("initialize", {})
    print("Initialize:", json.dumps(result, indent=2))

    # Create some documents
    result = await server.call_tool(
        "create_document",
        {
            "title": "Getting Started with Python",
            "content": "Python is a versatile programming language...",
            "tags": ["python", "programming", "tutorial"],
        },
    )
    doc_id_1 = result["id"]
    print(f"\nCreated document: {doc_id_1}")

    result = await server.call_tool(
        "create_document",
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI...",
            "tags": ["machine-learning", "ai", "tutorial"],
        },
    )
    doc_id_2 = result["id"]
    print(f"Created document: {doc_id_2}")

    # List tools
    tools = await server.list_tools()
    print(f"\nAvailable tools: {[t['name'] for t in tools]}")

    # Search
    result = await server.call_tool(
        "search_documents", {"query": "programming", "limit": 5}
    )
    print(f"\nSearch results: {json.dumps(result, indent=2)}")

    # List resources
    resources = await server.list_resources()
    print(f"\nAvailable resources: {[r['name'] for r in resources[:5]]}...")

    # Read KB stats
    stats = await server.read_resource("kb://stats")
    print(f"\nKB Stats: {json.dumps(stats, indent=2)}")

    # Get a prompt
    prompt = await server.get_prompt("summarize_document", {"doc_id": doc_id_1})
    print(
        f"\nSummarize prompt preview: {prompt['messages'][0]['content']['text'][:100]}..."
    )

    # Export a document
    exporter = DocumentExporter()
    kb = server._kb
    doc = await kb.get_document(doc_id_1)
    if doc:
        md_export = exporter.export(doc, ExportFormat.MARKDOWN)
        print(f"\nMarkdown export:\n{md_export[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())

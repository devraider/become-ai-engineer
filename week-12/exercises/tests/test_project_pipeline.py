"""
Tests for Week 12 - Project: Knowledge Base MCP Server
"""

import pytest
import asyncio
from datetime import datetime
from project_pipeline import (
    DocumentStatus,
    DocumentVersion,
    Document,
    SearchResult,
    StorageBackend,
    InMemoryStorage,
    FileStorage,
    SearchIndex,
    TagIndex,
    KnowledgeBase,
    KnowledgeBaseResources,
    KnowledgeBaseTools,
    KnowledgeBasePrompts,
    ExportFormat,
    DocumentExporter,
    KnowledgeBaseMCPServer,
    KnowledgeBaseServerFactory,
)


# =============================================================================
# Test Document Models
# =============================================================================
class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert DocumentStatus.DRAFT.value == "draft"
        assert DocumentStatus.PUBLISHED.value == "published"
        assert DocumentStatus.ARCHIVED.value == "archived"


class TestDocumentVersion:
    """Tests for DocumentVersion."""

    def test_version_creation(self):
        """Test creating a document version."""
        version = DocumentVersion(
            version=1, content="Test content", created_at=datetime.now()
        )

        assert version.version == 1
        assert version.content == "Test content"


class TestDocument:
    """Tests for Document."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            title="Test Document",
            content="Test content",
            tags=["test", "example"],
        )

        assert doc.id == "doc1"
        assert doc.title == "Test Document"
        assert "test" in doc.tags

    def test_document_default_status(self):
        """Test default document status."""
        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        assert doc.status == DocumentStatus.DRAFT

    def test_document_to_dict(self):
        """Test converting document to dict."""
        doc = Document(id="doc1", title="Test", content="Content", tags=["tag1"])

        data = doc.to_dict()

        assert data["id"] == "doc1"
        assert data["title"] == "Test"

    def test_document_from_dict(self):
        """Test creating document from dict."""
        data = {
            "id": "doc1",
            "title": "Test",
            "content": "Content",
            "tags": ["tag1"],
            "status": "draft",
            "metadata": {},
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc1"
        assert doc.title == "Test"


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Test creating search result."""
        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        result = SearchResult(
            document=doc, score=0.95, matched_fields=["title", "content"]
        )

        assert result.score == 0.95
        assert "title" in result.matched_fields


# =============================================================================
# Test Storage Backends
# =============================================================================
class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        """Test saving and retrieving document."""
        storage = InMemoryStorage()

        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        await storage.save(doc)
        retrieved = await storage.get("doc1")

        assert retrieved.id == "doc1"
        assert retrieved.title == "Test"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting document."""
        storage = InMemoryStorage()

        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        await storage.save(doc)
        result = await storage.delete("doc1")

        assert result is True
        assert await storage.get("doc1") is None

    @pytest.mark.asyncio
    async def test_list_all(self):
        """Test listing all documents."""
        storage = InMemoryStorage()

        doc1 = Document(id="doc1", title="Test 1", content="Content 1", tags=[])
        doc2 = Document(id="doc2", title="Test 2", content="Content 2", tags=[])

        await storage.save(doc1)
        await storage.save(doc2)

        all_docs = await storage.list_all()

        assert len(all_docs) == 2

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test checking document existence."""
        storage = InMemoryStorage()

        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        await storage.save(doc)

        assert await storage.exists("doc1") is True
        assert await storage.exists("nonexistent") is False


class TestFileStorage:
    """Tests for FileStorage."""

    @pytest.mark.asyncio
    async def test_file_storage_initialization(self, tmp_path):
        """Test initializing file storage."""
        storage = FileStorage(str(tmp_path))
        await storage.initialize()

        assert tmp_path.exists()


# =============================================================================
# Test Search Index
# =============================================================================
class TestSearchIndex:
    """Tests for SearchIndex."""

    @pytest.mark.asyncio
    async def test_index_document(self):
        """Test indexing a document."""
        index = SearchIndex()

        doc = Document(
            id="doc1",
            title="Python Programming",
            content="Learn Python basics",
            tags=["python", "programming"],
        )

        await index.index_document(doc)

        # Document should be searchable
        results = await index.search("Python")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_by_content(self):
        """Test searching by content."""
        index = SearchIndex()

        doc = Document(
            id="doc1", title="Guide", content="Machine learning fundamentals", tags=[]
        )

        await index.index_document(doc)

        results = await index.search("machine learning")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_remove_from_index(self):
        """Test removing document from index."""
        index = SearchIndex()

        doc = Document(id="doc1", title="Test", content="Content", tags=[])

        await index.index_document(doc)
        await index.remove_document("doc1")

        results = await index.search("Test")

        assert len(results) == 0


# =============================================================================
# Test Tag Index
# =============================================================================
class TestTagIndex:
    """Tests for TagIndex."""

    @pytest.mark.asyncio
    async def test_add_tags(self):
        """Test adding document tags."""
        tag_index = TagIndex()

        await tag_index.add_document("doc1", ["python", "web"])

        docs = await tag_index.get_documents_by_tag("python")

        assert "doc1" in docs

    @pytest.mark.asyncio
    async def test_remove_tags(self):
        """Test removing document from tags."""
        tag_index = TagIndex()

        await tag_index.add_document("doc1", ["python", "web"])
        await tag_index.remove_document("doc1")

        docs = await tag_index.get_documents_by_tag("python")

        assert "doc1" not in docs

    @pytest.mark.asyncio
    async def test_get_all_tags(self):
        """Test getting all tags."""
        tag_index = TagIndex()

        await tag_index.add_document("doc1", ["python", "web"])
        await tag_index.add_document("doc2", ["python", "api"])

        all_tags = await tag_index.get_all_tags()

        assert "python" in all_tags
        assert "web" in all_tags
        assert "api" in all_tags


# =============================================================================
# Test Knowledge Base
# =============================================================================
class TestKnowledgeBase:
    """Tests for KnowledgeBase."""

    @pytest.fixture
    def kb(self):
        """Create a knowledge base instance."""
        return KnowledgeBase(InMemoryStorage())

    @pytest.mark.asyncio
    async def test_create_document(self, kb):
        """Test creating a document."""
        doc = await kb.create_document(
            title="Test Document", content="Test content", tags=["test"]
        )

        assert doc.title == "Test Document"
        assert doc.id is not None

    @pytest.mark.asyncio
    async def test_get_document(self, kb):
        """Test getting a document."""
        doc = await kb.create_document(title="Test", content="Content", tags=[])

        retrieved = await kb.get_document(doc.id)

        assert retrieved.id == doc.id

    @pytest.mark.asyncio
    async def test_update_document(self, kb):
        """Test updating a document."""
        doc = await kb.create_document(
            title="Original", content="Original content", tags=[]
        )

        updated = await kb.update_document(doc.id, title="Updated Title")

        assert updated.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_document(self, kb):
        """Test deleting a document."""
        doc = await kb.create_document(title="Test", content="Content", tags=[])

        result = await kb.delete_document(doc.id)

        assert result is True
        assert await kb.get_document(doc.id) is None

    @pytest.mark.asyncio
    async def test_search(self, kb):
        """Test searching documents."""
        await kb.create_document(
            title="Python Guide", content="Learn Python programming", tags=["python"]
        )

        results = await kb.search("Python")

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_by_tag(self, kb):
        """Test getting documents by tag."""
        await kb.create_document(
            title="Tagged Doc", content="Content", tags=["special"]
        )

        docs = await kb.get_documents_by_tag("special")

        assert len(docs) == 1


# =============================================================================
# Test MCP Resources
# =============================================================================
class TestKnowledgeBaseResources:
    """Tests for KnowledgeBaseResources."""

    @pytest.fixture
    def resources(self):
        """Create resources instance."""
        kb = KnowledgeBase(InMemoryStorage())
        return KnowledgeBaseResources(kb)

    @pytest.mark.asyncio
    async def test_get_resource_definitions(self, resources):
        """Test getting resource definitions."""
        definitions = await resources.get_resource_definitions()

        assert len(definitions) > 0
        assert any("document" in d["uri"].lower() for d in definitions)

    @pytest.mark.asyncio
    async def test_read_document_resource(self, resources):
        """Test reading document resource."""
        # Create a document first
        doc = await resources._kb.create_document(
            title="Test", content="Content", tags=[]
        )

        content = await resources.read_resource(f"kb://documents/{doc.id}")

        assert content is not None


# =============================================================================
# Test MCP Tools
# =============================================================================
class TestKnowledgeBaseTools:
    """Tests for KnowledgeBaseTools."""

    @pytest.fixture
    def tools(self):
        """Create tools instance."""
        kb = KnowledgeBase(InMemoryStorage())
        return KnowledgeBaseTools(kb)

    @pytest.mark.asyncio
    async def test_get_tool_definitions(self, tools):
        """Test getting tool definitions."""
        definitions = await tools.get_tool_definitions()

        assert len(definitions) > 0
        assert any("create" in d["name"].lower() for d in definitions)

    @pytest.mark.asyncio
    async def test_call_create_tool(self, tools):
        """Test calling create document tool."""
        result = await tools.call_tool(
            "create_document",
            {"title": "Test Document", "content": "Test content", "tags": ["test"]},
        )

        assert "id" in result or "document" in str(result).lower()

    @pytest.mark.asyncio
    async def test_call_search_tool(self, tools):
        """Test calling search tool."""
        # Create a document
        await tools.call_tool(
            "create_document",
            {"title": "Searchable", "content": "Unique content here", "tags": []},
        )

        result = await tools.call_tool("search_documents", {"query": "Unique"})

        assert result is not None


# =============================================================================
# Test MCP Prompts
# =============================================================================
class TestKnowledgeBasePrompts:
    """Tests for KnowledgeBasePrompts."""

    @pytest.fixture
    def prompts(self):
        """Create prompts instance."""
        kb = KnowledgeBase(InMemoryStorage())
        return KnowledgeBasePrompts(kb)

    @pytest.mark.asyncio
    async def test_get_prompt_definitions(self, prompts):
        """Test getting prompt definitions."""
        definitions = await prompts.get_prompt_definitions()

        assert len(definitions) > 0

    @pytest.mark.asyncio
    async def test_get_prompt(self, prompts):
        """Test getting a prompt."""
        definitions = await prompts.get_prompt_definitions()

        if definitions:
            prompt = await prompts.get_prompt(definitions[0]["name"], {})

            assert prompt is not None


# =============================================================================
# Test Exporter
# =============================================================================
class TestDocumentExporter:
    """Tests for DocumentExporter."""

    def test_export_json(self):
        """Test exporting to JSON."""
        doc = Document(id="doc1", title="Test", content="Content", tags=["test"])

        exporter = DocumentExporter()
        result = exporter.export(doc, ExportFormat.JSON)

        assert "doc1" in result
        assert "Test" in result

    def test_export_markdown(self):
        """Test exporting to Markdown."""
        doc = Document(
            id="doc1", title="Test Document", content="Content here", tags=["tag1"]
        )

        exporter = DocumentExporter()
        result = exporter.export(doc, ExportFormat.MARKDOWN)

        assert "# Test Document" in result or "Test Document" in result

    def test_export_html(self):
        """Test exporting to HTML."""
        doc = Document(id="doc1", title="Test", content="<p>Content</p>", tags=[])

        exporter = DocumentExporter()
        result = exporter.export(doc, ExportFormat.HTML)

        assert "<html>" in result.lower() or "<h1>" in result.lower()


# =============================================================================
# Test MCP Server
# =============================================================================
class TestKnowledgeBaseMCPServer:
    """Tests for KnowledgeBaseMCPServer."""

    @pytest.fixture
    def server(self):
        """Create server instance."""
        return KnowledgeBaseMCPServer(InMemoryStorage())

    @pytest.mark.asyncio
    async def test_list_resources(self, server):
        """Test listing resources."""
        resources = await server.list_resources()

        assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test listing tools."""
        tools = await server.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_list_prompts(self, server):
        """Test listing prompts."""
        prompts = await server.list_prompts()

        assert isinstance(prompts, list)


# =============================================================================
# Test Server Factory
# =============================================================================
class TestKnowledgeBaseServerFactory:
    """Tests for KnowledgeBaseServerFactory."""

    def test_create_with_memory_storage(self):
        """Test creating server with memory storage."""
        server = KnowledgeBaseServerFactory.create_with_memory_storage()

        assert server is not None
        assert isinstance(server, KnowledgeBaseMCPServer)

    def test_create_with_file_storage(self, tmp_path):
        """Test creating server with file storage."""
        server = KnowledgeBaseServerFactory.create_with_file_storage(str(tmp_path))

        assert server is not None
        assert isinstance(server, KnowledgeBaseMCPServer)


# =============================================================================
# Integration Tests
# =============================================================================
class TestKnowledgeBaseIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_full_document_workflow(self):
        """Test complete document lifecycle."""
        server = KnowledgeBaseServerFactory.create_with_memory_storage()

        # Create document via tools
        tools = await server.list_tools()
        create_tool = next(t for t in tools if "create" in t["name"].lower())

        # Create
        result = await server._tools.call_tool(
            create_tool["name"],
            {
                "title": "Integration Test",
                "content": "Integration test content",
                "tags": ["integration", "test"],
            },
        )

        # List all documents
        resources = await server.list_resources()

        # Search
        search_result = await server._tools.call_tool(
            "search_documents", {"query": "Integration"}
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_search_and_retrieve(self):
        """Test searching and retrieving documents."""
        kb = KnowledgeBase(InMemoryStorage())

        # Create documents
        doc1 = await kb.create_document(
            title="Python Tutorial",
            content="Learn Python step by step",
            tags=["python", "tutorial"],
        )

        doc2 = await kb.create_document(
            title="JavaScript Guide",
            content="JavaScript for beginners",
            tags=["javascript", "guide"],
        )

        # Search for Python
        results = await kb.search("Python")

        assert len(results) >= 1
        assert any(r.document.id == doc1.id for r in results)

        # Get by tag
        python_docs = await kb.get_documents_by_tag("python")

        assert len(python_docs) == 1
        assert python_docs[0].id == doc1.id

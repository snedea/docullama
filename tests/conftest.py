"""
PyTest configuration and fixtures for DocuLLaMA tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

# Import your app components
from config import Settings
from app import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test settings with mock values"""
    return Settings(
        environment="test",
        debug=True,
        azure_openai_endpoint="https://test.openai.azure.com/",
        azure_openai_api_key="test-key",
        qdrant_host="test-qdrant",
        qdrant_api_key="test-qdrant-key",
        jwt_secret_key="test-jwt-secret"
    )


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_azure_openai():
    """Mock Azure OpenAI client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    mock_client = AsyncMock()
    mock_client.get_collections.return_value = Mock()
    mock_client.get_collections.return_value.collections = []
    return mock_client


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return {
        "filename": "test_document.pdf",
        "content": "This is a test document with sample content for testing purposes.",
        "metadata": {
            "author": "Test Author",
            "title": "Test Document",
            "created_at": "2025-01-01T00:00:00Z"
        }
    }


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    from document_processor import DocumentChunk
    
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            content="First chunk content",
            chunk_type="introduction",
            position=0,
            metadata={"document_id": "doc_1", "page": 1}
        ),
        DocumentChunk(
            chunk_id="chunk_2", 
            content="Second chunk content",
            chunk_type="content",
            position=1,
            metadata={"document_id": "doc_1", "page": 1}
        )
    ]
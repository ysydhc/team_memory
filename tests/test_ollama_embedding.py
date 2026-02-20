"""Tests for Ollama embedding provider."""

import json

import httpx
import pytest

from team_memory.embedding.ollama_provider import OllamaEmbedding


class TestOllamaEmbedding:
    """Test Ollama embedding provider."""

    def test_dimension(self):
        provider = OllamaEmbedding(dim=768)
        assert provider.dimension == 768

    def test_custom_dimension(self):
        provider = OllamaEmbedding(dim=1024)
        assert provider.dimension == 1024

    def test_default_model(self):
        provider = OllamaEmbedding()
        assert provider._model == "nomic-embed-text"

    def test_custom_model(self):
        provider = OllamaEmbedding(model="mxbai-embed-large")
        assert provider._model == "mxbai-embed-large"

    def test_default_base_url(self):
        provider = OllamaEmbedding()
        assert provider._base_url == "http://localhost:11434"

    def test_trailing_slash_stripped(self):
        provider = OllamaEmbedding(base_url="http://localhost:11434/")
        assert provider._base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_encode_calls_api(self):
        """Test that encode calls the Ollama API correctly."""
        mock_response = {
            "model": "nomic-embed-text",
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
        }

        captured_request = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["body"] = json.loads(request.content)
            return httpx.Response(200, json=mock_response)

        provider = OllamaEmbedding(
            model="nomic-embed-text",
            dim=3,
            base_url="http://localhost:11434",
        )

        transport = httpx.MockTransport(mock_handler)
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            results = await provider.encode(["hello", "world"])
            assert len(results) == 2
            assert results[0] == [0.1, 0.2, 0.3]
            assert results[1] == [0.4, 0.5, 0.6]

            # Verify the request was correct
            assert captured_request["url"] == "http://localhost:11434/api/embed"
            assert captured_request["body"]["model"] == "nomic-embed-text"
            assert captured_request["body"]["input"] == ["hello", "world"]
        finally:
            httpx.AsyncClient.__init__ = original_init

    @pytest.mark.asyncio
    async def test_encode_single_text(self):
        """Test encoding a single text."""
        mock_response = {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        provider = OllamaEmbedding(dim=3)

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=mock_response)
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            result = await provider.encode_single("hello")
            assert result == [0.1, 0.2, 0.3]
        finally:
            httpx.AsyncClient.__init__ = original_init

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test that ConnectionError is raised when Ollama is not running."""
        provider = OllamaEmbedding(base_url="http://localhost:99999")

        transport = httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("Connection refused"))
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
                await provider.encode(["hello"])
        finally:
            httpx.AsyncClient.__init__ = original_init

    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test that HTTP errors are propagated."""
        provider = OllamaEmbedding()

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                404,
                json={"error": "model not found"},
            )
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        httpx.AsyncClient.__init__ = patched_init
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await provider.encode(["hello"])
        finally:
            httpx.AsyncClient.__init__ = original_init

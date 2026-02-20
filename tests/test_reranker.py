"""Tests for reranker providers."""

import pytest

from team_memory.config import LLMConfig, RerankerConfig
from team_memory.reranker.base import RerankerProvider, RerankResult
from team_memory.reranker.factory import create_reranker
from team_memory.reranker.noop_provider import NoopRerankerProvider

# ======================== NoopRerankerProvider ========================


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order():
    """Noop reranker should return docs in original order."""
    reranker = NoopRerankerProvider()
    docs = ["doc A", "doc B", "doc C"]
    results = await reranker.rank("query", docs)

    assert len(results) == 3
    assert results[0].index == 0
    assert results[1].index == 1
    assert results[2].index == 2
    # Scores should be decreasing
    assert results[0].score > results[1].score > results[2].score


@pytest.mark.asyncio
async def test_noop_reranker_respects_top_k():
    """Noop reranker should respect top_k parameter."""
    reranker = NoopRerankerProvider()
    docs = ["doc A", "doc B", "doc C", "doc D", "doc E"]
    results = await reranker.rank("query", docs, top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_noop_reranker_empty_docs():
    """Noop reranker should handle empty document list."""
    reranker = NoopRerankerProvider()
    results = await reranker.rank("query", [])
    assert results == []


@pytest.mark.asyncio
async def test_noop_reranker_provider_name():
    reranker = NoopRerankerProvider()
    assert reranker.provider_name == "noop"


# ======================== OllamaLLMRerankerProvider ========================


@pytest.mark.asyncio
async def test_ollama_llm_reranker_parse_scores():
    """Test score parsing from LLM response."""
    from team_memory.reranker.ollama_llm_provider import OllamaLLMRerankerProvider

    # Valid JSON array
    content = '[{"index": 0, "score": 8}, {"index": 1, "score": 3}]'
    scores = OllamaLLMRerankerProvider._parse_scores(content)
    assert scores == [8.0, 3.0]


@pytest.mark.asyncio
async def test_ollama_llm_reranker_parse_scores_markdown():
    """Test parsing scores wrapped in markdown code blocks."""
    from team_memory.reranker.ollama_llm_provider import OllamaLLMRerankerProvider

    content = '```json\n[{"index": 0, "score": 9}, {"index": 1, "score": 5}]\n```'
    scores = OllamaLLMRerankerProvider._parse_scores(content)
    assert scores == [9.0, 5.0]


@pytest.mark.asyncio
async def test_ollama_llm_reranker_parse_scores_fallback_regex():
    """Test fallback regex parsing when JSON fails."""
    from team_memory.reranker.ollama_llm_provider import OllamaLLMRerankerProvider

    content = 'Document 0 score: 7\nDocument 1 score: 4'
    scores = OllamaLLMRerankerProvider._parse_scores(content)
    assert scores is not None
    assert len(scores) == 2
    assert scores[0] == 7.0
    assert scores[1] == 4.0


@pytest.mark.asyncio
async def test_ollama_llm_reranker_parse_scores_clamp():
    """Test that scores are clamped to 0-10."""
    from team_memory.reranker.ollama_llm_provider import OllamaLLMRerankerProvider

    content = '[{"index": 0, "score": 15}, {"index": 1, "score": -3}]'
    scores = OllamaLLMRerankerProvider._parse_scores(content)
    assert scores[0] == 10.0
    assert scores[1] == 0.0


# ======================== Factory ========================


def test_factory_creates_noop():
    """Factory should create NoopRerankerProvider for 'none'."""
    config = RerankerConfig(provider="none")
    reranker = create_reranker(config)
    assert isinstance(reranker, NoopRerankerProvider)


def test_factory_creates_ollama_llm():
    """Factory should create OllamaLLMRerankerProvider."""
    from team_memory.reranker.ollama_llm_provider import OllamaLLMRerankerProvider

    config = RerankerConfig(provider="ollama_llm")
    llm_config = LLMConfig(model="test-model", base_url="http://localhost:11434")
    reranker = create_reranker(config, llm_config)
    assert isinstance(reranker, OllamaLLMRerankerProvider)
    assert reranker.provider_name == "ollama_llm(test-model)"


def test_factory_fallback_unknown_provider():
    """Factory should fall back to noop for unknown providers."""
    config = RerankerConfig.__new__(RerankerConfig)
    object.__setattr__(config, "provider", "unknown_provider")
    object.__setattr__(config, "ollama_llm", RerankerConfig().ollama_llm)
    object.__setattr__(config, "cross_encoder", RerankerConfig().cross_encoder)
    object.__setattr__(config, "jina", RerankerConfig().jina)
    reranker = create_reranker(config)
    assert isinstance(reranker, NoopRerankerProvider)


# ======================== Custom Provider ========================


class MockRerankerProvider(RerankerProvider):
    """Test that custom providers can be created."""

    async def rank(self, query, documents, top_k=10):
        return [
            RerankResult(index=i, score=10.0 - i, text=doc)
            for i, doc in enumerate(documents[:top_k])
        ]


@pytest.mark.asyncio
async def test_custom_provider():
    """Custom reranker should work through the base class interface."""
    reranker = MockRerankerProvider()
    docs = ["alpha", "beta", "gamma"]
    results = await reranker.rank("test query", docs)
    assert len(results) == 3
    assert results[0].score == 10.0
    assert results[0].text == "alpha"

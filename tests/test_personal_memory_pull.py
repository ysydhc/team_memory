"""Unit tests for personal memory pull (no DB): anonymous returns empty."""

from __future__ import annotations

import pytest

from team_memory.services.personal_memory import PersonalMemoryService
from tests.conftest import MockEmbeddingProvider


@pytest.fixture
def mock_embedding():
    return MockEmbeddingProvider(dimension=768)


@pytest.mark.asyncio
async def test_pull_anonymous_returns_empty(mock_embedding):
    """Anonymous user: pull returns [] (no generic). No DB required."""
    svc = PersonalMemoryService(embedding_provider=mock_embedding, db_url="")
    out = await svc.pull(user_id=None)
    assert out == []
    out = await svc.pull(user_id="anonymous")
    assert out == []

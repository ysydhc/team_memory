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

@pytest.mark.asyncio
async def test_build_profile_anonymous_empty(mock_embedding):
    svc = PersonalMemoryService(embedding_provider=mock_embedding, db_url="")
    p = await svc.build_profile_for_user(None)
    assert p == {"static": [], "dynamic": []}
    p2 = await svc.build_profile_for_user("anonymous")
    assert p2 == {"static": [], "dynamic": []}


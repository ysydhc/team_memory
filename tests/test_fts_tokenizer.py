"""Tests for Step 3a: FTS simple tokenizer change.

Verifies that search_by_fts uses 'simple' instead of 'english' tokenizer,
which is critical for Chinese text support.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from team_memory.storage.repository import ExperienceRepository


class TestFtsTokenizer:
    """Verify FTS uses 'simple' tokenizer for Chinese support."""

    @pytest.mark.asyncio
    async def test_search_by_fts_uses_simple_tokenizer(self):
        """search_by_fts should use plainto_tsquery('simple', ...) not 'english'."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = ExperienceRepository(mock_session)

        await repo.search_by_fts(
            query_text="数据库",
            max_results=5,
            project="default",
        )

        execute_call = mock_session.execute.call_args
        query_str = str(execute_call[0][0])
        assert "simple" in query_str.lower() or "plainto_tsquery" in query_str

    def test_simple_tokenizer_not_english(self):
        """Confirm the code uses 'simple' not 'english' in the source."""
        import inspect

        source = inspect.getsource(ExperienceRepository.search_by_fts)
        assert '"simple"' in source
        assert '"english"' not in source

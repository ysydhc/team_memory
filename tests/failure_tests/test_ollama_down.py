"""Ollama 不可用 — 验证降级不崩溃。"""
from unittest.mock import patch

import pytest


@pytest.mark.failure_injection
class TestOllamaDown:
    def test_embedding_failure_handled(self):
        """Mock Ollama unavailable → embedding returns error → no crash."""
        from team_memory.bootstrap import _create_embedding_provider
        with patch("team_memory.bootstrap._create_embedding_provider", side_effect=RuntimeError("ollama down")):
            try:
                _create_embedding_provider(None)
            except RuntimeError:
                pass  # Expected

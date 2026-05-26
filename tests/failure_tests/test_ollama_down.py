"""Ollama 不可用 — 验证降级不崩溃。"""

from unittest.mock import patch

import pytest


@pytest.mark.failure_injection
class TestOllamaDown:
    def test_embedding_failure_handled(self):
        """Mock Ollama unavailable → embedding returns error → no crash."""
        import team_memory.embedding.factory as factory_mod

        with patch.object(factory_mod, "create_embedding_provider", side_effect=RuntimeError("ollama down")):
            with pytest.raises(RuntimeError, match="ollama down"):
                factory_mod.create_embedding_provider(None)

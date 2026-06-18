"""Tests for embedding dimension configuration and startup validation."""

from __future__ import annotations

import logging

import pytest

from team_memory.config import load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Reset global settings before/after each test."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture()
def _ensure_propagate() -> None:
    """Ensure the team_memory logger propagates so caplog can capture records.

    Other tests (e.g. test_bootstrap._configure_logging) may set
    propagate=False on the root 'team_memory' logger, which prevents
    caplog from seeing child-logger output.
    """
    tm_logger = logging.getLogger("team_memory")
    original_propagate = tm_logger.propagate
    original_handlers = tm_logger.handlers[:]
    tm_logger.propagate = True
    # Remove any non-caplog handlers that would swallow records
    for h in tm_logger.handlers[:]:
        tm_logger.removeHandler(h)
    yield
    tm_logger.propagate = original_propagate
    for h in tm_logger.handlers[:]:
        tm_logger.removeHandler(h)
    for h in original_handlers:
        tm_logger.addHandler(h)


class TestDBVectorDimConstant:
    """Verify DB_VECTOR_DIM constant in models.py."""

    def test_db_vector_dim_value(self) -> None:
        """DB_VECTOR_DIM should be 4096 (matching qwen3-embedding-8b schema)."""
        from team_memory.storage.models import DB_VECTOR_DIM

        assert DB_VECTOR_DIM == 4096

    def test_db_vector_dim_is_int(self) -> None:
        """DB_VECTOR_DIM should be an integer."""
        from team_memory.storage.models import DB_VECTOR_DIM

        assert isinstance(DB_VECTOR_DIM, int)


class TestValidateEmbeddingDimension:
    """Verify _validate_embedding_dimension startup check."""

    def test_no_warning_when_dimensions_match(self, caplog: pytest.LogCaptureFixture, _ensure_propagate: None) -> None:
        """No warning should be emitted when config dimension == DB_VECTOR_DIM."""
        from team_memory.bootstrap import _validate_embedding_dimension
        from team_memory.storage.models import DB_VECTOR_DIM

        settings = load_settings()
        # Configure provider to exactly match the DB schema dimension.
        settings.embedding.provider = "openai"
        settings.embedding.openai.dimension = DB_VECTOR_DIM

        with caplog.at_level(logging.WARNING, logger="team_memory.bootstrap"):
            _validate_embedding_dimension(settings)

        assert "dimension mismatch" not in caplog.text.lower()

    def test_warns_on_dimension_mismatch(self, caplog: pytest.LogCaptureFixture, _ensure_propagate: None) -> None:
        """A warning should be emitted when config dimension != DB_VECTOR_DIM."""
        from team_memory.bootstrap import _validate_embedding_dimension
        from team_memory.storage.models import DB_VECTOR_DIM

        settings = load_settings()
        # Simulate switching to OpenAI with 1536 dims (differs from DB schema)
        settings.embedding.provider = "openai"
        settings.embedding.openai.dimension = 1536

        with caplog.at_level(logging.WARNING, logger="team_memory.bootstrap"):
            _validate_embedding_dimension(settings)

        assert "dimension mismatch" in caplog.text.lower()
        assert "1536" in caplog.text
        assert str(DB_VECTOR_DIM) in caplog.text

    def test_warns_on_local_dimension_mismatch(self, caplog: pytest.LogCaptureFixture, _ensure_propagate: None) -> None:
        """Warning should fire for local provider with non-1024 dimension."""
        from team_memory.bootstrap import _validate_embedding_dimension

        settings = load_settings()
        settings.embedding.provider = "local"
        settings.embedding.local.dimension = 512

        with caplog.at_level(logging.WARNING, logger="team_memory.bootstrap"):
            _validate_embedding_dimension(settings)

        assert "dimension mismatch" in caplog.text.lower()
        assert "512" in caplog.text

    def test_no_warning_when_custom_provider_matches(
        self, caplog: pytest.LogCaptureFixture, _ensure_propagate: None
    ) -> None:
        """No warning when a non-default provider is configured to match the DB schema dims."""
        from team_memory.bootstrap import _validate_embedding_dimension
        from team_memory.storage.models import DB_VECTOR_DIM

        settings = load_settings()
        settings.embedding.provider = "openai"
        settings.embedding.openai.dimension = DB_VECTOR_DIM

        with caplog.at_level(logging.WARNING, logger="team_memory.bootstrap"):
            _validate_embedding_dimension(settings)

        assert "dimension mismatch" not in caplog.text.lower()

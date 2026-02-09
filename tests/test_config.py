"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from team_doc.config import Settings, load_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset():
    """Reset global settings before each test."""
    reset_settings()
    yield
    reset_settings()


class TestSettings:
    """Test Settings model defaults."""

    def test_default_database_url(self):
        settings = Settings()
        assert "postgresql+asyncpg" in settings.database.url

    def test_default_embedding_provider(self):
        settings = Settings()
        assert settings.embedding.provider == "openai"

    def test_embedding_dimension_openai(self):
        settings = Settings()
        assert settings.embedding.dimension == 1536

    def test_embedding_dimension_local(self):
        settings = Settings(
            embedding={"provider": "local", "local": {"dimension": 1024}}
        )
        assert settings.embedding.dimension == 1024

    def test_default_auth_type(self):
        settings = Settings()
        assert settings.auth.type == "api_key"


class TestLoadSettings:
    """Test loading settings from YAML file."""

    def test_load_from_yaml(self, tmp_path: Path):
        config = {
            "database": {"url": "postgresql+asyncpg://test:test@localhost/test_db"},
            "embedding": {"provider": "local"},
            "auth": {"type": "none"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(config_file)
        assert settings.database.url == "postgresql+asyncpg://test:test@localhost/test_db"
        assert settings.embedding.provider == "local"
        assert settings.auth.type == "none"

    def test_load_nonexistent_file_uses_defaults(self):
        settings = load_settings("/nonexistent/config.yaml")
        assert settings.database.url is not None
        assert settings.embedding.provider == "openai"

    def test_env_var_resolution_in_yaml(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        config = {
            "embedding": {
                "provider": "openai",
                "openai": {"api_key": "${TEST_API_KEY}"},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(config_file)
        assert settings.embedding.openai.api_key == "sk-test-123"

    def test_env_var_missing_resolves_to_empty(self, tmp_path: Path):
        config = {
            "embedding": {
                "openai": {"api_key": "${NONEXISTENT_VAR}"},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(config_file)
        assert settings.embedding.openai.api_key == ""

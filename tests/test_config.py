"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from team_memory.config import Settings, load_settings, reset_settings


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
        assert settings.embedding.provider == "ollama"

    def test_embedding_dimension_ollama(self):
        """Default provider is Ollama with 768 dimensions."""
        settings = Settings()
        assert settings.embedding.dimension == 768

    def test_embedding_dimension_openai(self):
        settings = Settings(
            embedding={"provider": "openai", "openai": {"dimension": 1536}}
        )
        assert settings.embedding.dimension == 1536

    def test_embedding_dimension_local(self):
        settings = Settings(
            embedding={"provider": "local", "local": {"dimension": 1024}}
        )
        assert settings.embedding.dimension == 1024

    def test_ollama_config_defaults(self):
        """Ollama config has sensible defaults."""
        settings = Settings()
        assert settings.embedding.ollama.model == "nomic-embed-text"
        assert settings.embedding.ollama.base_url == "http://localhost:11434"
        assert settings.embedding.ollama.dimension == 768

    def test_default_auth_type(self):
        settings = Settings()
        assert settings.auth.type == "db_api_key"


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
        assert settings.embedding.provider == "ollama"

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

    def test_config_yaml_wins_over_minimal_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"auth": {"api_key": "from-main-yaml"}})
        )
        (tmp_path / "config.minimal.yaml").write_text(
            yaml.dump({"auth": {"api_key": "from-minimal-yaml"}})
        )
        monkeypatch.chdir(tmp_path)

        settings = load_settings()
        assert settings.auth.api_key == "from-main-yaml"

    def test_minimal_overlay_can_be_enabled_explicitly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"auth": {"api_key": "from-main-yaml"}})
        )
        (tmp_path / "config.minimal.yaml").write_text(
            yaml.dump({"auth": {"api_key": "from-minimal-yaml"}})
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TEAM_MEMORY_ENABLE_MINIMAL_OVERLAY", "1")

        settings = load_settings()
        assert settings.auth.api_key == "from-minimal-yaml"

    def test_environment_variables_override_yaml_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"web": {"port": 9111}}))
        monkeypatch.setenv("TEAM_MEMORY_WEB__PORT", "9555")

        settings = load_settings(config_file)
        assert settings.web.port == 9555

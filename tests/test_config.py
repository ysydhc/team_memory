"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from team_memory.config import AuthConfig, Settings, load_settings, reset_settings


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
        settings = Settings(embedding={"provider": "openai", "openai": {"dimension": 1536}})
        assert settings.embedding.dimension == 1536

    def test_embedding_dimension_local(self):
        settings = Settings(embedding={"provider": "local", "local": {"dimension": 1024}})
        assert settings.embedding.dimension == 1024

    def test_ollama_config_defaults(self):
        """Ollama config has sensible defaults."""
        settings = Settings()
        assert settings.embedding.ollama.model == "nomic-embed-text:latest"
        assert settings.embedding.ollama.base_url == "http://localhost:11434"
        assert settings.embedding.ollama.dimension == 768

    def test_default_auth_type(self):
        settings = Settings()
        assert settings.auth.type == "db_api_key"


def test_auth_config_has_default_admin_password_and_session_secret():
    c = AuthConfig(default_admin_password="x", session_secret="y")
    assert c.default_admin_password == "x"
    assert c.session_secret == "y"


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

    def test_team_memory_env_selects_config_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        (tmp_path / "config.development.yaml").write_text(
            yaml.dump({"auth": {"api_key": "dev-key"}, "web": {"port": 9111}})
        )
        (tmp_path / "config.production.yaml").write_text(
            yaml.dump({"auth": {"api_key": "prod-key"}, "web": {"port": 9200}})
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEAM_MEMORY_CONFIG_PATH", raising=False)

        monkeypatch.setenv("TEAM_MEMORY_ENV", "development")
        assert load_settings().auth.api_key == "dev-key"
        monkeypatch.setenv("TEAM_MEMORY_ENV", "production")
        reset_settings()
        assert load_settings().auth.api_key == "prod-key"
        monkeypatch.setenv("TEAM_MEMORY_ENV", "test")
        reset_settings()
        assert load_settings().auth.api_key == "dev-key"

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


def test_logging_config_defaults():
    from team_memory.config import load_settings

    s = load_settings()
    assert hasattr(s, "logging")
    lg = s.logging
    assert lg.log_io_enabled is False
    assert lg.log_io_detail == "mcp"
    assert lg.log_io_truncate == 300
    assert lg.log_file_enabled is False
    assert lg.log_file_max_bytes == 10 * 1024 * 1024


def test_web_config_security_defaults():
    """WebConfig has request body size limit, rate limit, and text field limits."""
    from team_memory.config import WebConfig

    wc = WebConfig()
    assert wc.max_request_body_bytes == 20_971_520  # 20 MB
    assert wc.max_text_field_bytes == 64_000  # 64 KB
    assert wc.max_raw_conversation_bytes == 640_000  # 640 KB
    assert wc.rate_limit_per_minute == 200


def test_mcp_config_security_defaults():
    """MCPConfig has content length and tags limits."""
    from team_memory.config import MCPConfig

    mc = MCPConfig()
    assert mc.max_content_chars == 200_000
    assert mc.max_tags == 20
    assert mc.max_tag_length == 50

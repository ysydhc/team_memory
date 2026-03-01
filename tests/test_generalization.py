"""Tests for the generalization platform features.

Covers:
  - Schema presets and SchemaRegistry
  - CustomSchemaConfig merge logic
  - AI behavior config
  - Prompt loader (file + builtin + variable substitution + ai_behavior)
  - Webhook config model
  - Extension framework loading
  - Web API endpoints (/schema, /schema/generate, /schema/presets, /config/webhooks)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from team_memory.config import (
    AIBehaviorConfig,
    CategoryDef,
    CustomSchemaConfig,
    ExperienceTypeDef,
    LLMConfig,
    Settings,
    StructuredFieldDef,
    WebhookItemConfig,
    load_settings,
    reset_settings,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset global singletons before each test."""
    reset_settings()
    from team_memory.schemas import reset_schema_registry

    reset_schema_registry()
    yield
    reset_settings()
    from team_memory.schemas import reset_schema_registry

    reset_schema_registry()


# ====================================================================
# Schema Presets
# ====================================================================


class TestSchemaPresets:
    """Test built-in schema preset packs."""

    def test_list_presets_returns_four(self):
        from team_memory.schema_presets import list_presets

        presets = list_presets()
        assert len(presets) == 4
        assert "software-dev" in presets
        assert "data-engineering" in presets
        assert "devops" in presets
        assert "general" in presets

    def test_get_preset_software_dev(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("software-dev")
        type_ids = [t.id for t in pack["experience_types"]]
        assert "general" in type_ids
        assert "bugfix" in type_ids
        assert "feature" in type_ids
        assert "tech_design" in type_ids
        assert "incident" in type_ids
        assert "best_practice" in type_ids
        assert "learning" in type_ids

    def test_get_preset_data_engineering(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("data-engineering")
        type_ids = [t.id for t in pack["experience_types"]]
        assert "data_quality" in type_ids
        assert "pipeline_failure" in type_ids
        assert "schema_change" in type_ids

    def test_get_preset_devops(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("devops")
        type_ids = [t.id for t in pack["experience_types"]]
        assert "incident" in type_ids
        assert "deployment" in type_ids
        assert "runbook" in type_ids
        assert "postmortem" in type_ids

    def test_get_preset_general_minimal(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("general")
        type_ids = [t.id for t in pack["experience_types"]]
        assert "general" in type_ids
        assert "note" in type_ids
        assert "decision" in type_ids
        assert "action_item" in type_ids
        # No severity for general preset
        assert pack["severity_levels"] == []

    def test_unknown_preset_falls_back_to_software_dev(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("nonexistent")
        sw_pack = get_preset("software-dev")
        assert len(pack["experience_types"]) == len(sw_pack["experience_types"])

    def test_preset_types_have_structured_fields(self):
        """Types like bugfix should have structured_fields."""
        from team_memory.schema_presets import get_preset

        pack = get_preset("software-dev")
        bugfix = next(t for t in pack["experience_types"] if t.id == "bugfix")
        assert len(bugfix.structured_fields) > 0
        field_names = [f.name for f in bugfix.structured_fields]
        assert "reproduction_steps" in field_names

    def test_preset_types_have_progress_states(self):
        from team_memory.schema_presets import get_preset

        pack = get_preset("software-dev")
        bugfix = next(t for t in pack["experience_types"] if t.id == "bugfix")
        assert bugfix.progress_states == ["open", "investigating", "fixed", "verified"]


# ====================================================================
# SchemaRegistry
# ====================================================================


class TestSchemaRegistry:
    """Test SchemaRegistry merge logic and validation methods."""

    def test_default_registry_matches_software_dev(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        type_ids = [t.id for t in reg.get_experience_types()]
        assert "bugfix" in type_ids
        assert "general" in type_ids

    def test_custom_type_appended(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(
            preset="software-dev",
            experience_types=[
                ExperienceTypeDef(id="custom_review", label="代码审查"),
            ],
        )
        reg = SchemaRegistry(config)
        assert reg.is_valid_type("custom_review")
        assert reg.is_valid_type("bugfix")  # preset type still exists

    def test_custom_type_overrides_preset(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(
            preset="software-dev",
            experience_types=[
                ExperienceTypeDef(id="bugfix", label="自定义 Bug 修复"),
            ],
        )
        reg = SchemaRegistry(config)
        t = reg.get_type_def("bugfix")
        assert t is not None
        assert t.label == "自定义 Bug 修复"

    def test_custom_category_appended(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(
            categories=[CategoryDef(id="data_pipeline", label="数据管道")],
        )
        reg = SchemaRegistry(config)
        assert reg.is_valid_category("data_pipeline")
        assert reg.is_valid_category("frontend")  # preset cat still there

    def test_custom_severity_overrides(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(severity_levels=["S1", "S2", "S3"])
        reg = SchemaRegistry(config)
        assert reg.is_valid_severity("S1")
        assert not reg.is_valid_severity("P0")  # preset severity replaced

    def test_empty_severity_uses_preset(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(preset="software-dev", severity_levels=[])
        reg = SchemaRegistry(config)
        assert reg.is_valid_severity("P0")

    def test_general_preset_no_severity(self):
        from team_memory.schemas import SchemaRegistry

        config = CustomSchemaConfig(preset="general")
        reg = SchemaRegistry(config)
        assert reg.get_severity_levels() == []

    def test_is_valid_type_returns_false_for_unknown(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        assert not reg.is_valid_type("nonexistent")

    def test_get_progress_states(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        states = reg.get_progress_states("bugfix")
        assert states == ["open", "investigating", "fixed", "verified"]
        assert reg.get_progress_states("general") == []

    def test_get_default_progress(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        assert reg.get_default_progress("bugfix", has_solution=False) == "open"
        assert reg.get_default_progress("bugfix", has_solution=True) == "fixed"
        assert reg.get_default_progress("general", has_solution=False) is None

    def test_to_dict_export(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        d = reg.to_dict()
        assert "experience_types" in d
        assert "categories" in d
        assert "severity_levels" in d
        assert "preset" in d
        assert isinstance(d["experience_types"], list)
        assert len(d["experience_types"]) > 0
        # Each type has structured_fields key
        for t in d["experience_types"]:
            assert "structured_fields" in t

    def test_types_for_prompt(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        prompt_text = reg.types_for_prompt()
        assert "bugfix" in prompt_text
        assert "general" in prompt_text

    def test_categories_for_prompt(self):
        from team_memory.schemas import SchemaRegistry

        reg = SchemaRegistry()
        prompt_text = reg.categories_for_prompt()
        assert "frontend" in prompt_text
        assert "/" in prompt_text  # slash-separated

    def test_init_and_get_schema_registry(self):
        from team_memory.schemas import get_schema_registry, init_schema_registry

        reg1 = init_schema_registry(CustomSchemaConfig(preset="devops"))
        reg2 = get_schema_registry()
        assert reg1 is reg2
        assert reg2.is_valid_type("deployment")

    def test_different_presets_have_different_types(self):
        from team_memory.schemas import SchemaRegistry

        sw = SchemaRegistry(CustomSchemaConfig(preset="software-dev"))
        de = SchemaRegistry(CustomSchemaConfig(preset="data-engineering"))
        sw_ids = {t.id for t in sw.get_experience_types()}
        de_ids = {t.id for t in de.get_experience_types()}
        assert "feature" in sw_ids
        assert "feature" not in de_ids
        assert "data_quality" in de_ids
        assert "data_quality" not in sw_ids


# ====================================================================
# Config model tests
# ====================================================================


class TestCustomSchemaConfig:
    """Test the Pydantic config models."""

    def test_default_preset(self):
        cfg = CustomSchemaConfig()
        assert cfg.preset == "software-dev"
        assert cfg.experience_types == []
        assert cfg.severity_levels == []

    def test_structured_field_def(self):
        sf = StructuredFieldDef(name="env", label="环境信息")
        assert sf.type == "text"
        assert sf.required is False

    def test_experience_type_def(self):
        t = ExperienceTypeDef(
            id="my_type",
            label="My Type",
            severity=True,
            progress_states=["a", "b"],
            structured_fields=[StructuredFieldDef(name="f1", label="Field 1")],
        )
        assert t.severity is True
        assert len(t.structured_fields) == 1

    def test_settings_has_custom_schema(self):
        s = Settings()
        assert isinstance(s.custom_schema, CustomSchemaConfig)
        assert s.custom_schema.preset == "software-dev"

    def test_settings_has_ai_behavior(self):
        s = Settings()
        assert isinstance(s.ai_behavior, AIBehaviorConfig)
        assert s.ai_behavior.output_language == "zh-CN"

    def test_settings_has_webhooks(self):
        s = Settings()
        assert isinstance(s.webhooks, list)
        assert len(s.webhooks) == 0

    def test_webhook_item_config(self):
        w = WebhookItemConfig(url="https://example.com", events=["experience.created"])
        assert w.active is True
        assert w.secret == ""

    def test_llm_config_has_prompt_dir(self):
        s = Settings()
        assert s.llm.prompt_dir is None


class TestAIBehaviorConfig:
    """Test AIBehaviorConfig defaults and serialization."""

    def test_defaults(self):
        cfg = AIBehaviorConfig()
        assert cfg.output_language == "zh-CN"
        assert cfg.detail_level == "detailed"
        assert "root_cause" in cfg.focus_areas
        assert cfg.custom_instructions == ""

    def test_custom_values(self):
        cfg = AIBehaviorConfig(
            output_language="en",
            detail_level="concise",
            focus_areas=["architecture"],
            custom_instructions="Focus on API design",
        )
        assert cfg.output_language == "en"
        assert cfg.custom_instructions == "Focus on API design"


class TestLoadSettingsWithSchema:
    """Test that config.yaml schema sections load correctly."""

    def test_load_custom_schema_from_yaml(self, tmp_path: Path):
        import yaml

        config = {
            "custom_schema": {
                "preset": "devops",
                "experience_types": [
                    {"id": "my_custom", "label": "自定义", "severity": True}
                ],
            },
            "ai_behavior": {
                "output_language": "en",
                "detail_level": "concise",
            },
            "webhooks": [
                {"url": "https://test.com/hook", "events": ["experience.created"]}
            ],
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(config_file)
        assert settings.custom_schema.preset == "devops"
        assert len(settings.custom_schema.experience_types) == 1
        assert settings.custom_schema.experience_types[0].id == "my_custom"
        assert settings.ai_behavior.output_language == "en"
        assert len(settings.webhooks) == 1
        assert settings.webhooks[0].url == "https://test.com/hook"


# ====================================================================
# Prompt Loader
# ====================================================================


class TestPromptLoader:
    """Test load_prompt() with file loading, variable substitution, ai_behavior."""

    def test_builtin_prompt_loads(self):
        from team_memory.schemas import init_schema_registry
        from team_memory.services.llm_parser import load_prompt

        init_schema_registry()
        prompt = load_prompt("parse_single")
        assert len(prompt) > 100
        # After variable substitution, should not contain raw placeholders
        assert "{{experience_types}}" not in prompt

    def test_builtin_suggest_type_prompt(self):
        from team_memory.schemas import init_schema_registry
        from team_memory.services.llm_parser import load_prompt

        init_schema_registry()
        prompt = load_prompt("suggest_type")
        assert "general" in prompt  # type should be listed

    def test_builtin_summary_prompt(self):
        from team_memory.services.llm_parser import load_prompt

        prompt = load_prompt("summary")
        assert "摘要" in prompt

    def test_unknown_prompt_returns_behavior_only(self):
        from team_memory.services.llm_parser import load_prompt

        # Unknown prompt name returns empty base + ai_behavior block
        prompt = load_prompt("nonexistent_prompt_name", ai_behavior=AIBehaviorConfig())
        # Should have the behavior block but no real prompt content
        assert "团队定制要求" in prompt or prompt.strip() == ""

    def test_file_prompt_loads_from_dir(self, tmp_path: Path):
        from team_memory.services.llm_parser import load_prompt

        prompt_file = tmp_path / "test_prompt.md"
        prompt_file.write_text("This is a custom {{experience_types}} prompt.")

        from team_memory.schemas import init_schema_registry

        init_schema_registry()

        llm_config = LLMConfig(prompt_dir=str(tmp_path))
        result = load_prompt("test_prompt", llm_config=llm_config)
        assert "This is a custom" in result
        assert "{{experience_types}}" not in result  # variables replaced

    def test_ai_behavior_injected(self):
        from team_memory.schemas import init_schema_registry
        from team_memory.services.llm_parser import load_prompt

        init_schema_registry()
        behavior = AIBehaviorConfig(
            output_language="en",
            custom_instructions="Focus on data quality",
        )
        prompt = load_prompt("parse_single", ai_behavior=behavior)
        assert "团队定制要求" in prompt
        assert "English" in prompt
        assert "Focus on data quality" in prompt

    def test_variable_substitution_includes_schema_types(self):
        from team_memory.schemas import init_schema_registry
        from team_memory.services.llm_parser import load_prompt

        config = CustomSchemaConfig(
            preset="devops",
            experience_types=[
                ExperienceTypeDef(id="custom_runbook", label="自定义手册"),
            ],
        )
        init_schema_registry(config)

        prompt = load_prompt("parse_single", ai_behavior=AIBehaviorConfig())
        assert "deployment" in prompt  # from devops preset
        assert "custom_runbook" in prompt  # from custom type


# ====================================================================
# Extension Framework
# ====================================================================


class TestExtensionFramework:
    """Test the extension loading mechanism."""

    def test_load_from_nonexistent_dir(self):
        from team_memory.extensions import ExtensionContext, load_extensions

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir="/nonexistent/path")
        assert count == 0

    def test_load_from_empty_dir(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 0

    def test_load_valid_extension(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions

        ext_file = tmp_path / "my_ext.py"
        ext_file.write_text(
            textwrap.dedent("""\
            def register(ctx):
                ctx.logger.info("test extension loaded")
            """)
        )

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 1
        assert "my_ext" in ctx._loaded_extensions

    def test_skip_underscore_files(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions

        (tmp_path / "_hidden.py").write_text("def register(ctx): pass")
        (tmp_path / "__init__.py").write_text("")

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 0

    def test_skip_file_without_register(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions

        (tmp_path / "no_register.py").write_text("x = 1\n")

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 0

    def test_failing_extension_does_not_crash(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions

        (tmp_path / "bad_ext.py").write_text(
            textwrap.dedent("""\
            def register(ctx):
                raise RuntimeError("intentional failure")
            """)
        )
        (tmp_path / "good_ext.py").write_text("def register(ctx): pass")

        ctx = ExtensionContext()
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 1  # good_ext loaded, bad_ext failed
        assert "good_ext" in ctx._loaded_extensions

    def test_extension_receives_event_bus(self, tmp_path: Path):
        from team_memory.extensions import ExtensionContext, load_extensions
        from team_memory.services.event_bus import EventBus

        (tmp_path / "bus_ext.py").write_text(
            textwrap.dedent("""\
            captured = []
            def register(ctx):
                async def handler(payload):
                    captured.append(payload)
                ctx.event_bus.on("test.event", handler)
            """)
        )

        bus = EventBus()
        ctx = ExtensionContext(event_bus=bus)
        count = load_extensions(ctx, ext_dir=str(tmp_path))
        assert count == 1
        assert bus.handler_count >= 1


# ====================================================================
# Webhook Config Integration
# ====================================================================


class TestWebhookConfig:
    """Test webhook config model and loading."""

    def test_webhook_config_from_yaml(self, tmp_path: Path):
        import yaml

        config = {
            "webhooks": [
                {
                    "url": "https://slack.example.com/hook",
                    "events": ["experience.created"],
                    "secret": "s3cret",
                    "active": True,
                },
                {
                    "url": "https://other.example.com",
                    "events": ["experience.published"],
                    "active": False,
                },
            ]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        settings = load_settings(config_file)
        assert len(settings.webhooks) == 2
        assert settings.webhooks[0].url == "https://slack.example.com/hook"
        assert settings.webhooks[0].secret == "s3cret"
        assert settings.webhooks[1].active is False

    def test_empty_webhooks_by_default(self):
        s = Settings()
        assert s.webhooks == []


# ====================================================================
# Web API Endpoint Tests (Schema)
# ====================================================================


class TestWebSchemaEndpoints:
    """Test schema-related web API endpoints.

    Uses the FastAPI test client via httpx to validate endpoints.
    """

    @pytest.fixture
    def client(self):
        """Create a test client with auth; use context manager so lifespan runs and routes are registered."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi.testclient import TestClient

        from team_memory.auth.provider import User
        from team_memory.schemas import init_schema_registry

        init_schema_registry()

        import team_memory.web.app as app_module
        from team_memory.web.app import app

        settings = Settings(
            auth={"type": "api_key", "api_key": "test-key-123"}
        )
        mock_auth = MagicMock()
        mock_auth.authenticate = AsyncMock(
            return_value=User(name="test", role="admin")
        )
        app_module._settings = settings
        app_module._auth = mock_auth
        app_module._service = None

        mock_ctx = MagicMock()
        mock_ctx.settings = settings
        mock_ctx.service = None
        mock_ctx.auth = mock_auth
        with patch("team_memory.web.app.bootstrap", return_value=mock_ctx), patch(
            "team_memory.web.app.start_background_tasks", new_callable=AsyncMock
        ), patch(
            "team_memory.web.app.stop_background_tasks", new_callable=AsyncMock
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                c.headers["Authorization"] = "Bearer test-key-123"
                yield c

        app_module._settings = None
        app_module._auth = None
        app_module._service = None

    def test_get_schema(self, client):
        resp = client.get("/api/v1/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "experience_types" in data
        assert "categories" in data
        assert "severity_levels" in data
        # Default preset should have bugfix
        type_ids = [t["id"] for t in data["experience_types"]]
        assert "bugfix" in type_ids

    def test_get_schema_presets(self, client):
        resp = client.get("/api/v1/schema/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4
        names = [p["name"] for p in data]
        assert "software-dev" in names
        assert "devops" in names

    def test_put_config_schema(self, client):
        resp = client.put(
            "/api/v1/config/schema",
            json={
                "preset": "devops",
                "experience_types": [],
                "categories": [],
                "severity_levels": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "schema" in data
        type_ids = [t["id"] for t in data["schema"]["experience_types"]]
        assert "deployment" in type_ids

    def test_get_config_webhooks_empty(self, client):
        resp = client.get("/api/v1/config/webhooks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_put_config_webhooks(self, client):
        resp = client.put(
            "/api/v1/config/webhooks",
            json=[
                {
                    "url": "https://test.com",
                    "events": ["experience.created"],
                    "active": True,
                }
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

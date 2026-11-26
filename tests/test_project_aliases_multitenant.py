"""Unit tests for de-hardcoded project aliases, config.toml loading, and multi-tenant isolation."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from team_memory import client_config, project_resolver
from team_memory.storage.tenant_context import tenant_scope
from team_memory.utils.project import normalize_project_name


@pytest.fixture(autouse=True)
def _reset_alias_state():
    """Ensure clean alias registries around every test."""
    orig_global = dict(project_resolver._alias_registry)
    orig_tenant_reg = {k: dict(v) for k, v in project_resolver._tenant_alias_registry.items()}
    orig_tenant_short = {k: dict(v) for k, v in project_resolver._tenant_shorthand_aliases.items()}
    orig_cfg_short = project_resolver._config_shorthand_aliases
    orig_cfg_map = project_resolver._config_alias_map

    project_resolver._alias_registry.clear()
    project_resolver.clear_tenant_aliases()
    project_resolver._config_shorthand_aliases = {}
    project_resolver._config_alias_map = {}

    yield

    project_resolver._alias_registry.clear()
    project_resolver._alias_registry.update(orig_global)
    project_resolver._tenant_alias_registry.clear()
    project_resolver._tenant_alias_registry.update(orig_tenant_reg)
    project_resolver._tenant_shorthand_aliases.clear()
    project_resolver._tenant_shorthand_aliases.update(orig_tenant_short)
    project_resolver._config_shorthand_aliases = orig_cfg_short
    project_resolver._config_alias_map = orig_cfg_map


class TestBuiltinDeHardcoding:
    """Verify personal/company business projects are removed from public source code."""

    def test_builtin_aliases_only_contain_team_memory_self_aliases(self) -> None:
        builtins = project_resolver._BUILTIN_SHORTHAND_ALIASES
        assert set(builtins.keys()) == {"team_doc", "team_memory"}
        for removed in (
            "adx-sdk-android",
            "adx-sdk-ios",
            "inapp-dsp",
            "inapp-adx",
            "adx-neo",
            "inapp-jump",
            "agent-common",
            "inapp-game-dollshooter",
            "phone_pilot",
        ):
            assert removed not in builtins


class TestConfigTomlProjectAliases:
    """Verify ~/.config/tm/config.toml [project_aliases] loading & ClientConfig integration."""

    def test_reload_config_aliases_from_toml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            """
[server]
url = "http://127.0.0.1:9111"

[project_aliases]
adx-sdk-android = ["adx-sdk-android", "adx_sdk_android", "adx_android"]
inapp-game-dollshooter = ["dollshooter", "doll_shooter"]
""".strip(),
            encoding="utf-8",
        )

        loaded = project_resolver.reload_config_aliases(cfg_file)
        assert "adx-sdk-android" in loaded
        assert "dollshooter" in loaded["inapp-game-dollshooter"]

        assert project_resolver.resolve_alias("adx_android") == "adx_sdk_android"
        assert project_resolver.resolve_alias("dollshooter") == "inapp_game_dollshooter"

        proj, is_explicit = project_resolver.resolve_multi_root_project(
            workspace_roots=[],
            prompt="帮我排查 dollshooter 结算页的问题",
        )
        assert proj == "inapp_game_dollshooter"
        assert is_explicit is True

    def test_client_config_load_and_sources_include_project_aliases(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            """
[project_aliases]
my-mobile-sdk = ["msdk", "mobile_sdk"]
""".strip(),
            encoding="utf-8",
        )

        with patch.object(client_config, "_CONFIG_FILE", cfg_file):
            cfg = client_config.ClientConfig.load()
            assert cfg.project_aliases == {"my-mobile-sdk": ["msdk", "mobile_sdk"]}

            cfg2, sources = client_config.ClientConfig.load_with_sources()
            assert cfg2.project_aliases == {"my-mobile-sdk": ["msdk", "mobile_sdk"]}
            assert sources["project_aliases"] == "config.toml"
            assert cfg2.to_dict()["project_aliases"] == {"my-mobile-sdk": ["msdk", "mobile_sdk"]}


class TestMultiTenantAliasIsolation:
    """Verify strict per-tenant alias isolation across resolve_alias, multi-root, and utils.project."""

    def test_tenant_aliases_are_strictly_isolated(self) -> None:
        tenant_a = uuid.UUID("11111111-1111-4111-8111-111111111111")
        tenant_b = uuid.UUID("22222222-2222-4222-8222-222222222222")

        project_resolver.register_tenant_aliases(
            tenant_a,
            {"alpha-payments": ["pay", "alpha_pay"]},
        )
        project_resolver.register_tenant_aliases(
            tenant_b,
            {"beta-checkout": ["pay", "beta_pay"]},
        )

        # Same alias "pay" resolves to completely different canonical projects per tenant
        assert project_resolver.resolve_alias("pay", tenant_id=tenant_a) == "alpha_payments"
        assert project_resolver.resolve_alias("pay", tenant_id=tenant_b) == "beta_checkout"

        # Tenant A's private alias must NOT leak to Tenant B or unscoped callers
        assert project_resolver.resolve_alias("alpha_pay", tenant_id=tenant_a) == "alpha_payments"
        assert project_resolver.resolve_alias("alpha_pay", tenant_id=tenant_b) == "alpha_pay"
        assert project_resolver.resolve_alias("alpha_pay") == "alpha_pay"

    def test_utils_normalize_project_name_respects_tenant_scope(self) -> None:
        tenant_a = uuid.UUID("33333333-3333-4333-8333-333333333333")
        tenant_b = uuid.UUID("44444444-4444-4444-8444-444444444444")

        project_resolver.register_tenant_aliases(tenant_a, {"sdk-core": ["core_alias"]})
        project_resolver.register_tenant_aliases(tenant_b, {"web-core": ["core_alias"]})

        with tenant_scope(tenant_a):
            assert normalize_project_name("core_alias") == "sdk_core"

        with tenant_scope(tenant_b):
            assert normalize_project_name("core_alias") == "web_core"

    def test_multi_root_resolution_uses_tenant_shorthand_aliases(self) -> None:
        tenant_a = uuid.UUID("55555555-5555-4555-8555-555555555555")
        tenant_b = uuid.UUID("66666666-6666-4666-8666-666666666666")

        project_resolver.register_tenant_aliases(tenant_a, {"adx-sdk-android": ["adx_android"]})

        proj_a, explicit_a = project_resolver.resolve_multi_root_project(
            workspace_roots=[],
            prompt="修复 adx_android 激励视频加载超时问题",
            tenant_id=tenant_a,
        )
        assert proj_a == "adx_sdk_android"
        assert explicit_a is True

        # Tenant B does not have adx_android configured
        proj_b, explicit_b = project_resolver.resolve_multi_root_project(
            workspace_roots=[],
            prompt="修复 adx_android 激励视频加载超时问题",
            tenant_id=tenant_b,
        )
        assert proj_b != "adx_sdk_android"
        assert explicit_b is False

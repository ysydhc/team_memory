"""Tests for installable catalog service."""

from __future__ import annotations

import json

import pytest

from team_memory.config import InstallableCatalogConfig
from team_memory.services.installable_catalog import (
    InstallableCatalogError,
    InstallableCatalogService,
)


@pytest.mark.asyncio
async def test_local_manifest_list_preview_and_install(tmp_path):
    kb = tmp_path / ".debug" / "knowledge-pack"
    rules = kb / "rules"
    rules.mkdir(parents=True)
    (rules / "demo.mdc").write_text("demo rule content", encoding="utf-8")
    (kb / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "rule.demo",
                        "type": "rule",
                        "name": "demo",
                        "version": "1.0.0",
                        "source": "local",
                        "path": "rules/demo.mdc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cfg = InstallableCatalogConfig(
        sources=["local"],
        local_base_dir=".debug/knowledge-pack",
        target_rules_dir=".cursor/rules",
        target_prompts_dir=".cursor/prompts",
    )
    svc = InstallableCatalogService(cfg, workspace_root=tmp_path)

    items = await svc.list_items()
    assert len(items) == 1
    assert items[0].id == "rule.demo"

    preview = await svc.preview(item_id="rule.demo", source="local")
    assert "demo rule content" in preview["content"]

    install = await svc.install(item_id="rule.demo", source="local")
    assert install["installed"] is True
    target = tmp_path / ".cursor" / "rules" / "demo.mdc"
    assert target.exists()


def test_target_dir_outside_workspace_rejected(tmp_path):
    cfg = InstallableCatalogConfig(
        sources=["local"],
        local_base_dir=".debug/knowledge-pack",
        target_rules_dir="/tmp/outside-rules",
        target_prompts_dir=".cursor/prompts",
    )
    with pytest.raises(InstallableCatalogError):
        InstallableCatalogService(cfg, workspace_root=tmp_path)

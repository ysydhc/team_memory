"""Installable catalog service for rules/prompts.

Supports two sources:
- local catalog under `.debug/knowledge-pack`
- remote registry manifest over HTTP
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field, model_validator

from team_memory.config import InstallableCatalogConfig

logger = logging.getLogger("team_memory.installables")


class InstallableCatalogError(ValueError):
    """Raised when catalog loading/preview/install fails."""


class ManifestItem(BaseModel):
    """One installable entry in a manifest."""

    id: str
    type: str  # rule | prompt
    name: str
    version: str = "1.0.0"
    source: str | None = None
    description: str = ""
    path: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def validate_path_or_url(self) -> "ManifestItem":
        if not self.path and not self.url:
            raise ValueError("Manifest item requires either path or url")
        if self.type not in ("rule", "prompt"):
            raise ValueError("type must be 'rule' or 'prompt'")
        return self


class CatalogManifest(BaseModel):
    """Manifest model for local or registry sources."""

    items: list[ManifestItem] = Field(default_factory=list)


class CatalogItem(BaseModel):
    """Normalized installable item returned by API."""

    id: str
    type: str
    name: str
    version: str
    source: str
    description: str = ""
    path: str | None = None
    url: str | None = None
    file_name: str | None = None


class InstallableCatalogService:
    """Load/preview/install rules and prompts from configured sources."""

    def __init__(
        self,
        config: InstallableCatalogConfig,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._config = config
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._local_base = self._resolve_within_workspace(config.local_base_dir)
        self._target_rules = self._resolve_within_workspace(config.target_rules_dir)
        self._target_prompts = self._resolve_within_workspace(config.target_prompts_dir)

    def _resolve_within_workspace(self, raw_path: str) -> Path:
        """Resolve path and ensure it stays in workspace."""
        p = Path(raw_path)
        resolved = p.resolve() if p.is_absolute() else (self._workspace_root / p).resolve()
        if not self._is_within(self._workspace_root, resolved):
            raise InstallableCatalogError(
                f"Path '{raw_path}' is outside workspace allowlist"
            )
        return resolved

    @staticmethod
    def _is_within(root: Path, target: Path) -> bool:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            return False

    async def list_items(
        self,
        *,
        source: str | None = None,
        item_type: str | None = None,
    ) -> list[CatalogItem]:
        """Return normalized installables from enabled sources."""
        sources = set(self._config.sources or [])
        items: list[CatalogItem] = []

        if "local" in sources and (source in (None, "local")):
            items.extend(await self._load_local_items())

        if "registry" in sources and (source in (None, "registry")):
            items.extend(await self._load_registry_items())

        if item_type:
            items = [x for x in items if x.type == item_type]

        items.sort(key=lambda x: (x.type, x.name.lower(), x.version), reverse=False)
        return items

    async def preview(self, *, item_id: str, source: str | None = None) -> dict:
        """Load preview content for one item."""
        item = await self._find_item(item_id=item_id, source=source)
        content = await self._load_content(item)
        max_preview_chars = 12000
        return {
            "item": item.model_dump(),
            "content": content[:max_preview_chars],
            "truncated": len(content) > max_preview_chars,
        }

    async def install(self, *, item_id: str, source: str | None = None) -> dict:
        """Install one item into configured target directory."""
        item = await self._find_item(item_id=item_id, source=source)
        content = await self._load_content(item)
        target_dir = self._target_rules if item.type == "rule" else self._target_prompts
        target_dir.mkdir(parents=True, exist_ok=True)

        file_name = self._target_file_name(item)
        target_path = (target_dir / file_name).resolve()
        if not self._is_within(target_dir, target_path):
            raise InstallableCatalogError("Target path escapes allowlist directory")

        target_path.write_text(content, encoding="utf-8")
        logger.info(
            "Installed %s (%s) to %s",
            item.id,
            item.source,
            target_path,
        )
        return {
            "installed": True,
            "item": item.model_dump(),
            "target_path": str(target_path),
        }

    async def _find_item(self, *, item_id: str, source: str | None = None) -> CatalogItem:
        items = await self.list_items(source=source)
        for item in items:
            if item.id == item_id:
                return item
        raise InstallableCatalogError(f"Installable '{item_id}' not found")

    async def _load_local_items(self) -> list[CatalogItem]:
        if not self._local_base.exists():
            return []

        manifest_path = self._local_base / "manifest.json"
        if manifest_path.exists():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = CatalogManifest.model_validate(raw)
                items = []
                for m in manifest.items:
                    if m.source and m.source != "local":
                        continue
                    items.append(
                        CatalogItem(
                            id=m.id,
                            type=m.type,
                            name=m.name,
                            version=m.version,
                            source="local",
                            description=m.description,
                            path=m.path,
                            url=m.url,
                            file_name=Path(m.path).name if m.path else None,
                        )
                    )
                return items
            except Exception:
                logger.warning(
                    "Failed to parse local manifest, fallback to auto-discovery",
                    exc_info=True,
                )

        return self._auto_discover_local_items()

    def _auto_discover_local_items(self) -> list[CatalogItem]:
        items: list[CatalogItem] = []
        rules_dir = self._local_base / "rules"
        prompts_dir = self._local_base / "prompts"

        if rules_dir.exists():
            for p in sorted(rules_dir.glob("*.mdc")):
                item_id = f"local-rule-{p.stem}"
                items.append(
                    CatalogItem(
                        id=item_id,
                        type="rule",
                        name=p.stem,
                        version="local",
                        source="local",
                        description=f"Local rule from {p.relative_to(self._workspace_root)}",
                        path=str(p.relative_to(self._local_base)),
                        file_name=p.name,
                    )
                )

        if prompts_dir.exists():
            for p in sorted(prompts_dir.glob("*.md")):
                item_id = f"local-prompt-{p.stem}"
                items.append(
                    CatalogItem(
                        id=item_id,
                        type="prompt",
                        name=p.stem,
                        version="local",
                        source="local",
                        description=f"Local prompt from {p.relative_to(self._workspace_root)}",
                        path=str(p.relative_to(self._local_base)),
                        file_name=p.name,
                    )
                )
        return items

    async def _load_registry_items(self) -> list[CatalogItem]:
        manifest_url = (self._config.registry_manifest_url or "").strip()
        if not manifest_url:
            return []

        timeout = self._config.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(manifest_url)
            resp.raise_for_status()
            payload = resp.json()

        if isinstance(payload, list):
            payload = {"items": payload}
        manifest = CatalogManifest.model_validate(payload)

        items: list[CatalogItem] = []
        for m in manifest.items:
            item_url = m.url
            if not item_url and m.path:
                item_url = urljoin(manifest_url, m.path)
            items.append(
                CatalogItem(
                    id=m.id,
                    type=m.type,
                    name=m.name,
                    version=m.version,
                    source="registry",
                    description=m.description,
                    path=m.path,
                    url=item_url,
                    file_name=Path(m.path).name if m.path else None,
                )
            )
        return items

    async def _load_content(self, item: CatalogItem) -> str:
        if item.source == "local":
            if not item.path:
                raise InstallableCatalogError(f"Local item '{item.id}' has no path")
            src_path = self._resolve_local_content_path(item.path)
            return src_path.read_text(encoding="utf-8")

        if item.source == "registry":
            if not item.url:
                raise InstallableCatalogError(f"Registry item '{item.id}' has no url")
            async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
                resp = await client.get(item.url)
                resp.raise_for_status()
                return resp.text

        raise InstallableCatalogError(f"Unsupported source '{item.source}'")

    def _resolve_local_content_path(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute():
            raise InstallableCatalogError("Absolute local path is not allowed")
        candidate = (self._local_base / raw).resolve()
        if not self._is_within(self._local_base, candidate):
            raise InstallableCatalogError("Local path escapes catalog base directory")
        if not candidate.exists() or not candidate.is_file():
            raise InstallableCatalogError(f"Local source file not found: {relative_path}")
        return candidate

    def _target_file_name(self, item: CatalogItem) -> str:
        ext = ".mdc" if item.type == "rule" else ".md"
        base = item.file_name or item.name or item.id
        base_no_ext = Path(base).stem
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", base_no_ext).strip("-").lower()
        if not slug:
            slug = "installable"
        return f"{slug}{ext}"

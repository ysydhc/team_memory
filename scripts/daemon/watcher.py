"""Obsidian vault file watcher using watchfiles.

Monitors configured vault directories for .md file changes and
indexes them into TeamMemory via TMSink.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from watchfiles import awatch, Change

from daemon.config import DaemonConfig, VaultConfig
from daemon.tm_sink import TMSink

logger = logging.getLogger("tm_daemon.watcher")


async def start_watcher(config: DaemonConfig, sink: TMSink, buf: Any = None) -> None:
    """Watch all configured Obsidian vaults and index changes.

    Runs as a long-lived asyncio task. Cancel to stop.

    Args:
        config: Daemon configuration.
        sink: TMSink instance for draft_save calls.
        buf: Optional DraftBuffer. When provided, successfully saved drafts
             are also recorded in the local SQLite buffer so RefinementWorker
             can pick them up for LLM refinement.
    """
    from daemon.markdown_indexer import MarkdownIndexer

    indexer = MarkdownIndexer()
    paths = []
    for vault in config.obsidian.vaults:
        p = Path(vault.path).expanduser()
        if p.exists():
            paths.append(p)
            logger.info("Watching vault: %s (project=%s)", p, vault.project)
        else:
            logger.warning("Vault path does not exist: %s", p)

    if not paths:
        logger.warning("No Obsidian vaults configured or paths don't exist")
        return

    try:
        async for changes in awatch(*paths, debounce=2000):
            await _process_changes(changes, config, indexer, sink, buf)
    except asyncio.CancelledError:
        logger.info("File watcher stopped")
    except Exception:
        logger.exception("File watcher error")


async def _process_changes(
    changes: set[tuple[Change, str]],
    config: DaemonConfig,
    indexer: Any,  # MarkdownIndexer
    sink: TMSink,
    buf: Any = None,  # DraftBuffer | None
) -> None:
    """Process a batch of file changes from watchfiles."""
    for change_type, path_str in changes:
        path = Path(path_str)

        # Only process .md files
        if path.suffix != ".md":
            continue

        # Find matching vault config
        vault = _find_vault(path, config)
        if vault is None:
            continue

        # Check exclude patterns
        if any(exc in path_str for exc in vault.exclude):
            continue

        if change_type in (Change.added, Change.modified):
            try:
                entry = indexer.parse_file(str(path))
                description = entry.get("description", "")
                solution = entry.get("solution", "")
                # Combine problem + solution into a single content block for
                # the draft pipeline so RefinementWorker can LLM-refine it.
                content_parts = []
                if description:
                    content_parts.append(f"## 背景\n{description}")
                if solution:
                    content_parts.append(f"## 内容\n{solution}")
                content = "\n\n".join(content_parts) or solution

                title = entry.get("title", path.stem)

                if buf is not None:
                    # Route through local DraftBuffer → RefinementWorker will
                    # LLM-refine and publish to PostgreSQL via sink.draft_save
                    # + sink.draft_publish.  We store title in the content
                    # preamble so the worker can extract it later.
                    full_content = f"# {title}\n\n{content}"
                    local_id = await buf.create_draft(
                        project=vault.project,
                        conversation_id=None,
                        content=full_content,
                        title=title,
                        source="obsidian",
                    )
                    # Immediately mark for refinement so RefinementWorker picks it up
                    await buf.mark_needs_refinement(local_id)
                    logger.info(
                        "[WATCH] obsidian → file=%s action=%s project=%s local_draft_id=%s",
                        path.name, change_type.name, vault.project, local_id,
                    )
                else:
                    # Fallback: save directly (no LLM refinement)
                    result = await sink.draft_save(
                        title=title,
                        content=content,
                        tags=entry.get("tags"),
                        project=vault.project,
                        group_key=entry.get("group_key"),
                    )
                    draft_id = result.get("id") if isinstance(result, dict) else None
                    logger.info(
                        "[WATCH] obsidian → file=%s action=%s project=%s draft_id=%s",
                        path.name, change_type.name, vault.project, draft_id,
                    )
            except Exception:
                logger.exception("Failed to index: %s", path)

        elif change_type == Change.deleted:
            logger.info("Deleted: %s (no-op, TM has no delete API yet)", path.name)


def _find_vault(path: Path, config: DaemonConfig) -> VaultConfig | None:
    """Find matching VaultConfig for a given file path."""
    for vault in config.obsidian.vaults:
        vault_path = Path(vault.path).expanduser()
        try:
            path.relative_to(vault_path)
            return vault
        except ValueError:
            continue
    return None

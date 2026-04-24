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


async def start_watcher(config: DaemonConfig, sink: TMSink) -> None:
    """Watch all configured Obsidian vaults and index changes.

    Runs as a long-lived asyncio task. Cancel to stop.
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
            await _process_changes(changes, config, indexer, sink)
    except asyncio.CancelledError:
        logger.info("File watcher stopped")
    except Exception:
        logger.exception("File watcher error")


async def _process_changes(
    changes: set[tuple[Change, str]],
    config: DaemonConfig,
    indexer: Any,  # MarkdownIndexer
    sink: TMSink,
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
                await sink.save(
                    title=entry.get("title", path.stem),
                    problem=entry.get("description", "")[:200],
                    solution=entry.get("solution", ""),
                    tags=entry.get("tags"),
                    project=vault.project,
                    group_key=entry.get("group_key"),
                )
                logger.info("Indexed: %s", path.name)
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

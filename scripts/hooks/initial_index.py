"""Initial full-index script for Obsidian vaults.

Scans all configured Obsidian vaults and imports existing Markdown files
into TeamMemory as published Experiences via the memory_save MCP tool.
"""

from __future__ import annotations

import logging
import os
import subprocess
from glob import glob

from scripts.hooks.markdown_indexer import MarkdownIndexer
from scripts.hooks.shared import TMClient

logger = logging.getLogger(__name__)


async def initial_index(
    config_path: str,
    tm_url: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Perform a full index of all configured Obsidian vaults.

    Scans each vault for Markdown files matching ``index_patterns``,
    skips files that fail ``should_index`` or are untracked by git,
    and imports them into TeamMemory via ``memory_save``.

    Args:
        config_path: Path to the obsidian_config.yaml file.
        tm_url: Base URL for the TeamMemory MCP-over-HTTP endpoint.
        dry_run: If True, only print what would be indexed without
                 actually calling the TM API.

    Returns:
        Dict with keys: total_files, indexed, skipped, errors.
    """
    config = MarkdownIndexer.load_config(config_path)
    indexer = MarkdownIndexer()
    tm = TMClient(tm_url)

    stats: dict[str, int] = {"total_files": 0, "indexed": 0, "skipped": 0, "errors": 0}

    for vault in config.get("vaults", []):
        repo_path = vault["path"]
        project = vault["project"]

        if not os.path.isdir(repo_path):
            print(f"跳过不存在的路径: {repo_path}")
            logger.warning("跳过不存在的路径: %s", repo_path)
            continue

        # Scan all files matching index_patterns
        for pattern in vault.get("index_patterns", []):
            full_pattern = os.path.join(repo_path, pattern)
            for md_file in glob(full_pattern, recursive=True):
                stats["total_files"] += 1

                if not indexer.should_index(md_file, config):
                    stats["skipped"] += 1
                    logger.debug("跳过 (should_index=False): %s", md_file)
                    continue

                # Skip untracked files — only index git-managed files
                if _is_untracked(md_file, repo_path):
                    stats["skipped"] += 1
                    logger.debug("跳过 (untracked): %s", md_file)
                    continue

                data = indexer.parse_file(md_file)

                if dry_run:
                    print(f"[DRY RUN] 将索引: {md_file} → {project}/{data['title']}")
                    stats["indexed"] += 1
                    continue

                try:
                    result = await tm.save(
                        title=data["title"],
                        content=data.get("solution", data.get("description", "")),
                        tags=data.get("tags", []),
                        project=project,
                        source="obsidian",
                    )
                    stats["indexed"] += 1
                    logger.info("已索引: %s → %s", md_file, result.get("id", "?"))
                except Exception as e:
                    print(f"索引失败: {md_file}: {e}")
                    logger.error("索引失败: %s: %s", md_file, e)
                    stats["errors"] += 1

    return stats


def _is_untracked(file_path: str, repo_path: str) -> bool:
    """Check whether *file_path* is untracked by git in *repo_path*.

    Returns True if the file is not tracked (``git ls-files --error-unmatch``
    exits with non-zero), False if it is tracked.

    If git itself fails to run, the file is conservatively treated as
    untracked (returns True).
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "ls-files", "--error-unmatch", file_path],
            capture_output=True,
            text=True,
        )
        return result.returncode != 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return True

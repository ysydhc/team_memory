"""GitWatcher — watches Git events (add/commit/rm/mv) and triggers TM indexing.

Bridges Git operations to the TeamMemory indexing pipeline:
- ``on_git_add``    → draft_save (stage = draft)
- ``on_git_commit`` → draft_publish (if draft exists) or direct published
- ``on_git_rm``     → soft-delete marker (TM has no delete API yet)
- ``on_git_mv``     → rm old + add new
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from scripts.hooks.markdown_indexer import MarkdownIndexer
from scripts.hooks.shared import TMClient

logger = logging.getLogger(__name__)


class GitWatcher:
    """React to Git events and index Markdown files via TeamMemory.

    Parameters
    ----------
    config : dict
        Obsidian vault configuration (same structure as ``obsidian_config.yaml``).
    indexer : MarkdownIndexer
        Stateless parser / filter helper.
    tm_client : TMClient
        Async HTTP client for TeamMemory MCP endpoints.
    """

    def __init__(
        self,
        config: dict[str, Any],
        indexer: MarkdownIndexer,
        tm_client: TMClient,
    ) -> None:
        self._config = config
        self._indexer = indexer
        self._tm = tm_client

    # -- git add -------------------------------------------------------------

    async def on_git_add(self, repo_path: str, files: list[str]) -> list[dict]:
        """Handle ``git add``: draft-save every indexable file.

        For each file in *files* that passes ``should_index``, parse it
        and save as a draft Experience via TM.

        Returns a list of draft-save results (one per file saved).
        """
        results: list[dict] = []
        for f in files:
            if not self._indexer.should_index(f, self._config):
                logger.debug("on_git_add: skip %s (should_index=False)", f)
                continue

            data = self._indexer.parse_file(f)
            project = self._indexer.resolve_project(f, self._config)

            result = await self._tm.draft_save(
                title=data["title"],
                content=data["solution"],
                tags=data["tags"],
                project=project,
                group_key=f,
            )
            results.append(result)
            logger.info("on_git_add: draft_save %s → %s", f, result.get("id"))

        return results

    # -- git commit ----------------------------------------------------------

    async def on_git_commit(
        self,
        repo_path: str,
        committed_files: list[str] | None = None,
    ) -> list[dict]:
        """Handle ``git commit``: publish drafts or create published Experiences.

        If *committed_files* is provided, only those files are processed.
        Otherwise, ``git log --name-only -1`` is used to discover them.

        For each indexable ``.md`` file:
        - If a draft exists (found via ``recall`` by ``group_key``),
          call ``draft_publish``.
        - Otherwise, create a new published Experience via ``draft_save``.

        Returns a list of publish / save results.
        """
        if committed_files is None:
            committed_files = self.get_committed_files(repo_path)
            # Convert relative paths from git output to absolute
            committed_files = [
                repo_path + "/" + f if not os.path.isabs(f) else f
                for f in committed_files
            ]

        results: list[dict] = []
        for f in committed_files:
            if not self._indexer.should_index(f, self._config):
                logger.debug("on_git_commit: skip %s (should_index=False)", f)
                continue

            # Try to find an existing draft for this file
            recall_result = await self._tm.recall(
                query=f"group_key:{f}",
                project=self._indexer.resolve_project(f, self._config),
            )

            drafts = _extract_drafts(recall_result, f)

            if drafts:
                # Publish the first matching draft with the latest file content
                draft_id = drafts[0]["id"]
                data = self._indexer.parse_file(f)
                result = await self._tm.draft_publish(
                    draft_id=draft_id,
                    refined_content=data["solution"],
                )
                results.append(result)
                logger.info("on_git_commit: draft_publish %s → %s", f, draft_id)
            else:
                # No draft found — create published directly
                data = self._indexer.parse_file(f)
                project = self._indexer.resolve_project(f, self._config)
                result = await self._tm.draft_save(
                    title=data["title"],
                    content=data["solution"],
                    tags=data["tags"],
                    project=project,
                    group_key=f,
                )
                results.append(result)
                logger.info("on_git_commit: draft_save (no prior draft) %s → %s", f, result.get("id"))

        return results

    # -- git rm --------------------------------------------------------------

    async def on_git_rm(self, repo_path: str, files: list[str]) -> list[dict]:
        """Handle ``git rm``: soft-delete Experiences for removed files.

        Since TM currently has no delete API, this only returns a hint
        indicating which Experiences should be soft-deleted.

        Returns a list of dicts with ``action``, ``id``, and ``message`` keys.
        """
        results: list[dict] = []
        for f in files:
            if not f.endswith(".md"):
                logger.debug("on_git_rm: skip %s (not .md)", f)
                continue

            # Try to find an existing Experience for this file
            recall_result = await self._tm.recall(
                query=f"group_key:{f}",
                project=self._indexer.resolve_project(f, self._config),
            )

            exps = _extract_drafts(recall_result, f)

            if exps:
                for exp in exps:
                    results.append({
                        "action": "soft_delete",
                        "id": exp["id"],
                        "message": "TM has no delete API; soft-delete pending",
                    })
                    logger.info("on_git_rm: soft_delete %s → %s", f, exp["id"])
            else:
                results.append({
                    "action": "not_found",
                    "file": f,
                    "message": "No Experience found for deleted file",
                })
                logger.info("on_git_rm: no experience found for %s", f)

        return results

    # -- git mv --------------------------------------------------------------

    async def on_git_mv(
        self, repo_path: str, old_path: str, new_path: str
    ) -> dict:
        """Handle ``git mv``: rm the old path + add the new path.

        Returns a dict with ``rm`` and ``add`` keys containing the
        respective operation results.
        """
        rm_results = await self.on_git_rm(repo_path, [old_path])
        add_results = await self.on_git_add(repo_path, [new_path])

        return {
            "rm": rm_results[0] if rm_results else {"action": "none"},
            "add": add_results[0] if add_results else None,
        }

    # -- git helpers ---------------------------------------------------------

    def get_staged_files(self, repo_path: str) -> list[str]:
        """Return the list of staged ``.md`` files in *repo_path*.

        Executes ``git -C <repo_path> diff --name-only --cached`` and
        filters the output to keep only ``.md`` files.
        """
        try:
            proc = subprocess.run(
                ["git", "-C", repo_path, "diff", "--name-only", "--cached"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("get_staged_files: git command failed for %s", repo_path)
            return []

        files = [line for line in proc.stdout.strip().splitlines() if line]
        return [f for f in files if f.endswith(".md")]

    def get_committed_files(self, repo_path: str) -> list[str]:
        """Return the list of ``.md`` files in the most recent commit.

        Executes ``git -C <repo_path> log --name-only -1 --pretty=format:``
        and filters the output to keep only ``.md`` files.
        """
        try:
            proc = subprocess.run(
                [
                    "git", "-C", repo_path, "log",
                    "--name-only", "-1", "--pretty=format:",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("get_committed_files: git command failed for %s", repo_path)
            return []

        files = [line for line in proc.stdout.strip().splitlines() if line]
        return [f for f in files if f.endswith(".md")]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_drafts(recall_result: dict | None, file_path: str) -> list[dict]:
    """Extract matching experiences from a recall result by group_key.

    Looks through ``recall_result["results"]`` for items whose
    ``group_key`` matches *file_path*.

    Returns a (possibly empty) list of matching experience dicts.
    """
    if not recall_result or "results" not in recall_result:
        return []

    matches = []
    for item in recall_result["results"]:
        if isinstance(item, dict) and item.get("group_key") == file_path:
            matches.append(item)
    return matches

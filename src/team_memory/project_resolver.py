"""Project name inference — zero-config project identification.

Priority chain:
  1. .tm.toml [project].name — explicit declaration
  2. Git repo basename → snake_case — covers 80%+ projects
  3. Directory basename → snake_case — last resort

Usage:
  from team_memory.project_resolver import infer_project_name, read_tm_toml

  name = infer_project_name()           # from cwd
  name = infer_project_name("/path/to") # from specific dir

  config = read_tm_toml("/path/to")
  # → {"name": "my_project", "aliases": ["old_name"]}
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("team_memory.project_resolver")

# ---------------------------------------------------------------------------
# Global alias registry (populated by .tm.toml at server startup)
# ---------------------------------------------------------------------------

# alias_name → canonical_name
_alias_registry: dict[str, str] = {}
_aliases = _alias_registry

_BUILTIN_SHORTHAND_ALIASES: dict[str, list[str]] = {
    "team_doc": ["tm", "team_memory", "team-memory", "team_doc", "tm-cli", "tm-daemon"],
    "team_memory": ["tm", "team_memory", "team-memory", "team_doc", "tm-cli", "tm-daemon"],
    "phone_pilot": ["pp", "phone_pilot", "phone-pilot"],
    "adx-sdk-android": ["adx-sdk-android", "adx_android"],
    "adx-sdk-ios": ["adx-sdk-ios", "adx_ios"],
    "inapp-game-dollshooter": ["dollshooter", "doll_shooter"],
}


def register_aliases(canonical: str, aliases: list[str]) -> None:
    """Register project aliases (called at server startup from .tm.toml)."""
    for alias in aliases:
        _alias_registry[alias] = canonical


def resolve_alias(project: str) -> str:
    """Resolve a project name through the alias registry.

    If the name is a known alias, return the canonical name.
    Otherwise return the name unchanged.
    """
    return _alias_registry.get(project, project)


# ---------------------------------------------------------------------------
# .tm.toml reader
# ---------------------------------------------------------------------------


def read_tm_toml(directory: str | Path) -> dict[str, Any] | None:
    """Read .tm.toml from the given directory.

    Returns dict with keys: name, aliases (if present).
    Returns None if file does not exist or is invalid.
    """
    tm_path = Path(directory) / ".tm.toml"
    if not tm_path.is_file():
        return None

    try:
        # Use tomllib (Python 3.11+) or fallback to tomli
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                logger.warning(
                    ".tm.toml found at %s but no TOML parser available. Install tomli for Python < 3.11.",
                    tm_path,
                )
                return None

        with open(tm_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        if not isinstance(project, dict) or "name" not in project:
            logger.warning(".tm.toml at %s has no [project].name", tm_path)
            return None

        result: dict[str, Any] = {"name": str(project["name"])}
        aliases = project.get("aliases")
        if isinstance(aliases, list):
            result["aliases"] = [str(a) for a in aliases]
        return result

    except Exception as e:
        logger.warning("Failed to parse .tm.toml at %s: %s", tm_path, e)
        return None


# ---------------------------------------------------------------------------
# Git repo detection
# ---------------------------------------------------------------------------


def _git_repo_name(cwd: str) -> str | None:
    """Get git repo basename from cwd. Returns None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.basename(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def canonicalize_project_name(name: str) -> str:
    """Canonical project slug: path-aware, case/separator normalized.

    Rules (single source of truth for every write path):
      1. Path-like input → take the last segment (git/dir basename),
         so "gensoft-ipcs-mobile/adx-sdk-android" → "adx-sdk-android".
      2. Lowercase.
      3. Collapse any run of separators ( - . space / \\ ) into "_".
      4. Strip leading/trailing underscores.

    Examples:
      "adx-sdk-android"                     → "adx_sdk_android"
      "gensoft-ipcs-mobile/adx-sdk-android" → "adx_sdk_android"
      "MyGuguGaga"                          → "mygugugaga"
    """
    import re

    value = (name or "").strip()
    if not value:
        return ""
    # 1. basename for path-like names
    for sep in ("/", "\\"):
        if sep in value:
            value = value.rsplit(sep, 1)[-1]
    # 2-4. lowercase, collapse separators, strip
    value = value.lower()
    value = re.sub(r"[-.\s]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _normalize(name: str) -> str:
    """Backward-compatible alias for :func:`canonicalize_project_name`."""
    return canonicalize_project_name(name)


def normalize_project_name(name: str) -> str:
    """Alias for :func:`canonicalize_project_name`."""
    return canonicalize_project_name(name)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def infer_project_name(cwd: str | None = None) -> str:
    """Infer project name from directory, zero-config.

    Priority:
      1. .tm.toml [project].name
      2. Git repo basename (normalized)
      3. Directory basename (normalized)

    Returns a non-empty string. Falls back to "default" if all methods fail.
    """
    cwd = cwd or os.getcwd()

    # 1. .tm.toml
    tm_config = read_tm_toml(cwd)
    if tm_config and tm_config.get("name"):
        return _normalize(tm_config["name"])

    # 2. Git repo name
    git_name = _git_repo_name(cwd)
    if git_name:
        return _normalize(git_name)

    # 3. Directory basename
    dir_name = os.path.basename(os.path.abspath(cwd))
    if dir_name:
        return _normalize(dir_name)

    # Last resort
    return "default"


def resolve_project(cwd: str | None = None) -> str:
    """Resolve project name for *cwd* (alias for :func:`infer_project_name`)."""
    return infer_project_name(cwd)


def _prompt_matches_keyword(prompt_lower: str, keyword: str) -> bool:
    """Check if *keyword* appears in *prompt_lower* with ASCII identifier boundaries."""
    kw = (keyword or "").strip().lower()
    if len(kw) < 2:
        return False
    pattern = rf"(?<![a-z0-9_-]){re.escape(kw)}(?![a-z0-9_-])"
    return bool(re.search(pattern, prompt_lower))


def resolve_multi_root_project(
    workspace_roots: list[str] | None = None,
    prompt: str = "",
    target_files: list[str] | None = None,
    sticky_project: str = "",
    config_project: str = "",
) -> tuple[str, bool]:
    """Resolve active project across multi-root workspaces.

    Returns ``(resolved_project, is_explicit)`` where ``is_explicit`` is ``True``
    if resolved via ``config_project``, single ``workspace_roots``,
    ``target_files``, or ``prompt`` explicit match, and ``False`` if resolved
    via ``sticky_project`` or multi-root fallback ``workspace_roots[0]``.
    """
    # 1. Explicit config_project override
    if config_project and config_project != "default":
        return (normalize_project_name(config_project), True)

    # 2. Candidate roots (non-empty)
    candidate_roots = [r.strip() for r in (workspace_roots or []) if r and r.strip()]

    # 3. Check target_files against candidate roots
    if target_files and candidate_roots:
        for tf in target_files:
            if not tf:
                continue
            tf_norm = str(Path(tf))
            for root in candidate_roots:
                root_norm = str(Path(root))
                root_name = Path(root).name
                if (
                    tf.startswith(root)
                    or tf_norm == root_norm
                    or tf_norm.startswith(root_norm.rstrip("/") + "/")
                    or (root_name and f"/{root_name}/" in tf)
                ):
                    return (resolve_project(cwd=root), True)

    # 4. Check prompt (if non-empty)
    if prompt and prompt.strip():
        prompt_lower = prompt.lower()

        # 4a. Direct root path or /{root_name}/ mention in prompt
        for root in candidate_roots:
            root_name = Path(root).name
            if (root and root.lower() in prompt_lower) or (
                root_name and f"/{root_name.lower()}/" in prompt_lower
            ):
                return (resolve_project(cwd=root), True)

        # 4b. Keyword/alias match for candidate roots
        for root in candidate_roots:
            proj = resolve_project(cwd=root)
            raw_basename = Path(root).name.lower()
            basename = normalize_project_name(Path(root).name)
            proj_hyphen = proj.replace("_", "-")
            basename_hyphen = basename.replace("_", "-")

            keywords: list[str] = [
                proj,
                basename,
                proj_hyphen,
                basename_hyphen,
                raw_basename,
            ]
            for lookup_key in (proj, basename, proj_hyphen, basename_hyphen, raw_basename):
                keywords.extend(_BUILTIN_SHORTHAND_ALIASES.get(lookup_key, []))
            keywords.extend(
                alias
                for alias, canonical in _aliases.items()
                if canonical in (proj, basename, proj_hyphen, basename_hyphen)
                or normalize_project_name(canonical) in (proj, basename)
            )

            seen: set[str] = set()
            for kw in keywords:
                kw_lower = kw.lower()
                if len(kw_lower) < 2 or kw_lower in seen:
                    continue
                seen.add(kw_lower)
                if _prompt_matches_keyword(prompt_lower, kw_lower):
                    return (proj, True)

        # 4c. Shorthand or registered alias match even when workspace_roots is empty / unmatched
        for canonical, aliases in _BUILTIN_SHORTHAND_ALIASES.items():
            for kw in [canonical, *aliases]:
                if len(kw) >= 2 and _prompt_matches_keyword(prompt_lower, kw):
                    return (normalize_project_name(canonical), True)
        for alias, canonical in _aliases.items():
            for kw in (alias, canonical):
                if len(kw) >= 2 and _prompt_matches_keyword(prompt_lower, kw):
                    return (normalize_project_name(canonical), True)

    # 5. Single candidate root
    if len(candidate_roots) == 1:
        return (resolve_project(cwd=candidate_roots[0]), True)

    # 6. Sticky session project
    if sticky_project and sticky_project != "default":
        return (normalize_project_name(sticky_project), False)

    # 7. Multi-root fallback to first root
    if candidate_roots:
        return (resolve_project(cwd=candidate_roots[0]), False)

    # 8. Final fallback
    return (resolve_project(), False)


def infer_project_with_aliases(cwd: str | None = None) -> dict[str, Any]:
    """Infer project name + aliases from directory.

    Returns:
      {"name": "team_doc", "aliases": ["team_memory"]}
    """
    cwd = cwd or os.getcwd()
    name = infer_project_name(cwd)

    aliases: list[str] = []
    tm_config = read_tm_toml(cwd)
    if tm_config and tm_config.get("aliases"):
        aliases = tm_config["aliases"]

    return {"name": name, "aliases": aliases}


# ---------------------------------------------------------------------------
# Client config (~/.config/tm/config.toml)
# ---------------------------------------------------------------------------

_CLIENT_CONFIG_PATH = Path.home() / ".config" / "tm" / "config.toml"


def load_client_config() -> dict[str, Any]:
    """Load client configuration from ~/.config/tm/config.toml.

    Returns dict with sections: server, llm, conversation, wiki.
    Returns empty dict if file does not exist.
    """
    if not _CLIENT_CONFIG_PATH.is_file():
        return {}

    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                logger.debug("No TOML parser available for %s", _CLIENT_CONFIG_PATH)
                return {}

        with open(_CLIENT_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", _CLIENT_CONFIG_PATH, e)
        return {}


def find_project_root(start: str | None = None) -> str:
    """Walk up from start to find project root (contains .git or pyproject.toml)."""
    current = Path(start or os.getcwd())
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return str(parent)
    return str(current)


def load_dotenv_if_exists(start: str | None = None) -> None:
    """Load .env from project root if it exists.

    Walks up from *start* (or cwd) to find the project root, then loads
    .env if present.  Safe to call multiple times — dotenv skips already-set
    variables by default (override=False).
    """
    root = find_project_root(start)
    env_path = os.path.join(root, ".env")
    if os.path.isfile(env_path):
        try:
            from dotenv import load_dotenv as _load_dotenv

            _load_dotenv(env_path)
        except ImportError:
            pass


def get_server_url() -> str:
    """Get TM server URL from config or env."""
    from team_memory.client_config import get_server_url as _get_url

    return _get_url()


def get_api_key() -> str:
    """Get TM API key from env > config."""
    from team_memory.client_config import get_api_key as _get_key

    return _get_key()

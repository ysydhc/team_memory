"""0ms code-grounded evidence anchor extraction and epistemic tier classification.

Extracts structured code anchors (`file_paths`, `symbols`, `commit_sha`) from
experience text using deterministic local regex rules with strict boundary
guards to avoid false positives (URLs, versions, dates, placeholders, bare
ambiguous filenames).
"""

from __future__ import annotations

import re
from typing import Any

VALID_EPISTEMIC_TIERS: frozenset[str] = frozenset({"verified", "tool_grounded", "auto_extract"})
VALID_STALENESS_STATUSES: frozenset[str] = frozenset({"fresh", "possibly_stale", "stale"})
_EPISTEMIC_RANK: dict[str, int] = {"auto_extract": 0, "tool_grounded": 1, "verified": 2}

_MAX_STORED_PATHS = 20
_MAX_STORED_SYMBOLS = 30

# Recognized source/config file extensions for path extraction.
_CODE_EXTENSIONS = (
    "py",
    "pyi",
    "java",
    "kt",
    "kts",
    "m",
    "mm",
    "h",
    "hpp",
    "c",
    "cc",
    "cpp",
    "go",
    "rs",
    "ts",
    "tsx",
    "js",
    "jsx",
    "mjs",
    "cjs",
    "swift",
    "rb",
    "php",
    "sql",
    "proto",
    "yaml",
    "yml",
    "toml",
    "json",
    "ini",
    "cfg",
    "sh",
    "bash",
    "zsh",
    "gradle",
    "xml",
    "plist",
    "podspec",
)

# Explicit root-level build/config files allowed without a directory slash.
_ROOT_BUILD_FILES: frozenset[str] = frozenset(
    {
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Gemfile",
        "Podfile",
        "CMakeLists.txt",
        "build.gradle",
        "settings.gradle",
        "alembic.ini",
    }
)

_URL_RE = re.compile(r"https?://[^\s)\"'`<>]+", re.IGNORECASE)

# Path MUST contain at least one '/' (prevents bare filename ambiguity like config.py).
_EXT_PATTERN = "|".join(_CODE_EXTENSIONS)
_SLASH_PATH_RE = re.compile(
    rf"(?<![\w:/.-])"
    rf"((?:[a-zA-Z0-9_.-]+/)+[a-zA-Z0-9_.-]+\.(?:{_EXT_PATTERN}))"
    rf"(?![\w./-])"
)
_ROOT_FILE_RE = re.compile(
    r"(?<![\w/.-])("
    + "|".join(re.escape(name) for name in sorted(_ROOT_BUILD_FILES))
    + r")(?![\w./-])"
)

_PLACEHOLDER_PATH_FRAGMENTS: tuple[str, ...] = (
    "...",
    "*",
    "<",
    ">",
    "{",
    "}",
    "path/to/",
    "your/",
    "example/",
    "foo/bar",
)

_MIME_PREFIXES: tuple[str, ...] = (
    "application/",
    "text/",
    "image/",
    "audio/",
    "video/",
    "multipart/",
)

_COMMIT_SHA_RE = re.compile(
    r"(?:\bcommit\s+|@)([0-9a-f]{7,40})\b",
    re.IGNORECASE,
)

# Code symbol extraction patterns.
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.:-]{3,60})`")
_CAMEL_CLASS_RE = re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]{1,50})\b")
_SNAKE_SYMBOL_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,5})\b")
_DOTTED_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]{2,40})\b")

_SYMBOL_STOPWORDS: frozenset[str] = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "self",
        "cls",
        "args",
        "kwargs",
        "return",
        "import",
        "from",
        "class",
        "async",
        "await",
        "default",
        "example",
        "string",
        "integer",
        "boolean",
        "object",
        "array",
        "error",
        "warning",
        "info",
        "debug",
        "http",
        "https",
        "localhost",
        "utf_8",
        "json",
        "yaml",
        "toml",
    }
)

_TOOL_GROUNDED_SIGNALS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:pytest|unittest|go\s+test|cargo\s+test|npm\s+test|jest|vitest)\b"
        r".*\b(?:pass(?:ed)?|ok|0\s+failed)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:exit\s*code|returncode|exit\s*status)\s*[:=]?\s*0\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+passed\b", re.IGNORECASE),
    re.compile(
        r"\b(?:build|compile|make\s+\w+|alembic\s+upgrade)\b"
        r".*\b(?:succeed(?:ed)?|pass(?:ed)?|ok|done)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:单测|测试|编译|构建|验证|校验)(?:全部)?(?:通过|成功)"),
    re.compile(r"退出码\s*[:=]?\s*0"),
)


def _is_valid_file_path(raw: str) -> bool:
    p = raw.strip().lstrip("./")
    if not p or len(p) > 260:
        return False
    lower = p.lower()
    if any(frag in lower for frag in _PLACEHOLDER_PATH_FRAGMENTS):
        return False
    if any(lower.startswith(prefix) for prefix in _MIME_PREFIXES):
        return False
    # Exclude date-like segments (e.g. 2026/11/25.json)
    if re.match(r"^\d{4}/\d{1,2}/\d{1,2}", p):
        return False
    # Exclude version-like prefixes (e.g. v1.2/foo)
    if re.match(r"^v\d+\.\d+", lower):
        return False
    return True


def extract_evidence_anchors(
    *texts: str | None,
    commit_sha: str | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract structured code anchors from text with 0ms local regex rules.

    Returns a normalized dict:
        {"file_paths": [...], "symbols": [...], "commit_sha": "..."}
    Omits empty keys only if everything is empty (returns `{}` when no anchors found).
    """
    combined = "\n".join(t for t in texts if isinstance(t, str) and t.strip())
    # Strip URLs first so paths inside https://... are never extracted as repo files
    sanitized = _URL_RE.sub(" ", combined) if combined else ""

    file_paths: list[str] = []
    seen_paths: set[str] = set()

    def _add_path(candidate: str) -> None:
        norm = candidate.strip().lstrip("./")
        if not norm or norm in seen_paths:
            return
        if len(file_paths) >= _MAX_STORED_PATHS:
            return
        if norm in _ROOT_BUILD_FILES or _is_valid_file_path(norm):
            seen_paths.add(norm)
            file_paths.append(norm)

    if sanitized:
        for m in _SLASH_PATH_RE.finditer(sanitized):
            _add_path(m.group(1))
        for m in _ROOT_FILE_RE.finditer(sanitized):
            _add_path(m.group(1))

    symbols: list[str] = []
    seen_symbols: set[str] = set()

    def _add_symbol(raw_sym: str) -> None:
        sym = raw_sym.strip().strip("`'\"()[].,;:")
        if len(sym) < 4 or len(sym) > 64:
            return
        if "/" in sym or sym in seen_paths or sym in _ROOT_BUILD_FILES:
            return
        low = sym.lower()
        if low in _SYMBOL_STOPWORDS or low in seen_symbols:
            return
        # Skip pure numbers or file extensions
        if re.match(r"^[0-9._-]+$", sym):
            return
        if len(symbols) >= _MAX_STORED_SYMBOLS:
            return
        seen_symbols.add(low)
        symbols.append(sym)

    if sanitized:
        for m in _BACKTICK_SYMBOL_RE.finditer(sanitized):
            tok = m.group(1)
            if "/" not in tok and not re.search(rf"\.(?:{_EXT_PATTERN})$", tok):
                _add_symbol(tok)
        for m in _CAMEL_CLASS_RE.finditer(sanitized):
            _add_symbol(m.group(1))
        for m in _DOTTED_SYMBOL_RE.finditer(sanitized):
            tok = m.group(1)
            if not re.search(rf"\.(?:{_EXT_PATTERN})$", tok):
                _add_symbol(tok)
        for m in _SNAKE_SYMBOL_RE.finditer(sanitized):
            _add_symbol(m.group(1))

    resolved_sha = (commit_sha or "").strip() or None
    if not resolved_sha and sanitized:
        sha_m = _COMMIT_SHA_RE.search(sanitized)
        if sha_m:
            resolved_sha = sha_m.group(1).lower()

    extracted: dict[str, Any] = {}
    if file_paths:
        extracted["file_paths"] = file_paths
    if symbols:
        extracted["symbols"] = symbols
    if resolved_sha:
        extracted["commit_sha"] = resolved_sha

    if existing and isinstance(existing, dict):
        return merge_evidence_anchors(existing, extracted)
    return extracted


def merge_evidence_anchors(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge two evidence_anchors dicts preserving order and caps."""
    a = existing if isinstance(existing, dict) else {}
    b = incoming if isinstance(incoming, dict) else {}

    paths: list[str] = []
    seen_p: set[str] = set()
    for p in list(a.get("file_paths") or []) + list(b.get("file_paths") or []):
        if isinstance(p, str) and p.strip() and p.strip() not in seen_p:
            seen_p.add(p.strip())
            paths.append(p.strip())
            if len(paths) >= _MAX_STORED_PATHS:
                break

    syms: list[str] = []
    seen_s: set[str] = set()
    for s in list(a.get("symbols") or []) + list(b.get("symbols") or []):
        if isinstance(s, str) and s.strip() and s.strip().lower() not in seen_s:
            seen_s.add(s.strip().lower())
            syms.append(s.strip())
            if len(syms) >= _MAX_STORED_SYMBOLS:
                break

    sha = b.get("commit_sha") or a.get("commit_sha")
    out: dict[str, Any] = {}
    if paths:
        out["file_paths"] = paths
    if syms:
        out["symbols"] = syms
    if isinstance(sha, str) and sha.strip():
        out["commit_sha"] = sha.strip()
    return out


def compact_evidence_anchors(
    anchors: dict[str, Any] | None,
    *,
    max_paths: int = 8,
    max_symbols: int = 8,
) -> dict[str, Any]:
    """Trim evidence_anchors to protect recall token budgets."""
    if not isinstance(anchors, dict) or not anchors:
        return {}
    out: dict[str, Any] = {}
    raw_paths = anchors.get("file_paths")
    if isinstance(raw_paths, list) and raw_paths and max_paths > 0:
        out["file_paths"] = [str(p) for p in raw_paths[:max_paths] if p]
    raw_syms = anchors.get("symbols")
    if isinstance(raw_syms, list) and raw_syms and max_symbols > 0:
        out["symbols"] = [str(s) for s in raw_syms[:max_symbols] if s]
    sha = anchors.get("commit_sha")
    if isinstance(sha, str) and sha.strip():
        out["commit_sha"] = sha.strip()
    return out


def infer_epistemic_tier(
    *,
    explicit_tier: str | None = None,
    is_pinned: bool = False,
    source: str = "auto_extract",
    evidence: str | None = None,
    text: str = "",
) -> str:
    """Resolve epistemic trust tier: verified > tool_grounded > auto_extract."""
    if explicit_tier and explicit_tier.strip().lower() in VALID_EPISTEMIC_TIERS:
        return explicit_tier.strip().lower()

    if is_pinned or (source or "").strip().lower() == "manual":
        return "verified"

    if evidence and evidence.strip():
        return "tool_grounded"

    if text:
        for pat in _TOOL_GROUNDED_SIGNALS:
            if pat.search(text):
                return "tool_grounded"

    return "auto_extract"


def promote_epistemic_tier(existing: str | None, incoming: str | None) -> str:
    """Return the higher-trust epistemic tier (verified > tool_grounded > auto_extract)."""
    cur = (existing or "auto_extract").strip().lower()
    inc = (incoming or "auto_extract").strip().lower()
    if cur not in _EPISTEMIC_RANK:
        cur = "auto_extract"
    if inc not in _EPISTEMIC_RANK:
        inc = "auto_extract"
    return inc if _EPISTEMIC_RANK[inc] > _EPISTEMIC_RANK[cur] else cur

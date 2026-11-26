"""Chinese-aware technical tokenizer for FTS indexing and search.

Combines deterministic ASCII technical identifier expansion (camelCase,
acronyms + camelCase, snake_case, kebab-case/error codes, file paths,
version numbers, alphanumeric boundaries) with jieba Chinese segmentation
(falling back to CJK bigrams when jieba is unavailable).

Zero external dependencies beyond optional jieba. Both write-side FTS indexing
and query-side `to_tsquery` call `tokenize()` so lexemes stay 100% aligned.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_jieba_available = False
_jieba = None

try:
    import jieba as _jieba_module

    _jieba = _jieba_module
    _jieba.setLogLevel(logging.WARNING)
    _jieba_available = True
except ImportError:
    logger.info("jieba not installed, falling back to simple tokenizer")


_TECH_TERMS: set[str] = set()

# Matches an ASCII technical identifier (may contain dots, hyphens, underscores, colons).
# Path slashes '/' are intentionally excluded so file paths split into path segments first.
_ASCII_TECH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:-][A-Za-z0-9]+)*")

# Matches version strings like v0.3.5, V2.10.3, 16.4
_VERSION_RE = re.compile(r"^[vV]?(\d+(?:\.\d+)+)$")

# Splits camelCase, ACRONYM+CamelCase, and letter<->digit boundaries.
_CAMEL_AND_ALNUM_SPLIT_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])"
    r"|(?<=[A-Z])(?=[A-Z][a-z])"
    r"|(?<=[A-Za-z])(?=\d)"
    r"|(?<=\d)(?=[A-Za-z])"
)

_COARSE_SEP_RE = re.compile(r"[.:]+")
_FINE_SEP_RE = re.compile(r"[_-]+")
_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def load_tech_dict(terms: list[str]) -> None:
    """Load technical terms into jieba's user dictionary."""
    global _TECH_TERMS
    _TECH_TERMS = set(terms)
    if _jieba_available and _jieba is not None:
        for term in terms:
            if len(term) >= 2:
                _jieba.add_word(term, freq=100000)


def load_tech_dict_from_tags(tag_synonyms: dict[str, str], tags: list[str]) -> None:
    """Build tech dictionary from tag_synonyms config and high-frequency tags."""
    terms = list(tag_synonyms.values()) + list(tag_synonyms.keys()) + tags
    unique_terms = list({t for t in terms if t and len(t) >= 2})
    load_tech_dict(unique_terms)


def _expand_camel_and_alnum(piece: str) -> list[str]:
    """Split a single separator-free ASCII piece on camelCase and letter/digit boundaries."""
    parts = _CAMEL_AND_ALNUM_SPLIT_RE.split(piece)
    if len(parts) <= 1:
        return []
    return [p.lower() for p in parts if len(p) >= 2]


def _expand_ascii_token(token: str) -> list[str]:
    """Expand an ASCII technical token using 'preserve original + append split sub-tokens'.

    Examples:
        AdLifecycleMonitor -> ['adlifecyclemonitor', 'ad', 'lifecycle', 'monitor']
        ADXManager         -> ['adxmanager', 'adx', 'manager']
        use_worktree.sh    -> ['use_worktree.sh', 'use_worktree', 'use', 'worktree', 'sh']
        adx-sdk-518        -> ['adx-sdk-518', 'adx', 'sdk', '518']
        v0.3.5             -> ['v0.3.5', '0.3.5']
    """
    lower = token.lower()
    out: list[str] = [lower]

    # Version numbers (e.g. v0.3.5, 16.4): keep full token (+ stripped leading 'v') without noisy single digits
    ver_match = _VERSION_RE.match(token)
    if ver_match:
        bare_ver = ver_match.group(1)
        if bare_ver != lower:
            out.append(bare_ver)
        return out

    # Step 1: split coarse separators ('.' and ':') while preserving compound intermediate segments (e.g. use_worktree)
    coarse_parts = [p for p in _COARSE_SEP_RE.split(token) if p]
    if len(coarse_parts) > 1:
        for cp in coarse_parts:
            cp_lower = cp.lower()
            if len(cp_lower) >= 2:
                out.append(cp_lower)
            fine_parts = [p for p in _FINE_SEP_RE.split(cp) if p]
            if len(fine_parts) > 1:
                for fp in fine_parts:
                    if len(fp) >= 2:
                        out.append(fp.lower())
                    out.extend(_expand_camel_and_alnum(fp))
            else:
                out.extend(_expand_camel_and_alnum(cp))
        return out

    # Step 2: split fine separators ('_' and '-')
    fine_parts = [p for p in _FINE_SEP_RE.split(token) if p]
    if len(fine_parts) > 1:
        for fp in fine_parts:
            if len(fp) >= 2:
                out.append(fp.lower())
            out.extend(_expand_camel_and_alnum(fp))
        return out

    # Step 3: pure camelCase / acronym+camelCase / alphanumeric boundary
    out.extend(_expand_camel_and_alnum(token))
    return out


def _tokenize_non_ascii_jieba(segment: str) -> list[str]:
    """Tokenize a non-ASCII (mostly CJK + punctuation) segment using jieba."""
    if not segment or not segment.strip():
        return []
    tokens: list[str] = []
    for tok in _jieba.cut(segment, cut_all=False):
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) == 1 and not tok.isalnum():
            continue
        tokens.append(tok.lower() if tok.isascii() else tok)
    return tokens


def _tokenize_non_ascii_simple(segment: str) -> list[str]:
    """Fallback non-ASCII tokenizer: split CJK into unigrams + bigrams."""
    if not segment or not segment.strip():
        return []
    tokens: list[str] = []
    for cjk_seg in _CJK_RANGE.findall(segment):
        for i in range(len(cjk_seg)):
            tokens.append(cjk_seg[i])
            if i < len(cjk_seg) - 1:
                tokens.append(cjk_seg[i : i + 2])
    return tokens


def _tokenize_with_segmenter(text: str, non_ascii_fn) -> str:
    """Core tokenizer pipeline shared by jieba and simple fallback modes."""
    result: list[str] = []
    seen: set[str] = set()

    def _emit(tok: str) -> None:
        if not tok:
            return
        if len(tok) == 1 and not tok.isalnum():
            return
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            result.append(tok)

    pos = 0
    for match in _ASCII_TECH_TOKEN_RE.finditer(text):
        start, end = match.span()
        if start > pos:
            for tok in non_ascii_fn(text[pos:start]):
                _emit(tok)
        for tok in _expand_ascii_token(match.group(0)):
            _emit(tok)
        pos = end

    if pos < len(text):
        for tok in non_ascii_fn(text[pos:]):
            _emit(tok)

    return " ".join(result)


def tokenize(text: str) -> str:
    """Tokenize text for FTS indexing and query building. Returns space-separated tokens.

    - ASCII technical identifiers are preserved (lowercased) and expanded into
      camelCase / acronym / snake_case / kebab-case / version sub-tokens.
    - Chinese text is segmented using jieba (or CJK bigram fallback).
    - Single-character non-alphanumeric symbols are stripped.
    - Output tokens are deduplicated in order for compact FTS indexes.
    """
    if not text or not text.strip():
        return ""

    if _jieba_available and _jieba is not None:
        return _tokenize_jieba(text)
    return _tokenize_simple(text)


def _tokenize_jieba(text: str) -> str:
    """Tokenize using jieba for Chinese + technical identifier expansion for ASCII."""
    return _tokenize_with_segmenter(text, _tokenize_non_ascii_jieba)


def _tokenize_simple(text: str) -> str:
    """Fallback: CJK bigrams + technical identifier expansion for ASCII."""
    return _tokenize_with_segmenter(text, _tokenize_non_ascii_simple)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses ~1 token per CJK character and ~4 characters per token for non-CJK text.
    """
    if not text:
        return 0
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_cjk_count = len(text) - cjk_count
    return max(1, cjk_count + non_cjk_count // 4)

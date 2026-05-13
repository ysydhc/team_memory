"""Chinese-aware tokenizer for FTS indexing and search.

Uses jieba for Chinese segmentation with a fallback to simple whitespace
splitting when jieba is unavailable. Preserves English words as-is.
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


def tokenize(text: str) -> str:
    """Tokenize text for FTS indexing. Returns space-separated tokens.

    Chinese text is segmented using jieba (or character-level fallback).
    English words and numbers are preserved as-is.
    """
    if not text or not text.strip():
        return ""

    if _jieba_available and _jieba is not None:
        return _tokenize_jieba(text)
    return _tokenize_simple(text)


def _tokenize_jieba(text: str) -> str:
    """Tokenize using jieba for Chinese + preserve English words."""
    tokens = _jieba.cut(text, cut_all=False)
    result = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if len(token) == 1 and not token.isalnum():
            continue
        result.append(token)
    return " ".join(result)


_CJK_RANGE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")


def _tokenize_simple(text: str) -> str:
    """Fallback: split CJK into bigrams, keep English words."""
    parts = text.split()
    result = []
    for part in parts:
        if _CJK_RANGE.search(part):
            cjk_chars = _CJK_RANGE.findall(part)
            for segment in cjk_chars:
                for i in range(len(segment)):
                    result.append(segment[i])
                    if i < len(segment) - 1:
                        result.append(segment[i : i + 2])
        else:
            if part.strip():
                result.append(part.strip())
    return " ".join(result)

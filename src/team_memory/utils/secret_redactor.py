"""0ms write-time secret redactor for multi-tenant memory safety.

Detects and redacts common credentials, API keys, tokens, private keys, and
secret assignments before text is persisted or sent to embedding/LLM providers,
while avoiding false positives on UUIDs, git commit SHAs, or normal code
variable names.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

audit_logger = logging.getLogger("team_memory.security.audit")

# Values that look like placeholders, env references, or already-redacted markers
_PLACEHOLDER_VALUES = frozenset(
    {
        "none",
        "null",
        "true",
        "false",
        "changeme",
        "placeholder",
        "example",
        "your_api_key",
        "your-api-key",
        "your_secret",
        "your-secret",
        "your_password",
        "your-password",
        "replace_me",
        "replace-me",
        "todo",
        "redacted",
    }
)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_CODE_ATTR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_SNAKE_VAR_RE = re.compile(r"^[a-z][a-z_]{3,}$")
_REPEATED_MASK_RE = re.compile(r"^[\*xX#_\-.]{4,}$")

# ---------------------------------------------------------------------------
# Ordered redaction patterns (most specific first)
# ---------------------------------------------------------------------------

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{20,}\b")
_OPENAI_KEY_RE = re.compile(
    r"\bsk-(?:(?:proj|svcacct|live|test)-[A-Za-z0-9_-]{20,}|[A-Za-z0-9]{20,})\b"
)
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"
)
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")
_AWS_SECRET_KEY_RE = re.compile(
    r"(?i)(\baws_secret_access_key\s*[:=]\s*['\"]?)([A-Za-z0-9/+=]{40})(['\"]?)"
)
_ALIYUN_KEY_RE = re.compile(r"\bLTAI[A-Za-z0-9]{12,24}\b")
_BAIDU_CLOUD_KEY_RE = re.compile(
    r"\bALTAK-[A-Za-z0-9]{20,}\b"
)
_BAIDU_CLOUD_ASSIGN_RE = re.compile(
    r"(?i)(\bbce[_-]?(?:ak|sk|access[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?)([A-Za-z0-9]{24,})(['\"]?)"
)
_BASIC_AUTH_URL_RE = re.compile(
    r"\b(https?://)([^/\s:@\"'<>]+:[^/\s@\"'<>]+)(@[^\s\"'<>]+)"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_BEARER_TOKEN_RE = re.compile(
    r"(?i)(\bBearer\s+)([A-Za-z0-9_\-.~+/]{16,}=*)"
)
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)(\b(?:password|passwd|api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|"
    r"auth[_-]?token|client[_-]?secret|private[_-]?key)\s*[:=]\s*)(['\"]?)([^\s'\"`,;}\]]{8,})(\2)"
)


@dataclass(slots=True)
class RedactionResult:
    """Outcome of redacting a single text string."""

    text: str
    matches: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.matches)

    @property
    def total_count(self) -> int:
        return sum(self.matches.values())


def _bump(counts: dict[str, int], secret_type: str, delta: int = 1) -> None:
    if delta > 0:
        counts[secret_type] = counts.get(secret_type, 0) + delta


def _looks_like_benign_value(raw_val: str, *, quoted: bool) -> bool:
    """Return True if an assignment RHS looks like a variable, placeholder, or already redacted."""
    if not raw_val:
        return True
    if raw_val.startswith(("[REDACTED:", "$", "${", "<", "{{")):
        return True
    lower = raw_val.lower()
    if lower in _PLACEHOLDER_VALUES or _REPEATED_MASK_RE.match(raw_val):
        return True
    if _UUID_RE.match(raw_val) or _SHA_RE.match(raw_val):
        return True
    # Function call or expression in code (e.g. api_key=os.getenv("KEY"))
    if not quoted:
        if "(" in raw_val or ")" in raw_val:
            return True
        if _CODE_ATTR_RE.match(raw_val):
            return True
        # Plain lowercase snake_case identifier without digits (e.g. password=user_password)
        if "_" in raw_val and _SNAKE_VAR_RE.match(raw_val):
            return True
    return False


def redact_text(text: str | None) -> RedactionResult:
    """Scan and redact sensitive credentials in ``text`` in 0ms (pure regex)."""
    if not text:
        return RedactionResult(text=text or "")

    counts: dict[str, int] = {}
    out = text

    # 1. PEM private keys (multi-line)
    out, n = _PRIVATE_KEY_RE.subn("[REDACTED:PRIVATE_KEY]", out)
    _bump(counts, "PRIVATE_KEY", n)

    # 2. Anthropic API keys (must run before generic OpenAI sk- pattern)
    out, n = _ANTHROPIC_KEY_RE.subn("[REDACTED:ANTHROPIC_KEY]", out)
    _bump(counts, "ANTHROPIC_KEY", n)

    # 3. OpenAI API keys
    out, n = _OPENAI_KEY_RE.subn("[REDACTED:OPENAI_KEY]", out)
    _bump(counts, "OPENAI_KEY", n)

    # 4. GitHub tokens (classic & fine-grained PAT)
    out, n = _GITHUB_TOKEN_RE.subn("[REDACTED:GITHUB_TOKEN]", out)
    _bump(counts, "GITHUB_TOKEN", n)

    # 5. AWS Access Key ID & Secret Access Key
    out, n = _AWS_ACCESS_KEY_RE.subn("[REDACTED:AWS_ACCESS_KEY]", out)
    _bump(counts, "AWS_ACCESS_KEY", n)

    out, n = _AWS_SECRET_KEY_RE.subn(r"\1[REDACTED:AWS_SECRET_KEY]\3", out)
    _bump(counts, "AWS_SECRET_KEY", n)

    # 6. Aliyun AccessKey ID
    out, n = _ALIYUN_KEY_RE.subn("[REDACTED:ALIYUN_ACCESS_KEY]", out)
    _bump(counts, "ALIYUN_ACCESS_KEY", n)

    # 7. Baidu Cloud AK/SK
    out, n = _BAIDU_CLOUD_KEY_RE.subn("[REDACTED:BAIDU_CLOUD_KEY]", out)
    _bump(counts, "BAIDU_CLOUD_KEY", n)

    out, n = _BAIDU_CLOUD_ASSIGN_RE.subn(r"\1[REDACTED:BAIDU_CLOUD_KEY]\3", out)
    _bump(counts, "BAIDU_CLOUD_KEY", n)

    # 8. Basic Auth URLs (https://user:pass@host)
    def _replace_basic_auth(m: re.Match[str]) -> str:
        creds = m.group(2)
        if "[REDACTED:" in creds:
            return m.group(0)
        _bump(counts, "BASIC_AUTH_URL")
        return f"{m.group(1)}[REDACTED:BASIC_AUTH_URL]{m.group(3)}"

    out = _BASIC_AUTH_URL_RE.sub(_replace_basic_auth, out)

    # 9. JWT tokens
    out, n = _JWT_RE.subn("[REDACTED:JWT]", out)
    _bump(counts, "JWT", n)

    # 10. Bearer tokens (skip if already redacted)
    def _replace_bearer(m: re.Match[str]) -> str:
        token = m.group(2)
        if token.startswith("[REDACTED:"):
            return m.group(0)
        _bump(counts, "BEARER_TOKEN")
        return f"{m.group(1)}[REDACTED:BEARER_TOKEN]"

    out = _BEARER_TOKEN_RE.sub(_replace_bearer, out)

    # 11. Generic secret/password/api_key assignments
    def _replace_assignment(m: re.Match[str]) -> str:
        prefix, quote, val, end_quote = m.group(1), m.group(2), m.group(3), m.group(4)
        if _looks_like_benign_value(val, quoted=bool(quote)):
            return m.group(0)
        _bump(counts, "SECRET_ASSIGNMENT")
        return f"{prefix}{quote}[REDACTED:SECRET_ASSIGNMENT]{end_quote}"

    out = _SECRET_ASSIGN_RE.sub(_replace_assignment, out)

    return RedactionResult(text=out, matches=counts)


def redact_fields(
    fields: dict[str, str | None],
    *,
    context: str = "write",
    actor: str = "unknown",
) -> tuple[dict[str, str | None], dict[str, int]]:
    """Redact multiple string fields and emit a security audit log if any secret matched."""
    cleaned: dict[str, str | None] = {}
    aggregate: dict[str, int] = {}
    redacted_field_names: list[str] = []

    for name, value in fields.items():
        if not isinstance(value, str) or not value:
            cleaned[name] = value
            continue
        res = redact_text(value)
        cleaned[name] = res.text
        if res.redacted:
            redacted_field_names.append(name)
            for k, v in res.matches.items():
                aggregate[k] = aggregate.get(k, 0) + v

    if aggregate:
        total = sum(aggregate.values())
        audit_logger.warning(
            "security.secret_redacted context=%s actor=%s fields=%s types=%s total=%d",
            context,
            actor,
            ",".join(redacted_field_names),
            aggregate,
            total,
            extra={
                "event": "secret_redacted",
                "context": context,
                "actor": actor,
                "fields_redacted": redacted_field_names,
                "secret_types": aggregate,
                "total_count": total,
            },
        )

    return cleaned, aggregate

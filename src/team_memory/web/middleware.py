"""Web middleware: API version compatibility and cookie utilities."""

from urllib.parse import unquote

from fastapi import Request


def _decode_api_key_cookie(raw: str) -> str:
    """Decode cookie-stored API key."""
    if not raw:
        return ""
    return unquote(raw)


async def api_version_compat(request: Request, call_next):
    """Rewrite legacy /api/xxx paths to /api/v1/xxx for backward compatibility."""
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/v1/"):
        request.scope["path"] = "/api/v1" + path[4:]
    return await call_next(request)

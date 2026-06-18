"""Daemon HTTP client — 统一 DAEMON_URL 解析 + sync httpx POST。

消除 12 处 DAEMON_URL 硬编码。
"""

import os

import httpx

DAEMON_URL = os.environ.get("TM_DAEMON_URL", "http://127.0.0.1:3901")


def post_hook(hook_name: str, payload: dict, timeout: float = 10.0) -> httpx.Response:
    """POST to daemon hook endpoint. 向 daemon hook 端点发送 POST 请求。

    Args:
        hook_name: Hook endpoint name (e.g. "session_start", "before_prompt").
        payload: JSON payload to send.
        timeout: Request timeout in seconds.

    Returns:
        httpx.Response object.
    """
    return httpx.post(f"{DAEMON_URL}/hooks/{hook_name}", json=payload, timeout=timeout)


async def async_post_hook(hook_name: str, payload: dict, timeout: float = 10.0) -> dict:
    """Async POST to a daemon endpoint. 异步向 daemon 端点发送 POST 请求。

    Uses a path of /hooks/<hook_name> for hook endpoints, and supports
    arbitrary non-hook paths via leading-slash convention:
    if hook_name starts with '/', it is used as-is.

    Args:
        hook_name: Hook name (e.g. "session_timeout_check") or absolute
            path starting with '/' (e.g. "/wiki/index").
        payload: JSON payload to send.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict, or {} on error.
    """
    path = hook_name if hook_name.startswith("/") else f"/hooks/{hook_name}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{DAEMON_URL}{path}", json=payload)
            return resp.json() if resp.content else {}
    except Exception:
        return {}


def post_hook_fire_and_forget(hook_name: str, payload: dict, timeout: float = 10.0) -> None:
    """POST to daemon hook endpoint, swallowing connection errors. 发送 POST 请求，忽略连接错误。

    Used for best-effort notifications (e.g. after_response, mcp_callback).
    """
    try:
        httpx.post(f"{DAEMON_URL}/hooks/{hook_name}", json=payload, timeout=timeout)
    except httpx.ConnectError:
        pass
    except Exception:
        pass

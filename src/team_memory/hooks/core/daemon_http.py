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

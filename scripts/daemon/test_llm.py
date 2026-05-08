#!/usr/bin/env python
"""test_llm.py — Quick LLM connectivity test for the daemon.

Verifies that the LLM proxy (LiteLLM) is reachable and returns valid
chat-completion responses. Uses the same config as FaithfulnessJudge and
the RefinementWorker so operators can validate connectivity before
starting the daemon.

Usage:
    PYTHONPATH=src:scripts python scripts/daemon/test_llm.py
    PYTHONPATH=src:scripts python scripts/daemon/test_llm.py --base-url http://localhost:4000/v1
    PYTHONPATH=src:scripts python scripts/daemon/test_llm.py --model glm-4-flash --timeout 30

Exit codes:
    0  All tests passed
    1  Connection / response error
    2  Invalid / unexpected response format
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


async def _test_llm_connectivity(
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    """Test LLM connectivity by sending a simple chat completion request.

    Returns a dict with test results and metadata.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    result: dict[str, Any] = {
        "url": url,
        "model": model,
        "status": "unknown",
        "latency_ms": 0,
        "error": None,
        "response_content": None,
        "response_model": None,
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            elapsed = time.monotonic() - start
            result["latency_ms"] = round(elapsed * 1000)

            if resp.status_code != 200:
                result["status"] = "http_error"
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            data = resp.json()

            # Validate response structure
            choices = data.get("choices", [])
            if not choices:
                result["status"] = "invalid_response"
                result["error"] = "No 'choices' in response"
                return result

            message = choices[0].get("message", {})
            content = message.get("content", "")
            resp_model = data.get("model", "")

            result["status"] = "ok"
            result["response_content"] = content[:200]
            result["response_model"] = resp_model

    except httpx.ConnectError as e:
        result["status"] = "connection_error"
        result["error"] = f"Cannot connect to {url}: {e}"
    except httpx.TimeoutException as e:
        result["status"] = "timeout"
        result["error"] = f"Request timed out after {timeout}s: {e}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def _test_models_endpoint(
    base_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    """Test /models endpoint to list available models.

    Returns a dict with test results.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    result: dict[str, Any] = {
        "url": url,
        "status": "unknown",
        "models": [],
        "error": None,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                result["status"] = "http_error"
                result["error"] = f"HTTP {resp.status_code}"
                return result

            data = resp.json()
            models = [m.get("id", "?") for m in data.get("data", [])]
            result["status"] = "ok"
            result["models"] = models[:20]  # cap display

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def run_tests(args: argparse.Namespace) -> int:
    """Run all LLM connectivity tests. Returns exit code."""
    # Resolve config from daemon config if not explicitly provided
    base_url = args.base_url
    model = args.model
    api_key = args.api_key
    timeout = args.timeout

    if not base_url or not model:
        try:
            from daemon.config import load_config
            config = load_config()
            ref = config.refinement
            base_url = base_url or ref.base_url
            model = model or ref.model
            if not api_key:
                api_key = os.environ.get(ref.api_key_env, "")
            timeout = timeout or ref.timeout
        except Exception:
            pass

    # Defaults if still not set
    base_url = base_url or "http://localhost:4000/v1"
    model = model or "glm-4-flash"
    timeout = timeout or 30

    print("=" * 60)
    print("TM Daemon — LLM Connectivity Test")
    print("=" * 60)
    print(f"  Base URL : {base_url}")
    print(f"  Model    : {model}")
    print(f"  API Key  : {'***' + api_key[-4:] if api_key and len(api_key) > 4 else '(none)'}")
    print(f"  Timeout  : {timeout}s")
    print()

    # Test 1: List models
    print("[1/2] Testing /models endpoint ...")
    models_result = await _test_models_endpoint(base_url, api_key, timeout)
    if models_result["status"] == "ok":
        n = len(models_result["models"])
        print(f"       OK — {n} model(s) available")
        for m in models_result["models"][:10]:
            print(f"         - {m}")
        if n > 10:
            print(f"         ... and {n - 10} more")
    else:
        print(f"       SKIP — {models_result['error']}")
    print()

    # Test 2: Chat completion
    print(f"[2/2] Testing /chat/completions (model={model}) ...")
    chat_result = await _test_llm_connectivity(base_url, model, api_key, timeout)

    if chat_result["status"] == "ok":
        print(f"       OK — response received in {chat_result['latency_ms']}ms")
        print(f"       Model  : {chat_result['response_model']}")
        print(f"       Content: {chat_result['response_content']!r}")
        print()
        print("All tests passed!")
        return 0
    elif chat_result["status"] == "invalid_response":
        print(f"       FAIL — Invalid response: {chat_result['error']}")
        print()
        print("LLM responded but with unexpected format.")
        return 2
    else:
        print(f"       FAIL — {chat_result['status']}: {chat_result['error']}")
        print()
        print("LLM connectivity test FAILED.")
        if chat_result["status"] == "connection_error":
            print("Hint: Is the LiteLLM proxy running? Try: docker compose up -d litellm")
        elif chat_result["status"] == "timeout":
            print("Hint: The LLM backend may be overloaded. Try increasing --timeout.")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test LLM connectivity for TM Daemon",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="LLM proxy base URL (default: from daemon config or http://localhost:4000/v1)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name (default: from daemon config or glm-4-flash)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key (default: from LITELLM_MASTER_KEY env or daemon config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Request timeout in seconds (default: from daemon config or 30)",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(run_tests(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

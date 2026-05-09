#!/usr/bin/env python
"""test_llm.py — Full connectivity test for TM services.

Tests all external model dependencies:
1. LiteLLM proxy: /models + /chat/completions
2. Ollama embedding: /api/embed
3. Web service: /api/v1/health

Usage:
    python -m team_memory.daemon.test_llm
    python -m team_memory.daemon.test_llm --skip-ollama
    python -m team_memory.daemon.test_llm --skip-web

Exit codes:
    0  All tests passed
    1  Connection / response error
    2  Invalid / unexpected response format
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


async def _test_litellm_models(
    base_url: str, api_key: str, timeout: int,
) -> dict[str, Any]:
    """Test LiteLLM /models endpoint."""
    import httpx

    url = f"{base_url.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    result: dict[str, Any] = {"url": url, "status": "unknown", "models": [], "error": None}
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
            result["models"] = models[:20]
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


async def _test_litellm_chat(
    base_url: str, model: str, api_key: str, timeout: int,
) -> dict[str, Any]:
    """Test LiteLLM /chat/completions."""
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
        "url": url, "model": model, "status": "unknown",
        "latency_ms": 0, "error": None,
        "response_content": None, "response_model": None,
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
            choices = data.get("choices", [])
            if not choices:
                result["status"] = "invalid_response"
                result["error"] = "No 'choices' in response"
                return result

            message = choices[0].get("message", {})
            result["status"] = "ok"
            result["response_content"] = message.get("content", "")[:200]
            result["response_model"] = data.get("model", "")

    except httpx.ConnectError as e:
        result["status"] = "connection_error"
        result["error"] = f"Cannot connect: {e}"
    except httpx.TimeoutException:
        result["status"] = "timeout"
        result["error"] = f"Request timed out after {timeout}s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


async def _test_ollama_embed(
    base_url: str, model: str, timeout: int,
) -> dict[str, Any]:
    """Test Ollama /api/embed endpoint."""
    import httpx

    url = f"{base_url.rstrip('/')}/api/embed"
    payload = {"model": model, "input": ["test embedding"]}

    result: dict[str, Any] = {
        "url": url, "model": model, "status": "unknown",
        "latency_ms": 0, "dimension": 0, "error": None,
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            elapsed = time.monotonic() - start
            result["latency_ms"] = round(elapsed * 1000)

            if resp.status_code != 200:
                result["status"] = "http_error"
                result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                result["status"] = "ok"
                result["dimension"] = len(embeddings[0])
            else:
                result["status"] = "invalid_response"
                result["error"] = "No embeddings in response"

    except httpx.ConnectError:
        result["status"] = "connection_error"
        result["error"] = f"Cannot connect to Ollama at {base_url}"
    except httpx.TimeoutException:
        result["status"] = "timeout"
        result["error"] = f"Ollama timed out after {timeout}s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


async def _test_web_health(
    base_url: str, timeout: int,
) -> dict[str, Any]:
    """Test Web service health endpoint."""
    import httpx

    url = f"{base_url.rstrip('/')}/health"

    result: dict[str, Any] = {
        "url": url, "status": "unknown",
        "latency_ms": 0, "error": None, "detail": None,
    }

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            elapsed = time.monotonic() - start
            result["latency_ms"] = round(elapsed * 1000)

            if resp.status_code != 200:
                result["status"] = "http_error"
                result["error"] = f"HTTP {resp.status_code}"
                return result

            data = resp.json()
            result["status"] = "ok"
            result["detail"] = data.get("status", "unknown")

    except httpx.ConnectError:
        result["status"] = "connection_error"
        result["error"] = f"Cannot connect to Web service at {base_url}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_tests(args: argparse.Namespace) -> int:
    """Run all connectivity tests. Returns exit code."""
    # Resolve config from daemon config
    base_url = args.base_url
    model = args.model
    api_key = args.api_key
    timeout = args.timeout or 30

    if not base_url or not model:
        try:
            from team_memory.daemon.config import load_config
            config = load_config()
            ref = config.refinement
            base_url = base_url or ref.base_url
            model = model or ref.model
            if not api_key:
                api_key = os.environ.get(ref.api_key_env, "")
        except Exception:
            pass

    # Defaults
    llm_base = base_url or "http://localhost:4000/v1"
    llm_model = model or "glm-4-flash"
    ollama_base = args.ollama_base or "http://127.0.0.1:11434"
    ollama_model = args.ollama_model or "qwen3-embedding:0.6b"
    web_base = args.web_base or "http://localhost:9111"

    any_fail = False

    # ── Section 1: LiteLLM ──
    print("=" * 60)
    print("TM Service Connectivity Test")
    print("=" * 60)

    if not args.skip_litellm:
        print(f"\n[1/3] LiteLLM Proxy ({llm_base})")
        print(f"  Model: {llm_model}")

        # Models endpoint
        r = await _test_litellm_models(llm_base, api_key, timeout)
        if r["status"] == "ok":
            n = len(r["models"])
            print(f"  /models     OK ({n} models)")
        else:
            print(f"  /models     SKIP — {r['error']}")
            any_fail = True

        # Chat completion
        r = await _test_litellm_chat(llm_base, llm_model, api_key, timeout)
        if r["status"] == "ok":
            print(f"  /chat       OK ({r['latency_ms']}ms, model={r['response_model']})")
        else:
            label = "FAIL" if r["status"] not in ("timeout",) else "SLOW"
            print(f"  /chat       {label} — {r['error']}")
            if r["status"] == "http_error" and "429" in (r["error"] or ""):
                print("              (rate-limited, connectivity OK)")
            else:
                any_fail = True
    else:
        print("\n[1/3] LiteLLM Proxy — SKIPPED")

    # ── Section 2: Ollama ──
    if not args.skip_ollama:
        print(f"\n[2/3] Ollama Embedding ({ollama_base})")
        print(f"  Model: {ollama_model}")

        r = await _test_ollama_embed(ollama_base, ollama_model, timeout)
        if r["status"] == "ok":
            print(f"  /api/embed  OK ({r['latency_ms']}ms, dim={r['dimension']})")
        else:
            print(f"  /api/embed  FAIL — {r['error']}")
            any_fail = True
    else:
        print("\n[2/3] Ollama Embedding — SKIPPED")

    # ── Section 3: Web Service ──
    if not args.skip_web:
        print(f"\n[3/3] Web Service ({web_base})")

        r = await _test_web_health(web_base, timeout)
        if r["status"] == "ok":
            print(f"  /health     OK ({r['latency_ms']}ms, status={r['detail']})")
        else:
            print(f"  /health     FAIL — {r['error']}")
            any_fail = True
    else:
        print("\n[3/3] Web Service — SKIPPED")

    # ── Summary ──
    print("\n" + "=" * 60)
    if any_fail:
        print("RESULT: Some tests FAILED")
        return 1
    else:
        print("RESULT: All tests PASSED")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test TM service connectivity (LiteLLM + Ollama + Web)",
    )
    parser.add_argument("--base-url", default="", help="LiteLLM base URL")
    parser.add_argument("--model", default="", help="LiteLLM model name")
    parser.add_argument("--api-key", default="", help="LiteLLM API key")
    parser.add_argument("--timeout", type=int, default=0, help="Request timeout (default: 30s)")
    parser.add_argument("--ollama-base", default="", help="Ollama base URL (default: http://127.0.0.1:11434)")
    parser.add_argument("--ollama-model", default="", help="Ollama embedding model (default: qwen3-embedding:0.6b)")
    parser.add_argument("--web-base", default="", help="Web service base URL (default: http://localhost:9111)")
    parser.add_argument("--skip-litellm", action="store_true", help="Skip LiteLLM test")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama test")
    parser.add_argument("--skip-web", action="store_true", help="Skip Web service test")

    args = parser.parse_args()
    exit_code = asyncio.run(run_tests(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

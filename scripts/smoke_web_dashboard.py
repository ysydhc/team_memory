#!/usr/bin/env python3
"""Smoke test: GET /health and GET /api/v1/stats to diagnose dashboard failure.

Usage:
  python scripts/smoke_web_dashboard.py [--api-key KEY] [--port PORT]
  Default API key CHANGE_ME_your_api_key; port 9111.

Prints: health (including dashboard_stats), and /api/v1/stats response or error.
No browser required. For full UI test use Playwright (see README).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def get_health(port: int) -> dict | None:
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (503, 500) and e.fp:
            try:
                return json.loads(e.read().decode())
            except Exception:
                pass
        print(f"Health fetch failed: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"Health fetch failed: {e}")
        return None


def get_stats(port: int, api_key: str) -> tuple[int, str, dict | None]:
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/api/v1/stats",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            return r.getcode(), raw, data
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            data = json.loads(body) if body.strip().startswith("{") else None
        except json.JSONDecodeError:
            data = None
        return e.code, body, data
    except Exception as e:
        return 0, str(e), None


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test Web UI dashboard (health + /stats)")
    ap.add_argument("--api-key", default="CHANGE_ME_your_api_key", help="API key")
    ap.add_argument("--port", type=int, default=9111, help="Web port")
    args = ap.parse_args()

    # 1. Health and dashboard_stats diagnostic
    health = get_health(args.port)
    if health:
        checks = health.get("checks", {})
        ds = checks.get("dashboard_stats", {})
        print("Health status:", health.get("status"))
        if ds:
            print("dashboard_stats:", ds.get("status"), ds.get("error", ""))
            if ds.get("status") == "down" and ds.get("ops_hint"):
                print("ops_hint:", ds["ops_hint"])
        if checks.get("database", {}).get("status") == "down":
            print("Database is down → dashboard will fail. Start DB (e.g. docker compose up -d).")
    else:
        print("Web server not reachable at http://localhost:%s" % args.port)
        return 1

    # 2. GET /api/v1/stats (what dashboard calls)
    print("\nGET /api/v1/stats:")
    code, body, data = get_stats(args.port, args.api_key)
    if code == 200 and data is not None:
        print("  OK", list(data.keys()))
        return 0
    print("  HTTP", code)
    if data and isinstance(data, dict):
        print("  detail:", data.get("detail"))
        print("  ops_hint:", data.get("ops_hint"))
        print("  message:", data.get("message"))
    else:
        print("  body (first 200 chars):", (body or "")[:200])
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Stdin JSON parsing — 统一 stdin 解析逻辑。"""

import json
import sys


def parse_stdin_json() -> dict:
    """Read and parse JSON from stdin. 从 stdin 读取并解析 JSON。

    Returns empty dict on parse failure (consistent with existing hook behavior).
    """
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError, Exception):
        return {}

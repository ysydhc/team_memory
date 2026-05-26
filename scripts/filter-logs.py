#!/usr/bin/env python3
"""Filter TM daemon logs — structured JSON + legacy text lines.

Usage:
  python scripts/filter-logs.py --session abc123 < /tmp/tm-daemon.log
  python scripts/filter-logs.py --source cursor --level ERROR
  python scripts/filter-logs.py --tail 50
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", help="Filter by session_id")
    p.add_argument("--source", help="Filter by source (cursor/claude/hermes)")
    p.add_argument("--level", help="Filter by level (INFO/WARNING/ERROR)")
    p.add_argument("--tail", type=int, default=0, help="Show last N lines")
    args = p.parse_args()

    lines = sys.stdin.readlines()
    if args.tail:
        lines = lines[-args.tail :]

    for line in lines:
        line = line.rstrip("\n")
        # JSON line
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print(line)
                continue
            if args.session and d.get("session") != args.session:
                continue
            if args.source and d.get("source") != args.source:
                continue
            if args.level and d.get("level", "").upper() != args.level.upper():
                continue
            # Pretty print
            ts = d.get("ts", "")[:19]
            lv = d.get("level", "")[:5]
            ev = d.get("event", "")
            sid = d.get("session", "")[:12]
            src = d.get("source", "")
            extra = {k: v for k, v in d.items() if k not in ("ts", "level", "event", "session", "source")}
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            print(f"{ts} {lv:5s} [{src:6s}] {ev:20s} sid={sid:12s} {extra_str}"[:160])
        else:
            print(line)


if __name__ == "__main__":
    main()

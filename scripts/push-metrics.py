#!/usr/bin/env python3
"""Push metrics summary to Feishu / 推送指标摘要到飞书."""

import os

import httpx


def main():
    webhook = os.environ.get("FEISHU_WEBHOOK_URL", "")
    if not webhook:
        print("FEISHU_WEBHOOK_URL not set")
        return

    # Resolve db_path same as CLI
    db_path = os.environ.get("TM_PIPELINE_DB_PATH", os.path.expanduser("~/.cache/tm-pipeline/drafts.db"))

    from team_memory.daemon.metrics_collector import MetricsCollector

    mc = MetricsCollector(db_path=db_path)
    s = mc.summary(hours=6)

    lines = ["**TM Metrics (6h)**"]
    for name, labels_dict in sorted(s.items()):
        parts = [
            f"{k.replace('source=', '')}: {int(v)}" if v == int(v) else f"{k.replace('source=', '')}: {v:.2f}"
            for k, v in sorted(labels_dict.items())
        ]
        lines.append(f"{name}: {', '.join(parts)}")

    text = "\n".join(lines)
    try:
        httpx.post(webhook, json={"msg_type": "text", "content": {"text": text}}, timeout=10)
    except Exception as e:
        print(f"Push failed: {e}")


if __name__ == "__main__":
    main()

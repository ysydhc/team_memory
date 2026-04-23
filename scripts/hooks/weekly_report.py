"""WeeklyReport — generates a weekly summary of the memory system's usage.

Calls the TeamMemory API for search_log stats and formats a human-readable
report showing retrieval counts, hit rates, and system health assessment.
"""
from __future__ import annotations

import logging

from scripts.hooks.shared import TMClient

logger = logging.getLogger("team_memory.hooks.weekly_report")


class WeeklyReport:
    """Generate weekly memory system usage reports.

    Usage::

        report = WeeklyReport("http://localhost:3900")
        text = await report.generate(days=7)
        print(text)
    """

    def __init__(self, tm_url: str) -> None:
        self._tm = TMClient(tm_url)

    async def generate(self, days: int = 7) -> str:
        """Generate memory system weekly report.

        Calls the TM API to fetch search_log stats, then formats the report.

        Args:
            days: Number of days to look back (default 7).

        Returns:
            Formatted report string.
        """
        try:
            stats = await self._tm._call(
                "memory_search_log_stats",
                {"days": days},
            )
        except Exception:
            logger.exception("Failed to fetch weekly stats from TM API")
            stats = {"total": 0, "hit": 0, "used": 0, "use_rate": 0.0}

        return self.format_report(stats)

    def format_report(
        self,
        stats: dict,
        new_count: int = 0,
        promoted_count: int = 0,
    ) -> str:
        """Format weekly report from stats dict.

        Args:
            stats: Dict with total, hit, used, use_rate keys.
            new_count: Number of new traces this week.
            promoted_count: Number of traces promoted to knowledge.

        Returns:
            Formatted report string.
        """
        use_rate = stats.get("use_rate", 0)
        if use_rate > 0.4:
            assessment = "系统有效 ✓"
        elif use_rate > 0.2:
            assessment = "需要调整 ⚠"
        else:
            assessment = "系统无效 ✗"

        report = f"""
记忆系统周报
══════════════════════════

检索：{stats.get('total', 0)} 次
命中：{stats.get('hit', 0)} 次
使用：{stats.get('used', 0)} 次
使用率：{use_rate:.1%}

本周新增：{new_count} 条痕迹
本周提升：{promoted_count} 条 → 知识

评估：{assessment}"""
        return report

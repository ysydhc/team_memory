"""Panoramic status monitor and live TUI dashboard for TeamMemory."""

from __future__ import annotations

from team_memory.daemon.observe.status_monitor.collectors import (
    _WIKI_DB_CANDIDATES,
    _clean_query_display,
    _inspect_local_sqlite,
    _lookup_local_wiki_entries,
    _parse_json_field,
    _parse_list_field,
    _probe_daemon_status,
    _probe_web_health,
    _resolve_endpoints_and_paths,
    _sqlite_total_size_bytes,
    collect_status_snapshot,
)
from team_memory.daemon.observe.status_monitor.panels.plain import (
    _badge_plain,
    render_status_plain,
)
from team_memory.daemon.observe.status_monitor.panels.rich_dashboard import (
    _draw_bar,
    build_rich_renderable,
)
from team_memory.daemon.observe.status_monitor.panels.trace_panel import (
    build_recall_trace_rich,
    render_recall_trace_plain,
)
from team_memory.daemon.observe.status_monitor.pg_collector import (
    _fetch_recent_recalls_from_conn,
    _inspect_postgres_knowledge,
)
from team_memory.daemon.observe.status_monitor.runner import (
    _make_console,
    _print_trace_output,
    run_status_monitor,
)
from team_memory.daemon.observe.status_monitor.trace_collector import inspect_recall_trace

__all__ = [
    "_WIKI_DB_CANDIDATES",
    "_badge_plain",
    "_clean_query_display",
    "_draw_bar",
    "_fetch_recent_recalls_from_conn",
    "_inspect_local_sqlite",
    "_inspect_postgres_knowledge",
    "_lookup_local_wiki_entries",
    "_make_console",
    "_parse_json_field",
    "_parse_list_field",
    "_print_trace_output",
    "_probe_daemon_status",
    "_probe_web_health",
    "_resolve_endpoints_and_paths",
    "_sqlite_total_size_bytes",
    "build_recall_trace_rich",
    "build_rich_renderable",
    "collect_status_snapshot",
    "inspect_recall_trace",
    "render_recall_trace_plain",
    "render_status_plain",
    "run_status_monitor",
]

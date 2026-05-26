"""Hooks package — platform-specific hook scripts for Claude, Cursor, and Hermes.

Structure:
  core/          — shared utilities (daemon_http, stdin, mcp_callback, output)
  platforms/     — platform-specific hooks (claude/, cursor/, hermes/)
  integrations/  — cross-platform integrations (weekly_report, git_watcher)
"""

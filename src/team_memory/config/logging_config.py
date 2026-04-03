"""Logging configuration.

Named ``logging_config`` to avoid shadowing the stdlib ``logging`` module.
"""

from __future__ import annotations

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_io_enabled: bool = False
    log_io_detail: str = "mcp"
    log_io_truncate: int = 300
    log_file_enabled: bool = False
    log_file_path: str = "logs/team_memory.log"
    log_file_backup_count: int = 5
    log_file_max_bytes: int = 10 * 1024 * 1024

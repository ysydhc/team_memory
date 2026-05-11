"""StructuredLogger — 包装标准 logger，输出 JSON 格式日志。

每条日志自动注入 source，可选 session_id 等字段。
"""

from __future__ import annotations

import json
import logging
from typing import Any


class StructuredLogger:
    """JSON-line logger with automatic source injection / JSON 行日志。"""

    def __init__(self, logger: logging.Logger, source: str = "daemon") -> None:
        self._logger = logger
        self._source = source

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        data: dict[str, Any] = {"event": event, **kwargs}
        if self._source:
            data["source"] = self._source
        self._logger.log(level, json.dumps(data, ensure_ascii=False, default=str))

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

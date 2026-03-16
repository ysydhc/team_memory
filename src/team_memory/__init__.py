"""team_memory - Team experience database service based on MCP protocol."""

import sys
import warnings

# 屏蔽 jieba/pkg_resources/websockets 等第三方库的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, append=False)


class _StderrFilter:
    """过滤 stderr 中的 DeprecationWarning 相关行。"""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""
        self._skip_continuation = False

    def _should_skip(self, line: str) -> bool:
        if self._skip_continuation:
            self._skip_continuation = False
            return line.startswith(("  ", "\t"))
        if "DeprecationWarning" in line:
            self._skip_continuation = True
            return True
        if "pkg_resources" in line and ("deprecated" in line.lower() or "declare_namespace" in line):
            return True
        if "declare_namespace" in line or "Implementing implicit namespace" in line:
            return True
        if "jieba" in line and "pkg_resources" in line:
            return True
        return False

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf or "\r" in self._buf:
            line, sep, rest = self._buf.partition("\n")
            if not sep:
                line, sep, rest = self._buf.partition("\r")
            self._buf = rest
            if not self._should_skip(line):
                self._stream.write(line + sep)
        return len(s)

    def flush(self):
        if self._buf and "DeprecationWarning" not in self._buf:
            self._stream.write(self._buf)
        self._buf = ""
        self._skip_continuation = False
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if not isinstance(sys.stderr, _StderrFilter):
    sys.stderr = _StderrFilter(sys.stderr)

__version__ = "0.1.2"

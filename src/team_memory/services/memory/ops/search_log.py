"""Search log writing — canonical implementation lives in the facade module.

This re-export exists so that ``ops.search_log.write_search_log`` is importable
for code that references the sub-module directly.
"""

from __future__ import annotations

from team_memory.services.memory.memory_operations import write_search_log  # noqa: F401

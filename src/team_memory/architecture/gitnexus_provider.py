"""GitNexus-backed architecture provider (stub when bridge unavailable).

Full implementation (T3) will call bridge HTTP API; this module provides
a placeholder that returns available: false when bridge_url is empty.
"""

from __future__ import annotations

from team_memory.architecture.base import ArchitectureProvider
from team_memory.config import ArchitectureGitnexusConfig
from team_memory.web.architecture_models import (
    ArchitectureContext,
    ArchitectureGraph,
    ClusterMembers,
    ClusterSummary,
    ImpactResult,
)


class GitNexusProvider(ArchitectureProvider):
    """Architecture data from GitNexus (bridge or MCP).

    When bridge_url is empty, all methods return unavailable/empty responses.
    T3 will add real HTTP calls to the bridge.
    """

    def __init__(self, config: ArchitectureGitnexusConfig) -> None:
        self._config = config
        self._available = bool(config.bridge_url and config.bridge_url.strip())

    async def get_context(self, repo: str | None = None) -> ArchitectureContext:
        if not self._available:
            return ArchitectureContext(
                available=False,
                reason="Bridge not configured (architecture.gitnexus.bridge_url empty)",
                provider="gitnexus",
            )
        # T3: call bridge GET /context
        return ArchitectureContext(
            available=False,
            reason="Bridge not yet implemented (T3)",
            provider="gitnexus",
        )

    async def get_graph(
        self,
        repo: str | None = None,
        cluster: str | None = None,
        file_path: str | None = None,
    ) -> ArchitectureGraph:
        return ArchitectureGraph()

    async def get_clusters(self, repo: str | None = None) -> list[ClusterSummary]:
        return []

    async def get_cluster(self, name: str, repo: str | None = None) -> ClusterMembers:
        return ClusterMembers(name=name, members=[])

    async def get_impact(
        self,
        path: str,
        depth: int = 2,
        repo: str | None = None,
    ) -> ImpactResult:
        return ImpactResult(upstream=[], downstream=[])

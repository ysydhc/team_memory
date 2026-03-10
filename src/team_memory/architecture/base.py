"""Architecture provider protocol and factory.

Contract aligned with docs/exec-plans/completed/code-arch-viz-gitnexus/
code-arch-viz-provider-interface.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from team_memory.schemas_architecture import (
    ArchitectureContext,
    ArchitectureGraph,
    ClusterMembers,
    ClusterSummary,
    GraphNode,
    ImpactResult,
)


class ArchitectureProvider(ABC):
    """Abstract provider for architecture (code graph) data.

    Implementations (e.g. GitNexusProvider, BuiltinProvider) return
    unified response types; routing layer calls only this interface.
    """

    @abstractmethod
    async def get_context(self, repo: str | None = None) -> ArchitectureContext:
        """Return codebase overview and staleness."""
        ...

    @abstractmethod
    async def get_graph(
        self,
        repo: str | None = None,
        cluster: str | None = None,
        file_path: str | None = None,
        max_depth: int = 2,
    ) -> ArchitectureGraph:
        """Return nodes and edges; when no cluster, expand from roots by max_depth hops."""
        ...

    @abstractmethod
    async def get_clusters(self, repo: str | None = None) -> list[ClusterSummary]:
        """Return list of clusters."""
        ...

    @abstractmethod
    async def get_cluster(self, name: str, repo: str | None = None) -> ClusterMembers:
        """Return members of one cluster."""
        ...

    @abstractmethod
    async def get_impact(
        self,
        path: str,
        depth: int = 2,
        repo: str | None = None,
    ) -> ImpactResult:
        """Return upstream/downstream impact for the given path."""
        ...

    @abstractmethod
    async def search_nodes(
        self,
        q: str,
        scope: str = "global",
        clusters: list[str] | None = None,
        repo: str | None = None,
    ) -> list[GraphNode]:
        """Search nodes by name or filePath. scope: global|cluster. clusters: multi-select."""
        ...

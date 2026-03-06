"""Architecture provider protocol and factory.

Contract aligned with docs/design-docs/code-arch-viz/code-arch-viz-provider-interface.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from team_memory.web.architecture_models import (
    ArchitectureContext,
    ArchitectureGraph,
    ClusterMembers,
    ClusterSummary,
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
    ) -> ArchitectureGraph:
        """Return nodes and edges for visualization; focus_node_id when file_path given."""
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

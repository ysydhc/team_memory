"""GitNexus-backed architecture provider via Bridge HTTP API.

Calls tools/gitnexus-bridge endpoints; returns available: false when
bridge_url is empty or bridge is unreachable.
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

import httpx

from team_memory.architecture.base import ArchitectureProvider
from team_memory.config import ArchitectureGitnexusConfig
from team_memory.schemas_architecture import (
    ArchitectureContext,
    ArchitectureGraph,
    ClusterMember,
    ClusterMembers,
    ClusterSummary,
    GraphEdge,
    GraphNode,
    ImpactItem,
    ImpactResult,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_GRAPH_DEFAULT_TIMEOUT = 60.0  # no-cluster BFS expansion can take 20+ seconds


class BridgeUnavailableError(Exception):
    """Raised when Bridge HTTP call fails (connection refused, timeout, 5xx)."""


class GitNexusProvider(ArchitectureProvider):
    """Architecture data from GitNexus Bridge.

    When bridge_url is empty, all methods return unavailable/empty responses.
    """

    def __init__(self, config: ArchitectureGitnexusConfig) -> None:
        self._config = config
        self._available = bool(config.bridge_url and config.bridge_url.strip())
        self._base = (config.bridge_url or "").rstrip("/")

    def _url(self, endpoint: str, **params: str | int | list[str] | None) -> str:
        filtered: dict[str, str | int | list[str]] = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, list):
                if v:
                    filtered[k] = v
                continue
            filtered[k] = str(v)
        q = urlencode(filtered, doseq=True) if filtered else ""
        return f"{self._base}{endpoint}" + (f"?{q}" if q else "")

    async def get_context(self, repo: str | None = None) -> ArchitectureContext:
        if not self._available:
            return ArchitectureContext(
                available=False,
                reason="Bridge not configured (architecture.gitnexus.bridge_url empty)",
                provider="gitnexus",
            )
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(self._url("/context", repo=repo))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /context failed: %s", e)
            return ArchitectureContext(
                available=False,
                reason=f"Bridge unreachable: {e!s}",
                provider="gitnexus",
            )
        return ArchitectureContext(
            available=data.get("available", False),
            repo_name=data.get("repo_name"),
            symbols=data.get("symbols"),
            relationships=data.get("relationships"),
            processes=data.get("processes"),
            stale=data.get("stale"),
            provider="gitnexus",
            reason=data.get("reason"),
            default_clusters=getattr(self._config, "default_clusters", None) or [],
        )

    async def get_graph(
        self,
        repo: str | None = None,
        cluster: str | None = None,
        clusters: list[str] | None = None,
        file_path: str | None = None,
        max_depth: int = 2,
        ensure_nodes: list[str] | None = None,
    ) -> ArchitectureGraph:
        if not self._available:
            return ArchitectureGraph()
        effective = clusters if clusters else ([cluster] if cluster else None)
        timeout = _GRAPH_DEFAULT_TIMEOUT if not effective else _TIMEOUT
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                url = self._url(
                    "/graph",
                    repo=repo,
                    cluster=effective if effective else None,
                    file_path=file_path,
                    max_depth=max_depth,
                    ensure_nodes=ensure_nodes or None,
                )
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /graph failed: %s", e)
            raise BridgeUnavailableError(f"Bridge unreachable: {e!s}") from e
        nodes = [GraphNode(**n) for n in data.get("nodes", [])]
        edges = [GraphEdge(**e) for e in data.get("edges", [])]
        return ArchitectureGraph(
            nodes=nodes,
            edges=edges,
            focus_node_id=data.get("focus_node_id"),
        )

    async def get_clusters(self, repo: str | None = None) -> list[ClusterSummary]:
        if not self._available:
            return []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(self._url("/clusters", repo=repo))
                r.raise_for_status()
                items = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /clusters failed: %s", e)
            return []
        return [ClusterSummary(**x) for x in items]

    async def get_cluster(self, name: str, repo: str | None = None) -> ClusterMembers:
        if not self._available:
            return ClusterMembers(name=name, members=[])
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(self._url(f"/cluster/{name}", repo=repo))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /cluster/%s failed: %s", name, e)
            return ClusterMembers(name=name, members=[])
        members = [ClusterMember(**m) for m in data.get("members", [])]
        return ClusterMembers(name=data.get("name", name), members=members)

    async def get_impact(
        self,
        path: str,
        depth: int = 2,
        repo: str | None = None,
    ) -> ImpactResult:
        if not self._available:
            return ImpactResult(upstream=[], downstream=[])
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.get(
                    self._url("/impact", path=path, depth=depth, repo=repo)
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /impact failed: %s", e)
            return ImpactResult(upstream=[], downstream=[])
        upstream = [ImpactItem(**x) for x in data.get("upstream", [])]
        downstream = [ImpactItem(**x) for x in data.get("downstream", [])]
        return ImpactResult(upstream=upstream, downstream=downstream)

    async def search_nodes(
        self,
        q: str,
        scope: str = "global",
        clusters: list[str] | None = None,
        repo: str | None = None,
    ) -> list[GraphNode]:
        if not self._available:
            return []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                params: dict = {"q": q, "scope": scope, "repo": repo}
                if clusters:
                    params["clusters"] = clusters
                r = await client.get(self._url("/search", **params))
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            logger.warning("Bridge /search failed: %s", e)
            raise BridgeUnavailableError(f"Bridge unreachable: {e!s}") from e
        return [GraphNode(**n) for n in data.get("nodes", [])]

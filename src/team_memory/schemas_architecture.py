"""Pydantic models for architecture API responses.

Contract aligned with docs/exec-plans/completed/code-arch-viz-gitnexus/
code-arch-viz-provider-interface.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArchitectureContext(BaseModel):
    """Codebase overview and staleness."""

    available: bool
    repo_name: str | None = None
    symbols: int | None = None
    relationships: int | None = None
    processes: int | None = None
    stale: bool | None = None
    provider: Literal["gitnexus", "builtin"] | None = None
    reason: str | None = None
    default_clusters: list[str] = Field(
        default_factory=list,
        description="Default clusters for graph (project entry points from config).",
    )


class GraphNode(BaseModel):
    """Single node in architecture graph."""

    id: str
    label: str | None = None
    kind: str | None = None
    path: str | None = None
    meta: dict[str, Any] | None = None


class GraphEdge(BaseModel):
    """Single edge in architecture graph."""

    source: str
    target: str
    type: str | None = None
    meta: dict[str, Any] | None = None


class ArchitectureGraph(BaseModel):
    """Nodes and edges for architecture visualization."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    focus_node_id: str | None = None


class ClusterSummary(BaseModel):
    """Cluster list item."""

    name: str
    cohesion: float | None = None
    member_count: int | None = None


class ClusterMember(BaseModel):
    """Single member in a cluster."""

    id: str
    path: str | None = None
    kind: str | None = None


class ClusterMembers(BaseModel):
    """Members of one cluster."""

    name: str
    members: list[ClusterMember] = Field(default_factory=list)


class ImpactItem(BaseModel):
    """Single item in impact result (upstream/downstream)."""

    id: str
    path: str | None = None
    depth: int | None = None


class ImpactResult(BaseModel):
    """Blast radius: who depends on target, whom target depends on."""

    upstream: list[ImpactItem] = Field(default_factory=list)
    downstream: list[ImpactItem] = Field(default_factory=list)


class ExperienceRef(BaseModel):
    """Experience linked to an architecture node (TM-side)."""

    experience_id: str
    title: str | None = None
    node: str | None = None


class SearchNodesResponse(BaseModel):
    """Response for architecture node search."""

    nodes: list[GraphNode] = Field(default_factory=list)

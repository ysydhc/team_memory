"""Architecture (code graph) API routes.

Provides context, graph, clusters, impact, search, and experiences endpoints.
Provider is injected via get_provider(settings.architecture); returns 503
or available:false when provider is not configured.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from team_memory.architecture.factory import get_provider
from team_memory.architecture.gitnexus_provider import BridgeUnavailableError
from team_memory.auth.provider import User
from team_memory.schemas_architecture import (
    ArchitectureContext,
    ArchitectureGraph,
    ClusterMembers,
    ClusterSummary,
    ExperienceRef,
    ImpactResult,
    SearchNodesResponse,
)
from team_memory.storage.database import get_session
from team_memory.storage.repository import ExperienceRepository
from team_memory.web import app as app_module
from team_memory.web.app import _get_db_url, _resolve_project, get_optional_user

router = APIRouter(prefix="/architecture", tags=["architecture"])

# Whitelist for search q: only [a-zA-Z0-9_./\-: ] allowed
_SEARCH_Q_PATTERN = re.compile(r"^[a-zA-Z0-9_./\-: ]+$")


def _cfg():
    return app_module._settings


def _get_provider():
    """Get ArchitectureProvider from config; None when not configured."""
    settings = _cfg()
    if not settings:
        return None
    return get_provider(settings.architecture)


@router.get("/context")
async def get_architecture_context(
    repo: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> ArchitectureContext:
    """Return codebase overview and staleness. 200 + available:false when no bridge."""
    provider = _get_provider()
    if provider is None:
        return ArchitectureContext(
            available=False,
            reason="Architecture provider not configured (provider=builtin or missing)",
        )
    return await provider.get_context(repo=repo)


@router.get("/graph")
async def get_architecture_graph(
    repo: str | None = None,
    cluster: str | None = None,
    clusters: list[str] | None = Query(None),
    file_path: str | None = None,
    max_depth: int = 2,
    ensure_nodes: list[str] | None = Query(None),
    user: User | None = Depends(get_optional_user),
) -> ArchitectureGraph:
    """Return nodes and edges. clusters: multi-select; cluster: single.
    ensure_nodes: node ids to include when using clusters.
    Returns empty graph when provider not configured (avoids 503)."""
    provider = _get_provider()
    if provider is None:
        return ArchitectureGraph(nodes=[], edges=[])
    effective_clusters = clusters if clusters else ([cluster] if cluster else None)
    try:
        return await provider.get_graph(
            repo=repo,
            cluster=cluster,
            clusters=effective_clusters,
            file_path=file_path,
            max_depth=max_depth,
            ensure_nodes=ensure_nodes,
        )
    except BridgeUnavailableError:
        return ArchitectureGraph(nodes=[], edges=[])


@router.get("/clusters")
async def get_architecture_clusters(
    repo: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> list[ClusterSummary]:
    """Return list of clusters. Returns [] when provider not configured."""
    provider = _get_provider()
    if provider is None:
        return []
    return await provider.get_clusters(repo=repo)


@router.get("/cluster/{name}")
async def get_architecture_cluster(
    name: str,
    repo: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> ClusterMembers:
    """Return members of one cluster. Returns empty when provider not configured."""
    provider = _get_provider()
    if provider is None:
        return ClusterMembers(name=name, members=[])
    return await provider.get_cluster(name=name, repo=repo)


@router.get("/search")
async def search_architecture_nodes(
    q: str | None = Query(None),
    scope: str = Query("global"),
    cluster: str | None = Query(None),
    clusters: list[str] | None = Query(None),
    repo: str | None = Query(None),
    user: User | None = Depends(get_optional_user),
) -> SearchNodesResponse:
    """Search nodes by name or filePath. scope: global|cluster. clusters: multi-select."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    if not _SEARCH_Q_PATTERN.match(q):
        raise HTTPException(
            status_code=400,
            detail="q contains invalid characters (allowed: a-zA-Z0-9_./\\-: )",
        )
    if scope not in ("global", "cluster"):
        raise HTTPException(
            status_code=400,
            detail="scope must be global or cluster",
        )
    effective_clusters = clusters if clusters else ([cluster] if cluster else [])
    if scope == "cluster" and not effective_clusters:
        raise HTTPException(
            status_code=400,
            detail="cluster or clusters is required when scope=cluster",
        )
    for c in effective_clusters or []:
        if c and not _SEARCH_Q_PATTERN.match(c):
            raise HTTPException(
                status_code=400,
                detail="cluster contains invalid characters (allowed: a-zA-Z0-9_./\\-: )",
            )
    provider = _get_provider()
    if provider is None:
        return SearchNodesResponse(nodes=[])
    try:
        nodes = await provider.search_nodes(
            q=q,
            scope=scope,
            clusters=effective_clusters,
            repo=repo,
        )
    except BridgeUnavailableError:
        return SearchNodesResponse(nodes=[])
    return SearchNodesResponse(nodes=nodes)


@router.get("/node-clusters")
async def get_node_clusters(
    node: str,
    repo: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> dict:
    """Return clusters containing the node. Empty when provider not configured."""
    provider = _get_provider()
    if provider is None:
        return {"clusters": []}
    cfg = _cfg()
    bridge_url = getattr(cfg.architecture.gitnexus, "bridge_url", None) or ""
    bridge_url = (bridge_url or "").rstrip("/")
    if not bridge_url:
        return {"clusters": []}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            params = {"node": node}
            if repo:
                params["repo"] = repo
            r = await client.get(f"{bridge_url}/node-clusters", params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError:
        return {"clusters": []}


@router.get("/impact")
async def get_architecture_impact(
    path: str,
    depth: int = 2,
    repo: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> ImpactResult:
    """Return upstream/downstream impact. Empty when provider not configured."""
    provider = _get_provider()
    if provider is None:
        return ImpactResult(upstream=[], downstream=[])
    try:
        return await provider.get_impact(path=path, depth=depth, repo=repo)
    except Exception:
        return ImpactResult(upstream=[], downstream=[])


@router.get("/experiences")
async def get_architecture_experiences(
    node: str,
    repo: str | None = None,
    project: str | None = None,
    user: User | None = Depends(get_optional_user),
) -> list[ExperienceRef]:
    """Return experiences linked to an architecture node (node_key = GitNexus filePath)."""
    try:
        resolved_project = _resolve_project(project)
        current_user = user.name if user else None
        db_url = _get_db_url()
        async with get_session(db_url) as session:
            repo_inst = ExperienceRepository(session)
            items = await repo_inst.list_experiences_by_node(
                node_key=node,
                project=resolved_project,
                current_user=current_user,
            )
        return [ExperienceRef(**item) for item in items]
    except Exception:
        return []


@router.get("/open-in-editor")
async def get_open_in_editor_url(
    path: str,
    user: User | None = Depends(get_optional_user),
) -> dict:
    """Return vscode://file/ URL for opening the file in editor. path = relative file path."""
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="path is required")
    path = path.strip().lstrip("/")
    cfg = _cfg()
    if not cfg or not cfg.architecture:
        raise HTTPException(
            status_code=503,
            detail="Architecture provider not configured",
        )
    bridge_url = getattr(cfg.architecture.gitnexus, "bridge_url", None) or ""
    bridge_url = (bridge_url or "").rstrip("/")
    if not bridge_url:
        return {"path": path, "open_url": None}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bridge_url}/repo-root")
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return {"path": path, "open_url": None}
    repo_root = data.get("repo_root")
    if not repo_root:
        return {"path": path, "open_url": None}
    abs_path = str(Path(repo_root) / path).replace("\\", "/")
    open_url = f"vscode://file/{abs_path}"
    return {"path": path, "open_url": open_url}

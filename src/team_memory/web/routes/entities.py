"""Entity graph routes (L2.5).

GET /api/v1/entities?q=<name>&project=<proj>&limit=20
    Search entities by name/alias prefix.

GET /api/v1/entities/{entity_id}/graph?depth=2
    Return the entity and all nodes/edges within `depth` hops.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from team_memory.auth.provider import User
from team_memory.web.app import _get_db_url, _resolve_project
from team_memory.web.auth_session import get_current_user

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("")
async def search_entities_api(
    q: str = Query(default="", description="Entity name prefix to search"),
    project: str | None = Query(
        default=None,
        description="Project filter (default: current project)",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Max results"),
    user: User = Depends(get_current_user),
):
    """Search entities by name or alias prefix."""
    from team_memory.services.entity_search import search_entities

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    resolved_project = _resolve_project(project)
    db_url = _get_db_url()

    entities = await search_entities(
        db_url=db_url,
        q=q.strip(),
        projects=[resolved_project, "default"] if resolved_project != "default" else ["default"],
        limit=limit,
    )
    return {"entities": entities, "count": len(entities)}


@router.get("/{entity_id}/graph")
async def get_entity_graph_api(
    entity_id: str,
    depth: int = Query(default=2, ge=1, le=3, description="Graph traversal depth"),
    project: str | None = Query(default=None, description="Project filter"),
    user: User = Depends(get_current_user),
):
    """Return the entity and all connected nodes/edges within `depth` hops."""
    import uuid as _uuid

    from team_memory.services.entity_search import get_entity_graph

    # Validate UUID format
    try:
        _uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid entity_id UUID format")

    db_url = _get_db_url()

    try:
        graph = await get_entity_graph(
            db_url=db_url,
            entity_id=entity_id,
            max_depth=depth,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph traversal failed: {exc}")

    if not graph["nodes"]:
        raise HTTPException(status_code=404, detail="Entity not found")

    return graph

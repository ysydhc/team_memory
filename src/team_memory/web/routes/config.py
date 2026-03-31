"""Runtime config API (retrieval, search) — in-memory; restart restores YAML."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from team_memory.auth.provider import User
from team_memory.config import RetrievalConfig, SearchConfig
from team_memory.web import app as app_module
from team_memory.web.app import get_current_user
from team_memory.web.dependencies import require_role

logger = logging.getLogger("team_memory.web")

router = APIRouter(tags=["config"])


def _service():
    if app_module._service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return app_module._service


def _settings():
    if app_module._settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")
    return app_module._settings


def _model_dump_compat(model_cls: type[BaseModel], instance: Any) -> dict[str, Any]:
    """Dump config instance; works with real Pydantic models and MagicMock in tests."""
    md = getattr(instance, "model_dump", None)
    if callable(md):
        try:
            raw = md()
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
    defaults = model_cls().model_dump()
    return {k: getattr(instance, k, defaults[k]) for k in model_cls.model_fields}


def _apply_merged(instance: Any, model_cls: type[BaseModel], body: dict[str, Any]) -> BaseModel:
    base = _model_dump_compat(model_cls, instance)
    merged = {**base, **body}
    return model_cls.model_validate(merged)


@router.get("/config/retrieval")
async def get_retrieval_config(_user: User = Depends(get_current_user)) -> dict[str, Any]:
    return _model_dump_compat(RetrievalConfig, _settings().retrieval)


@router.put("/config/retrieval")
async def put_retrieval_config(
    body: dict[str, Any],
    user: User = Depends(require_role("update")),
) -> dict[str, Any]:
    try:
        new_cfg = _apply_merged(_settings().retrieval, RetrievalConfig, body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    r = _settings().retrieval
    for k in RetrievalConfig.model_fields:
        setattr(r, k, getattr(new_cfg, k))
    await _service().invalidate_search_cache()
    logger.info("Retrieval config updated by %s", user.name)
    return _model_dump_compat(RetrievalConfig, r)


@router.get("/config/search")
async def get_search_config(_user: User = Depends(get_current_user)) -> dict[str, Any]:
    return _model_dump_compat(SearchConfig, _settings().search)


@router.put("/config/search")
async def put_search_config(
    body: dict[str, Any],
    user: User = Depends(require_role("update")),
) -> dict[str, Any]:
    try:
        new_cfg = _apply_merged(_settings().search, SearchConfig, body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    s = _settings().search
    for k in SearchConfig.model_fields:
        setattr(s, k, getattr(new_cfg, k))
    await _service().invalidate_search_cache()
    logger.info("Search config updated by %s", user.name)
    return _model_dump_compat(SearchConfig, s)

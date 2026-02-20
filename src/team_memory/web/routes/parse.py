"""Document and URL parsing routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from team_memory.auth.provider import User
from team_memory.web import app as app_module
from team_memory.web.app import (
    ParseDocumentRequest,
    ParseURLRequest,
    SuggestTypeRequest,
    _extract_text_from_html,
    _llm_parse_content,
    get_current_user,
)

router = APIRouter(tags=["parse"])


@router.post("/experiences/parse-document")
async def parse_document(
    req: ParseDocumentRequest,
    user: User = Depends(get_current_user),
):
    """Parse a document using Ollama LLM and extract structured experience fields."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Document content is empty")

    return await _llm_parse_content(req.content)


@router.post("/experiences/parse-url")
async def parse_url(
    req: ParseURLRequest,
    user: User = Depends(get_current_user),
):
    """Fetch content from a URL and parse it into structured experience fields."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty")
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="URL must start with http:// or https://"
        )

    httpx = app_module.httpx
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "team_memory/1.0 (Experience Database Bot)",
                "Accept": "text/html,text/plain,text/markdown,application/json,*/*",
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except app_module.httpx.ConnectError:
        raise HTTPException(status_code=502, detail=f"Cannot connect to {url}")
    except app_module.httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Timeout fetching {url}")
    except app_module.httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch URL (HTTP {e.response.status_code}): {url}",
        )
    except app_module.httpx.InvalidURL:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")

    content_type = resp.headers.get("content-type", "")
    raw = resp.text

    if not raw or not raw.strip():
        raise HTTPException(status_code=502, detail="URL returned empty content")

    if "text/html" in content_type:
        text_content = _extract_text_from_html(raw)
    else:
        text_content = raw

    if not text_content.strip():
        raise HTTPException(
            status_code=502, detail="Could not extract text from the URL"
        )

    content_with_source = f"[Source URL: {url}]\n\n{text_content}"

    return await _llm_parse_content(content_with_source)


@router.post("/experiences/suggest-type")
async def suggest_experience_type_route(
    req: SuggestTypeRequest,
    user: User = Depends(get_current_user),
):
    """Suggest experience type based on title/problem using LLM."""
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title is required")

    from team_memory.services.llm_parser import suggest_experience_type

    result = await suggest_experience_type(
        title=req.title.strip(),
        problem=req.problem.strip(),
        llm_config=app_module._settings.llm if app_module._settings else None,
    )
    return result

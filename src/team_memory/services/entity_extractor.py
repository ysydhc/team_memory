"""Entity extraction service (L2.5).

Extracts named entities and relationships from an experience record,
then upserts them into the entity graph tables.

Design constraints:
- Async, non-blocking: extraction failure MUST NOT affect the Experience
  publish flow.  All errors are logged and swallowed.
- Uses LLM (glm-4-flash via LiteLLM / generic OpenAI-compat endpoint)
  with structured JSON output.
- Rule-based extraction for query path (see entity_search.py).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from team_memory.config.llm import LLMConfig
from team_memory.storage.database import get_session
from team_memory.storage.models import Entity, ExperienceEntity, Relationship

logger = logging.getLogger("team_memory.entity_extractor")

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

VALID_ENTITY_TYPES = {"tool", "concept", "person", "project", "config", "error", "service"}
VALID_RELATION_TYPES = {
    "depends_on", "configures", "causes", "fixes",
    "related_to", "uses", "part_of", "replaces",
}
VALID_ROLES = {"mentioned", "subject", "solution", "tool"}


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str = "concept"
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    role: str = "mentioned"  # role in the source experience


@dataclass
class ExtractedRelationship:
    source: str  # entity name
    target: str  # entity name
    relation_type: str = "related_to"


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are an expert knowledge-graph builder.
Given a technical experience record, extract:
1. Named entities (tools, projects, concepts, people, config items, errors).
2. Directional relationships between those entities.

Return ONLY valid JSON matching this schema:
{
  "entities": [
    {"name": string, "entity_type": string, "description": string,
     "aliases": [string], "role": string}
  ],
  "relationships": [
    {"source": string, "target": string, "relation_type": string}
  ]
}

entity_type must be one of: tool, concept, person, project, config, error, service
relation_type must be one of:
  depends_on, configures, causes, fixes, related_to, uses, part_of, replaces
role must be one of: mentioned, subject, solution, tool

Keep entity names short and canonical (e.g. "LiteLLM", "Clash", "Docker").
Extract at most 10 entities and 15 relationships. No markdown, no explanation.
"""


def _build_user_prompt(title: str, description: str, solution: str, tags: list[str]) -> str:
    parts = [f"Title: {title}", f"Problem: {description}"]
    if solution:
        parts.append(f"Solution: {solution}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n\n".join(parts)


async def _call_llm(prompt: str, llm_config: LLMConfig) -> str | None:
    """Call the LLM via OpenAI-compatible API. Returns raw response text or None on error."""
    api_key = llm_config.api_key or "none"
    base_url = llm_config.base_url.rstrip("/")
    # Ensure /v1 prefix for OpenAI-compatible endpoints
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    model = llm_config.model

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Entity extraction LLM call failed: %s", exc)
        return None


def _parse_llm_response(raw: str) -> ExtractionResult:
    """Parse JSON from LLM response, tolerating markdown code fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse entity extraction JSON: %.200s", raw)
        return ExtractionResult()

    result = ExtractionResult()

    for e in data.get("entities", [])[:10]:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        etype = e.get("entity_type", "concept")
        if etype not in VALID_ENTITY_TYPES:
            etype = "concept"
        role = e.get("role", "mentioned")
        if role not in VALID_ROLES:
            role = "mentioned"
        result.entities.append(ExtractedEntity(
            name=str(e["name"])[:255],
            entity_type=etype,
            description=str(e.get("description", ""))[:1000],
            aliases=[str(a)[:100] for a in e.get("aliases", []) if a][:5],
            role=role,
        ))

    for r in data.get("relationships", [])[:15]:
        if not isinstance(r, dict) or not r.get("source") or not r.get("target"):
            continue
        rtype = r.get("relation_type", "related_to")
        if rtype not in VALID_RELATION_TYPES:
            rtype = "related_to"
        result.relationships.append(ExtractedRelationship(
            source=str(r["source"])[:255],
            target=str(r["target"])[:255],
            relation_type=rtype,
        ))

    return result


# --------------------------------------------------------------------------- #
# DB upsert helpers
# --------------------------------------------------------------------------- #

async def _upsert_entity(
    session: Any,
    name: str,
    entity_type: str,
    description: str,
    aliases: list[str],
    project: str,
) -> uuid.UUID:
    """Upsert entity by (name, project), return its UUID."""
    stmt = (
        pg_insert(Entity)
        .values(
            id=uuid.uuid4(),
            name=name,
            entity_type=entity_type,
            description=description or None,
            aliases=aliases or None,
            source_count=1,
            project=project,
        )
        .on_conflict_do_update(
            constraint="uq_entity_name_project",
            set_={
                "source_count": text("entities.source_count + 1"),
                "entity_type": entity_type,
                "description": text(
                    "CASE WHEN excluded.description IS NOT NULL "
                    "THEN excluded.description ELSE entities.description END"
                ),
                "updated_at": text("now()"),
            },
        )
        .returning(Entity.id)
    )
    row = await session.execute(stmt)
    result = row.fetchone()
    if result:
        return result[0]
    # Fallback: fetch existing
    existing = await session.execute(
        select(Entity.id).where(Entity.name == name, Entity.project == project)
    )
    row2 = existing.fetchone()
    return row2[0] if row2 else uuid.uuid4()


async def _upsert_relationship(
    session: Any,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    relation_type: str,
    experience_id: str,
) -> None:
    """Upsert relationship and add evidence."""
    stmt = (
        pg_insert(Relationship)
        .values(
            id=uuid.uuid4(),
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=relation_type,
            weight=1.0,
            evidence=[experience_id],
        )
        .on_conflict_do_update(
            constraint="uq_relationship_src_tgt_type",
            set_={
                "weight": text("relationships.weight + 0.5"),
                "evidence": text(
                    f"(SELECT jsonb_agg(DISTINCT e) FROM jsonb_array_elements_text("
                    f"COALESCE(relationships.evidence, '[]'::jsonb) || "
                    f"'[\"{experience_id}\"]'::jsonb) AS e)"
                ),
            },
        )
    )
    await session.execute(stmt)


async def _upsert_experience_entity(
    session: Any,
    experience_id: uuid.UUID,
    entity_id: uuid.UUID,
    role: str,
) -> None:
    """Upsert experience-entity link."""
    stmt = (
        pg_insert(ExperienceEntity)
        .values(
            experience_id=experience_id,
            entity_id=entity_id,
            role=role,
        )
        .on_conflict_do_nothing()
    )
    await session.execute(stmt)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class EntityExtractor:
    """Async entity extractor — call extract_and_persist() after save()."""

    def __init__(self, llm_config: LLMConfig, db_url: str) -> None:
        self._llm_config = llm_config
        self._db_url = db_url

    async def extract_and_persist(
        self,
        experience_id: str,
        title: str,
        description: str,
        solution: str,
        tags: list[str],
        project: str,
    ) -> None:
        """Extract entities/relationships and persist them.

        Swallows all errors — this must never break the caller.
        """
        try:
            await self._run(experience_id, title, description, solution, tags, project)
        except Exception as exc:
            logger.error(
                "Entity extraction failed for experience %s: %s",
                experience_id,
                exc,
                exc_info=True,
            )

    async def _run(
        self,
        experience_id: str,
        title: str,
        description: str,
        solution: str,
        tags: list[str],
        project: str,
    ) -> None:
        prompt = _build_user_prompt(title, description, solution, tags)
        raw = await _call_llm(prompt, self._llm_config)
        if not raw:
            logger.debug("Entity extraction skipped (LLM returned nothing) for %s", experience_id)
            return

        result = _parse_llm_response(raw)
        if not result.entities:
            logger.debug("No entities extracted for experience %s", experience_id)
            return

        exp_uuid = uuid.UUID(experience_id)

        async with get_session(self._db_url) as session:
            # Map entity name -> UUID
            name_to_id: dict[str, uuid.UUID] = {}

            for ent in result.entities:
                eid = await _upsert_entity(
                    session,
                    name=ent.name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    aliases=ent.aliases,
                    project=project,
                )
                name_to_id[ent.name] = eid
                await _upsert_experience_entity(session, exp_uuid, eid, ent.role)

            for rel in result.relationships:
                src_id = name_to_id.get(rel.source)
                tgt_id = name_to_id.get(rel.target)
                if src_id and tgt_id and src_id != tgt_id:
                    await _upsert_relationship(
                        session, src_id, tgt_id, rel.relation_type, experience_id
                    )

            await session.commit()
            logger.info(
                "Entity graph updated for experience %s: %d entities, %d relationships",
                experience_id,
                len(result.entities),
                len(result.relationships),
            )

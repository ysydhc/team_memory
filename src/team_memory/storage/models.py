"""SQLAlchemy ORM models for team_memory (MVP)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Embedding vector dimension — matches the database schema.
# Updated from 768 (nomic-embed-text) to 1024 (qwen3-embedding:0.6b).
DB_VECTOR_DIM: int = 1024


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ============================================================
# Experience (core entity)
# ============================================================


class Experience(Base):
    """Team experience record — structured knowledge unit."""

    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Parent-child hierarchy: NULL = root/standalone, UUID = child of parent
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)  # = problem
    solution: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=list)

    # Auto-grouping key: experiences with the same group_key share a parent
    group_key: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)

    # Type label (free-form, no schema binding)
    experience_type: Mapped[str] = mapped_column(
        String(30), default="general", nullable=False, server_default="general"
    )

    # Vector embedding (DB_VECTOR_DIM dims, matches qwen3-embedding:0.6b)
    embedding = mapped_column(Vector(DB_VECTOR_DIM), nullable=True)

    # Full-text search (populated via trigger in DB migration)
    fts = mapped_column(TSVECTOR, nullable=True)

    # Source tracking
    source: Mapped[str] = mapped_column(String(50), default="manual")

    # Owner
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)

    # Project isolation
    project: Mapped[str] = mapped_column(
        String(100), default="default", nullable=False, server_default="default"
    )

    # Visibility & status
    visibility: Mapped[str] = mapped_column(
        String(20), default="project", nullable=False, server_default="project"
    )  # private, project, global
    exp_status: Mapped[str] = mapped_column(
        String(20), default="draft", nullable=False, server_default="draft"
    )  # draft, published, promoted

    # Quality scoring and management
    quality_score: Mapped[float] = mapped_column(
        Float, default=100.0, nullable=False, server_default="100.0"
    )
    quality_tier: Mapped[str] = mapped_column(
        String(20), default="Silver", nullable=False, server_default="Silver"
    )
    last_scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_pinned: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="false"
    )

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Recall counter (incremented when experience appears in search results)
    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    # Used counter (incremented when agent actually references the experience)
    used_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, server_default="0"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    feedbacks: Mapped[list[ExperienceFeedback]] = relationship(
        back_populates="experience", cascade="all, delete-orphan"
    )
    children: Mapped[list[Experience]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan",
        foreign_keys="Experience.parent_id",
    )
    parent: Mapped[Experience | None] = relationship(
        back_populates="children",
        remote_side=[id],
        foreign_keys="Experience.parent_id",
    )

    def to_dict(self, include_children: bool = False) -> dict:
        """Convert to a dictionary suitable for MCP tool responses."""
        d = {
            "id": str(self.id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "title": self.title,
            "description": self.description,
            "solution": self.solution,
            "tags": self.tags or [],
            "group_key": self.group_key,
            "experience_type": self.experience_type or "general",
            "source": self.source,
            "created_by": self.created_by,
            "visibility": self.visibility,
            "status": self.exp_status,
            "project": self.project,
            "quality_score": self.quality_score,
            "quality_tier": self.quality_tier,
            "last_scored_at": self.last_scored_at.isoformat() if self.last_scored_at else None,
            "is_pinned": self.is_pinned,
            "is_deleted": self.is_deleted,
            "recall_count": self.recall_count,
            "used_count": self.used_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_children and self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# ============================================================
# ExperienceFeedback
# ============================================================


class ExperienceFeedback(Base):
    """Feedback for an experience record."""

    __tablename__ = "experience_feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5, 5=best
    fitness_score: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # 1-5 post-use fitness
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_by: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    experience: Mapped[Experience] = relationship(back_populates="feedbacks")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "experience_id": str(self.experience_id),
            "rating": self.rating,
            "fitness_score": self.fitness_score,
            "comment": self.comment,
            "feedback_by": self.feedback_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# Archive system
# ============================================================


class Archive(Base):
    """Archive entry: session/plan-level summary doc and attachments."""

    __tablename__ = "archives"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    scope: Mapped[str] = mapped_column(String(20), nullable=False, server_default="session")
    scope_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)
    solution_doc: Mapped[str] = mapped_column(Text, nullable=False)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_conversation: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str] = mapped_column(String(100), nullable=False, server_default="default")
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    visibility: Mapped[str] = mapped_column(String(20), nullable=False, server_default="project")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="draft")
    content_type: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="session_archive"
    )
    value_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    embedding = mapped_column(Vector(DB_VECTOR_DIM), nullable=True)

    experience_links: Mapped[list[ArchiveExperienceLink]] = relationship(
        "ArchiveExperienceLink",
        back_populates="archive",
        cascade="all, delete-orphan",
    )
    attachments: Mapped[list[ArchiveAttachment]] = relationship(
        "ArchiveAttachment",
        back_populates="archive",
        cascade="all, delete-orphan",
    )
    tree_nodes: Mapped[list[DocumentTreeNode]] = relationship(
        "DocumentTreeNode",
        back_populates="archive",
        cascade="all, delete-orphan",
        order_by="DocumentTreeNode.node_order",
    )
    upload_failures: Mapped[list["ArchiveUploadFailure"]] = relationship(
        "ArchiveUploadFailure",
        back_populates="archive",
        cascade="all, delete-orphan",
    )


class ArchiveExperienceLink(Base):
    """Many-to-many link between Archive and Experience."""

    __tablename__ = "archive_experience_links"

    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("archives.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "archive_id",
            "experience_id",
            name="uq_archive_experience_links_archive_exp",
        ),
    )

    archive: Mapped[Archive] = relationship(back_populates="experience_links")


class ArchiveAttachment(Base):
    """Document/code snapshot or reference attached to an archive."""

    __tablename__ = "archive_attachments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("archives.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    content_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_refs: Mapped[dict | list | None] = mapped_column(JSONB, nullable=True)
    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    archive: Mapped[Archive] = relationship(back_populates="attachments")


class ArchiveUploadFailure(Base):
    """Record of a failed archive attachment upload (Web / API / agent)."""

    __tablename__ = "archive_upload_failures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("archives.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, server_default="web")
    error_code: Mapped[str] = mapped_column(String(50), nullable=False)
    error_message: Mapped[str] = mapped_column(String(500), nullable=False)
    client_filename_hint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    archive: Mapped[Archive] = relationship(back_populates="upload_failures")


# ============================================================
# DocumentTreeNode (bound to Archive for long-doc retrieval)
# ============================================================


class DocumentTreeNode(Base):
    """PageIndex-Lite tree node for long-document section retrieval."""

    __tablename__ = "document_tree_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    archive_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("archives.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    path: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    node_title: Mapped[str] = mapped_column(String(500), nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    node_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_leaf: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    archive: Mapped[Archive] = relationship(back_populates="tree_nodes")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "archive_id": str(self.archive_id),
            "path": self.path,
            "node_title": self.node_title,
            "depth": self.depth,
            "node_order": self.node_order,
            "content": self.content,
            "content_summary": self.content_summary,
            "char_count": self.char_count,
            "is_leaf": self.is_leaf,
        }


# ============================================================
# PersonalMemory
# ============================================================


class PersonalMemory(Base):
    """Per-user personal memory: preferences and habits for Agent context."""

    __tablename__ = "personal_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(
        String(20), default="generic", nullable=False, server_default="generic"
    )  # generic | context (mirrors profile_kind for HTTP compat)
    profile_kind: Mapped[str] = mapped_column(
        String(16), default="static", nullable=False, server_default="static"
    )  # static | dynamic
    context_hint: Mapped[str | None] = mapped_column(String(500), nullable=True)
    embedding = mapped_column(Vector(DB_VECTOR_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "content": self.content,
            "scope": self.scope,
            "profile_kind": self.profile_kind,
            "context_hint": self.context_hint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================
# ApiKey (authentication)
# ============================================================


class ApiKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(50), default="editor")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    key_prefix: Mapped[str | None] = mapped_column(String(4), nullable=True)
    key_suffix: Mapped[str | None] = mapped_column(String(4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ============================================================
# BackgroundTask (persistent task queue)
# ============================================================


class BackgroundTask(Base):
    """Persistent background task queue."""

    __tablename__ = "background_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3")


# ============================================================
# SearchLog (evaluation tracking)
# ============================================================


class SearchLog(Base):
    """Search query log for evaluation."""

    __tablename__ = "search_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    intent_type: Mapped[str] = mapped_column(String(20), default="unknown")
    project: Mapped[str] = mapped_column(String(100), default="default")
    source: Mapped[str] = mapped_column(String(20), default="mcp")
    result_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # [{"id": "exp-xxx", "score": 0.85, "source_layer": "L2"}, ...]
    was_used: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # None=unjudged
    judgment_source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # 'marker'|'fuzzy'|NULL
    agent_response_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ============================================================
# Entity Graph (L2.5)
# ============================================================


class Entity(Base):
    """Named entity extracted from experiences (tool, concept, person, project, etc.)."""

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="concept"
    )  # tool, concept, person, project, config, error
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    aliases: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=list)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    project: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default="default", index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # ORM relationships
    outgoing: Mapped[list["Relationship"]] = relationship(
        "Relationship",
        foreign_keys="Relationship.source_entity_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming: Mapped[list["Relationship"]] = relationship(
        "Relationship",
        foreign_keys="Relationship.target_entity_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )
    experience_links: Mapped[list["ExperienceEntity"]] = relationship(
        "ExperienceEntity",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("name", "project", name="uq_entity_name_project"),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "aliases": self.aliases or [],
            "source_count": self.source_count,
            "project": self.project,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Relationship(Base):
    """Directed relationship between two entities."""

    __tablename__ = "relationships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type: Mapped[str] = mapped_column(
        String(100), nullable=False, server_default="related_to"
    )  # depends_on, configures, causes, fixes, related_to, uses, part_of
    weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    evidence: Mapped[list | None] = mapped_column(JSONB, nullable=True, default=list)
    # evidence stores list of experience_id strings as supporting references
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    source: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="outgoing"
    )
    target: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="incoming"
    )

    __table_args__ = (
        UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_relationship_src_tgt_type",
        ),
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "source_entity_id": str(self.source_entity_id),
            "target_entity_id": str(self.target_entity_id),
            "relation_type": self.relation_type,
            "weight": self.weight,
            "evidence": self.evidence or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExperienceEntity(Base):
    """Junction table linking experiences to entities (many-to-many)."""

    __tablename__ = "experience_entities"

    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
        index=True,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="mentioned"
    )  # mentioned, subject, solution, tool

    entity: Mapped["Entity"] = relationship("Entity", back_populates="experience_links")

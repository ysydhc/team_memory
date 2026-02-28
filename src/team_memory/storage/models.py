"""SQLAlchemy ORM models for team_memory."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Experience(Base):
    """Team experience record."""

    __tablename__ = "experiences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Parent-child hierarchy: NULL = root/standalone, UUID = child of parent
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    # allow incomplete experiences (solution can be null)
    solution: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=list)
    programming_language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    framework: Mapped[str | None] = mapped_column(String(100), nullable=True)
    code_snippets: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Experience type system (v3) ---
    # Type: general / feature / bugfix / tech_design / incident / best_practice / learning
    experience_type: Mapped[str] = mapped_column(
        String(30), default="general", nullable=False, server_default="general"
    )
    # Bug/incident severity (P0-P4)
    severity: Mapped[str | None] = mapped_column(String(10), nullable=True)
    # Classification (frontend / backend / database / infra / performance / security / other)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Business progress (separate from publish_status which controls visibility)
    # bugfix: open/investigating/fixed/verified
    # feature: planning/developing/testing/released
    # tech_design: researching/reviewing/implementing/completed
    progress_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    # Type-specific structured data (JSONB) — schema varies per experience_type
    structured_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Git references — cross-type universal
    # Format: [{"type": "commit"|"pr"|"branch", "url": "...", "hash": "...", "description": "..."}]
    git_refs: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # Related links — cross-type universal
    # Format: [{"type": "issue"|"doc"|"wiki"|"other", "url": "...", "title": "..."}]
    related_links: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # LLM-generated summary for memory compaction (P0-4)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vector embedding — dimension must match the active embedding provider.
    embedding = mapped_column(Vector(768), nullable=True)

    # Embedding dimension tracking (P4-5) — allows mixed-dimension coexistence
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Full-text search (populated via trigger in DB migration)
    fts = mapped_column(TSVECTOR, nullable=True)

    # Source tracking
    source: Mapped[str] = mapped_column(String(50), default="manual")
    source_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Owner
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)

    # Scope: global (cross-project), team (project-bound), personal (creator-only)
    scope: Mapped[str] = mapped_column(
        String(20), default="team", nullable=False, server_default="team"
    )  # global, team, personal

    # Project isolation: experiences belong to a project (P4-1)
    project: Mapped[str] = mapped_column(
        String(100), default="default", nullable=False, server_default="default"
    )

    # --- New status model (v2): visibility + status ---
    # visibility: who can see when published (private/project/global)
    visibility: Mapped[str] = mapped_column(
        String(20), default="project", nullable=False, server_default="project"
    )  # private, project, global
    # Workflow status (draft/review/published/rejected)
    exp_status: Mapped[str] = mapped_column(
        String(20), default="draft", nullable=False, server_default="draft"
    )  # draft, review, published, rejected

    # --- Legacy fields (kept for backward compatibility during migration) ---
    # Review / publish workflow
    publish_status: Mapped[str] = mapped_column(
        String(20), default="personal", nullable=False
    )  # personal, draft, pending_team, published, rejected
    review_status: Mapped[str] = mapped_column(
        String(20), default="approved", nullable=False
    )  # pending, approved, rejected
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Embedding status (D2: async embedding queue)
    embedding_status: Mapped[str] = mapped_column(
        String(20), default="ready", nullable=False, server_default="ready"
    )  # ready, pending, failed

    # Quality scoring system (0-300 scale)
    quality_score: Mapped[int] = mapped_column(
        Integer, default=100, server_default="100", nullable=False
    )
    pinned: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    last_decay_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_rating: Mapped[float] = mapped_column(Float, default=0.0)
    use_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=_utcnow
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
    versions: Mapped[list[ExperienceVersion]] = relationship(
        back_populates="experience",
        cascade="all, delete-orphan",
        order_by="ExperienceVersion.version_number.desc()",
    )
    tree_nodes: Mapped[list[DocumentTreeNode]] = relationship(
        back_populates="experience",
        cascade="all, delete-orphan",
        order_by="DocumentTreeNode.node_order",
    )

    @property
    def completeness_score(self) -> int:
        """Calculate 0-100 completeness score for this experience."""
        from team_memory.schemas import compute_completeness_score
        return compute_completeness_score(
            title=self.title,
            description=self.description,
            solution=self.solution,
            root_cause=self.root_cause,
            code_snippets=self.code_snippets,
            tags=self.tags,
            git_refs=self.git_refs,
            related_links=self.related_links,
            structured_data=self.structured_data,
            experience_type=self.experience_type or "general",
            avg_rating=self.avg_rating or 0.0,
        )

    def _quality_tier(self) -> str:
        s = self.quality_score or 0
        if s >= 120:
            return "gold"
        if s >= 60:
            return "silver"
        if s >= 20:
            return "bronze"
        return "outdated"

    def to_dict(self, include_children: bool = False) -> dict:
        """Convert to a dictionary suitable for MCP tool responses."""
        d = {
            "id": str(self.id),
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "title": self.title,
            "description": self.description,
            "root_cause": self.root_cause,
            "solution": self.solution,
            "tags": self.tags or [],
            "programming_language": self.programming_language,
            "framework": self.framework,
            "code_snippets": self.code_snippets,
            "summary": self.summary,
            # Experience type system
            "experience_type": self.experience_type or "general",
            "severity": self.severity,
            "category": self.category,
            "progress_status": self.progress_status,
            "structured_data": self.structured_data,
            "git_refs": self.git_refs,
            "related_links": self.related_links,
            "completeness_score": self.completeness_score,
            "quality_score": self.quality_score,
            "pinned": self.pinned,
            "quality_tier": self._quality_tier(),
            # Metadata
            "source": self.source,
            "created_by": self.created_by,
            "visibility": self.visibility,
            "status": self.exp_status,
            "scope": self.scope,
            "project": self.project,
            "publish_status": self.publish_status,
            "review_status": self.review_status,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "is_deleted": self.is_deleted,
            "embedding_status": self.embedding_status,
            "view_count": self.view_count,
            "avg_rating": self.avg_rating,
            "use_count": self.use_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }
        if include_children and self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


class ExperienceArtifact(Base):
    """Verbatim artifact extracted from conversations/documents (P2.B).

    Stores direct quotes with context, avoiding summarization loss.
    """

    __tablename__ = "experience_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # decision, problem, pattern, constraint, fact
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_before: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_after: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "experience_id": str(self.experience_id),
            "artifact_type": self.artifact_type,
            "content": self.content,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "source_ref": self.source_ref,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExperienceReflection(Base):
    """Post-task reflection record (P2.C Reflexion loop).

    Stores what worked, what failed, and generalized strategies
    from completed tasks.
    """

    __tablename__ = "experience_reflections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personal_tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    experience_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    success_points: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_points: Mapped[str | None] = mapped_column(Text, nullable=True)
    improvements: Mapped[str | None] = mapped_column(Text, nullable=True)
    generalized_strategy: Mapped[str | None] = mapped_column(Text, nullable=True)
    judge_score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "task_id": str(self.task_id) if self.task_id else None,
            "experience_id": str(self.experience_id) if self.experience_id else None,
            "success_points": self.success_points,
            "failure_points": self.failure_points,
            "improvements": self.improvements,
            "generalized_strategy": self.generalized_strategy,
            "judge_score": self.judge_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    experience: Mapped[Experience] = relationship(back_populates="feedbacks")

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        return {
            "id": self.id,
            "experience_id": str(self.experience_id),
            "rating": self.rating,
            "fitness_score": self.fitness_score,
            "comment": self.comment,
            "feedback_by": self.feedback_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TaskGroup(Base):
    """A group of related tasks, often derived from a .debug document."""

    __tablename__ = "task_groups"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str] = mapped_column(
        String(100), default="default", nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    source_doc: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    archived: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    tasks: Mapped[list[PersonalTask]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )

    def to_dict(self, include_tasks: bool = False) -> dict:
        d = {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "project": self.project,
            "user_id": self.user_id,
            "source_doc": self.source_doc,
            "sort_order": self.sort_order,
            "archived": self.archived,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_tasks and self.tasks:
            d["tasks"] = [t.to_dict() for t in self.tasks]
            total = len(self.tasks)
            done = sum(1 for t in self.tasks if t.status == "completed")
            d["progress"] = {"total": total, "completed": done}
        return d


class PersonalTask(Base):
    """A personal task card, optionally belonging to a TaskGroup."""

    __tablename__ = "personal_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    group_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("task_groups.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="wait", nullable=False
    )  # wait, plan, in_progress, completed, cancelled
    priority: Mapped[str] = mapped_column(
        String(20), default="medium", nullable=False
    )  # low, medium, high, urgent
    importance: Mapped[int] = mapped_column(Integer, default=3)  # 1-5
    project: Mapped[str] = mapped_column(
        String(100), default="default", nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    labels: Mapped[list[str] | None] = mapped_column(ARRAY(String), default=list)
    experience_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="SET NULL"),
        nullable=True,
    )
    sediment_experience_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Phase 2: Atomic Claim
    assignee: Mapped[str | None] = mapped_column(String(100), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Phase 4: Hierarchy
    parent_task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personal_tasks.id", ondelete="SET NULL"),
        nullable=True,
    )
    path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    display_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    group: Mapped[TaskGroup | None] = relationship(back_populates="tasks")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "group_id": str(self.group_id) if self.group_id else None,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "importance": self.importance,
            "project": self.project,
            "user_id": self.user_id,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "sort_order": self.sort_order,
            "labels": self.labels or [],
            "experience_id": str(self.experience_id) if self.experience_id else None,
            "sediment_experience_id": (
                str(self.sediment_experience_id) if self.sediment_experience_id else None
            ),
            "assignee": self.assignee,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "parent_task_id": str(self.parent_task_id) if self.parent_task_id else None,
            "path": self.path,
            "display_id": self.display_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TaskDependency(Base):
    """Dependency relationship between tasks."""

    __tablename__ = "task_dependencies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personal_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personal_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dep_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # blocks, related, discovered_from
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "source_task_id": str(self.source_task_id),
            "target_task_id": str(self.target_task_id),
            "dep_type": self.dep_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


class TaskMessage(Base):
    """Message/comment on a task, supporting threaded discussions."""

    __tablename__ = "task_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personal_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    thread_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    author: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "thread_id": str(self.thread_id) if self.thread_id else None,
            "author": self.author,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentTreeNode(Base):
    """PageIndex-Lite tree node for long-document section retrieval."""

    __tablename__ = "document_tree_nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    experience: Mapped[Experience] = relationship(back_populates="tree_nodes")

    def to_dict(self) -> dict:
        """Convert node to a JSON-safe dict."""
        return {
            "id": str(self.id),
            "experience_id": str(self.experience_id),
            "path": self.path,
            "node_title": self.node_title,
            "depth": self.depth,
            "node_order": self.node_order,
            "content": self.content,
            "content_summary": self.content_summary,
            "char_count": self.char_count,
            "is_leaf": self.is_leaf,
        }


class ApiKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(50), default="editor")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    password_hash: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class ExperienceVersion(Base):
    """Version history snapshot for an experience record.

    Each time an experience is edited (hard_delete_and_rebuild) or merged,
    a version snapshot is saved here before the destructive operation.
    """

    __tablename__ = "experience_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experience_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    changed_by: Mapped[str] = mapped_column(String(100), nullable=False)
    change_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    experience: Mapped[Experience | None] = relationship(
        back_populates="versions",
    )

    def to_dict(self) -> dict:
        """Convert to a dictionary."""
        return {
            "id": str(self.id),
            "experience_id": str(self.experience_id) if self.experience_id else None,
            "version_number": self.version_number,
            "snapshot": self.snapshot,
            "changed_by": self.changed_by,
            "change_summary": self.change_summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExperienceLink(Base):
    """Relationship link between two experiences."""

    __tablename__ = "experience_links"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
    )
    # related | supersedes | derived_from
    link_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "target_id": str(self.target_id),
            "link_type": self.link_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }


class ReviewHistory(Base):
    """History of review actions on experiences."""

    __tablename__ = "review_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experience_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiences.id", ondelete="CASCADE"),
        nullable=False,
    )
    reviewer: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # approved, rejected
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "experience_id": str(self.experience_id),
            "reviewer": self.reviewer,
            "action": self.action,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    """Audit log for sensitive operations."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_name": self.user_name,
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "detail": self.detail,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class QueryLog(Base):
    """Log of search queries for analytics."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source: Mapped[str] = mapped_column(String(50), default="mcp")  # mcp, web
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    search_type: Mapped[str] = mapped_column(String(20), default="vector")  # vector, fts
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class ToolUsageLog(Base):
    """Log of MCP tool and skill usage for analytics."""

    __tablename__ = "tool_usage_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tool_type: Mapped[str] = mapped_column(
        String(20), default="mcp", nullable=False
    )  # mcp, skill
    user: Mapped[str] = mapped_column(String(100), default="anonymous", nullable=False)
    project: Mapped[str] = mapped_column(
        String(100), default="default", nullable=False
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "user": self.user,
            "project": self.project,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "metadata": self.metadata_extra,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

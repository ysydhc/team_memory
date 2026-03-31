"""Initial MVP schema.

Revision ID: 001_initial_mvp
Revises:
Create Date: 2026-03-29

Tables:
  - experiences (core entity with parent-child hierarchy, group_key, embedding, FTS)
  - experience_feedbacks
  - archives + archive_experience_links + archive_attachments
  - document_tree_nodes (bound to archives)
  - personal_memories
  - api_keys
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB, TSVECTOR

# revision identifiers, used by Alembic.
revision = "001_initial_mvp"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # -- experiences --
    op.create_table(
        "experiences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("solution", sa.Text, nullable=True),
        sa.Column("tags", ARRAY(sa.String), nullable=True),
        sa.Column("group_key", sa.String(200), nullable=True, index=True),
        sa.Column("experience_type", sa.String(30), nullable=False, server_default="general"),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("fts", TSVECTOR, nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column("visibility", sa.String(20), nullable=False, server_default="project"),
        sa.Column("exp_status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("use_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    # GIN index for FTS
    op.create_index("ix_experiences_fts", "experiences", ["fts"], postgresql_using="gin")
    # Index for project + status queries
    op.create_index("ix_experiences_project_status", "experiences", ["project", "exp_status"])

    # -- experience_feedbacks --
    op.create_table(
        "experience_feedbacks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("experience_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("fitness_score", sa.Integer, nullable=True),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("feedback_by", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_experience_feedbacks_experience_id", "experience_feedbacks", ["experience_id"])

    # -- archives --
    op.create_table(
        "archives",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("scope", sa.String(20), nullable=False, server_default="session"),
        sa.Column("scope_ref", sa.String(200), nullable=True),
        sa.Column("solution_doc", sa.Text, nullable=False),
        sa.Column("overview", sa.Text, nullable=True),
        sa.Column("conversation_summary", sa.Text, nullable=True),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column("visibility", sa.String(20), nullable=False, server_default="project"),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("embedding", Vector(768), nullable=True),
    )

    # -- archive_experience_links --
    op.create_table(
        "archive_experience_links",
        sa.Column("archive_id", UUID(as_uuid=True), sa.ForeignKey("archives.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("experience_id", UUID(as_uuid=True), sa.ForeignKey("experiences.id", ondelete="CASCADE"), nullable=False, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("archive_id", "experience_id", name="uq_archive_experience_links_archive_exp"),
    )

    # -- archive_attachments --
    op.create_table(
        "archive_attachments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("archive_id", UUID(as_uuid=True), sa.ForeignKey("archives.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("path", sa.String(1000), nullable=True),
        sa.Column("content_snapshot", sa.Text, nullable=True),
        sa.Column("git_commit", sa.String(64), nullable=True),
        sa.Column("git_refs", JSONB, nullable=True),
        sa.Column("snippet", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -- document_tree_nodes (bound to archives) --
    op.create_table(
        "document_tree_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("archive_id", UUID(as_uuid=True), sa.ForeignKey("archives.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("path", sa.String(100), nullable=False, index=True),
        sa.Column("node_title", sa.String(500), nullable=False),
        sa.Column("depth", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("node_order", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("content_summary", sa.Text, nullable=True),
        sa.Column("char_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("is_leaf", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # -- personal_memories --
    op.create_table(
        "personal_memories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(100), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("scope", sa.String(20), nullable=False, server_default="generic"),
        sa.Column("context_hint", sa.String(500), nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # -- api_keys --
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("key_hash", sa.String(256), unique=True, nullable=True),
        sa.Column("user_name", sa.String(100), nullable=False, unique=True),
        sa.Column("role", sa.String(50), nullable=False, server_default="editor"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("password_hash", sa.String(256), nullable=True),
        sa.Column("key_prefix", sa.String(4), nullable=True),
        sa.Column("key_suffix", sa.String(4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # FTS trigger: auto-update fts column on insert/update
    # Weights: A=title (highest), B=description+tags, C=solution
    op.execute("""
        CREATE OR REPLACE FUNCTION experiences_fts_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.fts :=
                setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
                setweight(to_tsvector('simple', coalesce(NEW.description, '') || ' ' || coalesce(array_to_string(NEW.tags, ' '), '')), 'B') ||
                setweight(to_tsvector('simple', coalesce(NEW.solution, '')), 'C');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiences_fts
        BEFORE INSERT OR UPDATE OF title, description, solution, tags
        ON experiences
        FOR EACH ROW EXECUTE FUNCTION experiences_fts_trigger();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_experiences_fts ON experiences")
    op.execute("DROP FUNCTION IF EXISTS experiences_fts_trigger()")
    op.drop_table("api_keys")
    op.drop_table("personal_memories")
    op.drop_table("document_tree_nodes")
    op.drop_table("archive_attachments")
    op.drop_table("archive_experience_links")
    op.drop_table("archives")
    op.drop_table("experience_feedbacks")
    op.drop_table("experiences")

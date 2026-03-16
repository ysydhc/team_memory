"""add_archives_tables

Revision ID: c2d3e4f5a6b7
Revises: 6ab06751f40e
Create Date: 2026-03-16

Creates archives, archive_experience_links, archive_attachments with pgvector.
"""
from typing import Sequence, Union

import pgvector.sqlalchemy.vector
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c2d3e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "6ab06751f40e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. archives（含 overview、vector(768)）
    op.create_table(
        "archives",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("scope", sa.String(20), nullable=False, server_default="session"),
        sa.Column("scope_ref", sa.String(200), nullable=True),
        sa.Column("solution_doc", sa.Text(), nullable=False),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("conversation_summary", sa.Text(), nullable=True),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column("created_by", sa.String(100), nullable=False),
        sa.Column("visibility", sa.String(20), nullable=False, server_default="project"),
        sa.Column("status", sa.String(20), nullable=False, server_default="draft"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.vector.VECTOR(dim=768),
            nullable=True,
        ),
    )
    op.create_index("ix_archives_project", "archives", ["project"], unique=False)
    op.create_index("ix_archives_status", "archives", ["status"], unique=False)
    op.create_index("ix_archives_created_at", "archives", ["created_at"], unique=False)
    op.execute("""
        CREATE INDEX idx_archives_embedding ON archives
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)

    # 2. archive_experience_links（唯一约束 (archive_id, experience_id)）
    op.create_table(
        "archive_experience_links",
        sa.Column("archive_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("experience_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["archive_id"],
            ["archives.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "archive_id",
            "experience_id",
            name="uq_archive_experience_links_archive_exp",
        ),
    )
    op.create_index(
        "ix_archive_experience_links_archive_id",
        "archive_experience_links",
        ["archive_id"],
        unique=False,
    )
    op.create_index(
        "ix_archive_experience_links_experience_id",
        "archive_experience_links",
        ["experience_id"],
        unique=False,
    )

    # 3. archive_attachments
    op.create_table(
        "archive_attachments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("archive_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("path", sa.String(1000), nullable=True),
        sa.Column("content_snapshot", sa.Text(), nullable=True),
        sa.Column("git_commit", sa.String(64), nullable=True),
        sa.Column("git_refs", postgresql.JSONB(), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["archive_id"],
            ["archives.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_archive_attachments_archive_id",
        "archive_attachments",
        ["archive_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop archive tables and indexes. WARNING: Downgrade deletes all archive data."""
    op.drop_index(
        "ix_archive_attachments_archive_id",
        table_name="archive_attachments",
    )
    op.drop_table("archive_attachments")

    op.drop_index(
        "ix_archive_experience_links_experience_id",
        table_name="archive_experience_links",
    )
    op.drop_index(
        "ix_archive_experience_links_archive_id",
        table_name="archive_experience_links",
    )
    op.drop_table("archive_experience_links")

    op.execute("DROP INDEX IF EXISTS idx_archives_embedding")
    op.drop_index("ix_archives_created_at", table_name="archives")
    op.drop_index("ix_archives_status", table_name="archives")
    op.drop_index("ix_archives_project", table_name="archives")
    op.drop_table("archives")

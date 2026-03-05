"""Add PageIndex-Lite tree nodes table.

Revision ID: h4i5j6k7l8m
Revises: g3h4i5j6k7l8
Create Date: 2026-02-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "h4i5j6k7l8m"
down_revision = "g3h4i5j6k7l8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_tree_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "experience_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.String(length=100), nullable=False),
        sa.Column("node_title", sa.String(length=500), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("node_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_summary", sa.Text(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_leaf", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_tree_nodes_experience_id",
        "document_tree_nodes",
        ["experience_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_tree_nodes_path",
        "document_tree_nodes",
        ["path"],
        unique=False,
    )
    op.create_index(
        "ix_document_tree_nodes_depth",
        "document_tree_nodes",
        ["depth"],
        unique=False,
    )
    op.create_index(
        "ix_document_tree_nodes_order",
        "document_tree_nodes",
        ["experience_id", "node_order"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX ix_document_tree_nodes_fts "
        "ON document_tree_nodes USING gin "
        "(to_tsvector('simple', coalesce(node_title, '') || ' ' || coalesce(content, '')))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_tree_nodes_fts")
    op.drop_index("ix_document_tree_nodes_order", table_name="document_tree_nodes")
    op.drop_index("ix_document_tree_nodes_depth", table_name="document_tree_nodes")
    op.drop_index("ix_document_tree_nodes_path", table_name="document_tree_nodes")
    op.drop_index("ix_document_tree_nodes_experience_id", table_name="document_tree_nodes")
    op.drop_table("document_tree_nodes")

"""Add embedding_status column to experiences table.

Supports async embedding queue (D2): new experiences can be saved
with embedding_status='pending', then updated to 'ready' when the
background worker finishes generating the embedding.

Revision ID: a7b8c9d0e1f2
Revises: f5a6b7c8d9e0
Create Date: 2026-02-13
"""

import sqlalchemy as sa
from alembic import op

revision = "a7b8c9d0e1f2"
down_revision = "f5a6b7c8d9e0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add embedding_status column with default 'ready'
    # (all existing experiences already have embeddings)
    op.add_column(
        "experiences",
        sa.Column(
            "embedding_status",
            sa.String(20),
            nullable=False,
            server_default="ready",
        ),
    )
    op.create_index(
        "ix_experiences_embedding_status",
        "experiences",
        ["embedding_status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_experiences_embedding_status", table_name="experiences")
    op.drop_column("experiences", "embedding_status")

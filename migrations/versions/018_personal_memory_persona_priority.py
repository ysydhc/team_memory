"""Add persona and priority columns to personal_memories.

- persona VARCHAR(20) DEFAULT 'general' — one of: coding, writing, research, debugging, general
- priority INTEGER DEFAULT 50 — 0-100, higher means load first

Revision ID: 018_persona_priority
Revises: 017_embedding_dim_1024
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018_persona_priority"
down_revision = "017_embedding_dim_1024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "personal_memories",
        sa.Column(
            "persona",
            sa.String(20),
            nullable=False,
            server_default="general",
        ),
    )
    op.add_column(
        "personal_memories",
        sa.Column(
            "priority",
            sa.Integer(),
            nullable=False,
            server_default="50",
        ),
    )
    # Add check constraint for valid persona values
    op.create_check_constraint(
        "ck_personal_memories_persona",
        "personal_memories",
        sa.text(
            "(persona)::text = ANY (ARRAY['coding'::text, 'writing'::text, "
            "'research'::text, 'debugging'::text, 'general'::text])"
        ),
    )
    # Add check constraint for priority range
    op.create_check_constraint(
        "ck_personal_memories_priority_range",
        "personal_memories",
        sa.text("priority >= 0 AND priority <= 100"),
    )
    # Index for efficient count/lookup by (user_id, persona)
    op.create_index(
        "ix_personal_memories_user_persona",
        "personal_memories",
        ["user_id", "persona"],
    )


def downgrade() -> None:
    op.drop_index("ix_personal_memories_user_persona", table_name="personal_memories")
    op.drop_constraint(
        "ck_personal_memories_priority_range",
        "personal_memories",
        type_="check",
    )
    op.drop_constraint(
        "ck_personal_memories_persona",
        "personal_memories",
        type_="check",
    )
    op.drop_column("personal_memories", "priority")
    op.drop_column("personal_memories", "persona")

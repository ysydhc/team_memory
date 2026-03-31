"""Add personal_memories.profile_kind (static vs dynamic).

Revision ID: 004_profile_kind
Revises: 003_personal_memories_if_missing
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_profile_kind"
down_revision = "003_personal_memories_if_missing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "personal_memories",
        sa.Column(
            "profile_kind",
            sa.String(16),
            nullable=False,
            server_default="static",
        ),
    )
    op.execute(
        "UPDATE personal_memories SET profile_kind = 'dynamic' WHERE scope = 'context'"
    )
    op.create_check_constraint(
        "ck_personal_memories_profile_kind",
        "personal_memories",
        sa.text("(profile_kind)::text = ANY (ARRAY['static'::text, 'dynamic'::text])"),
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_personal_memories_profile_kind",
        "personal_memories",
        type_="check",
    )
    op.drop_column("personal_memories", "profile_kind")

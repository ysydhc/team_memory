"""Add project column and scope/project composite index to personal_memories.

Revision ID: 024_personal_memory_scope_project
Revises: 023_multitenant_rls

Keeps the strict single-tenant PostgreSQL RLS policy installed in 023_multitenant_rls
(tenant_id = current_tenant_id) intact with zero cross-tenant leakage.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "024_personal_memory_scope_project"
down_revision = "023_multitenant_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "personal_memories",
        sa.Column("project", sa.String(100), nullable=True),
    )
    op.alter_column(
        "personal_memories",
        "scope",
        server_default="global",
    )
    op.create_index(
        "ix_personal_memories_tenant_user_scope_project",
        "personal_memories",
        ["tenant_id", "user_id", "scope", "project"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_personal_memories_tenant_user_scope_project",
        table_name="personal_memories",
    )
    op.alter_column(
        "personal_memories",
        "scope",
        server_default="generic",
    )
    op.drop_column("personal_memories", "project")

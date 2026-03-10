"""Add experience_architecture_bindings table for experience-node binding.

Revision ID: v9w0x1y2z3
Revises: u8v9w0x1y2
Create Date: 2026-03-10

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "v9w0x1y2z3"
down_revision: Union[str, Sequence[str], None] = "u8v9w0x1y2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experience_architecture_bindings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experience_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_key", sa.String(500), nullable=False),
        sa.Column("project", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["experience_id"],
            ["experiences.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "experience_id",
            "node_key",
            name="uq_exp_arch_binding_exp_node",
        ),
    )
    op.create_index(
        "ix_experience_architecture_bindings_experience_id",
        "experience_architecture_bindings",
        ["experience_id"],
        unique=False,
    )
    op.create_index(
        "ix_experience_architecture_bindings_node_key",
        "experience_architecture_bindings",
        ["node_key"],
        unique=False,
    )
    op.create_index(
        "ix_experience_architecture_bindings_project",
        "experience_architecture_bindings",
        ["project"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_experience_architecture_bindings_project",
        table_name="experience_architecture_bindings",
    )
    op.drop_index(
        "ix_experience_architecture_bindings_node_key",
        table_name="experience_architecture_bindings",
    )
    op.drop_index(
        "ix_experience_architecture_bindings_experience_id",
        table_name="experience_architecture_bindings",
    )
    op.drop_table("experience_architecture_bindings")

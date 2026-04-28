"""Add entity graph tables (L2.5): entities, relationships, experience_entities.

Revision ID: 014_add_entity_graph
Revises: 013_add_search_logs
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "014_add_entity_graph"
down_revision = "013_add_search_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # entities
    # ------------------------------------------------------------------ #
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False, server_default="concept"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("aliases", JSONB, nullable=True),
        sa.Column("source_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("project", sa.String(100), nullable=False, server_default="default"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("name", "project", name="uq_entity_name_project"),
    )
    op.create_index("ix_entities_name", "entities", ["name"])
    op.create_index("ix_entities_project", "entities", ["project"])

    # ------------------------------------------------------------------ #
    # relationships
    # ------------------------------------------------------------------ #
    op.create_table(
        "relationships",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_entity_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_entity_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "relation_type", sa.String(100), nullable=False, server_default="related_to"
        ),
        sa.Column("weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("evidence", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_relationship_src_tgt_type",
        ),
    )
    op.create_index("ix_relationships_source", "relationships", ["source_entity_id"])
    op.create_index("ix_relationships_target", "relationships", ["target_entity_id"])

    # ------------------------------------------------------------------ #
    # experience_entities  (junction)
    # ------------------------------------------------------------------ #
    op.create_table(
        "experience_entities",
        sa.Column(
            "experience_id",
            UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("role", sa.String(50), nullable=False, server_default="mentioned"),
    )
    op.create_index(
        "ix_experience_entities_exp", "experience_entities", ["experience_id"]
    )
    op.create_index(
        "ix_experience_entities_ent", "experience_entities", ["entity_id"]
    )


def downgrade() -> None:
    op.drop_table("experience_entities")
    op.drop_table("relationships")
    op.drop_table("entities")

"""Add epistemic_tier, evidence_anchors, and staleness_status to experiences.

Revision ID: 026_memory_accuracy_defenses
Revises: 025_pm_scope_project

Supports the 4-defense memory accuracy architecture:
  - Defense 1 (Write-Time): epistemic_tier (verified / tool_grounded / auto_extract)
  - Defense 2 (Code-Grounded): evidence_anchors (JSONB) & staleness_status (fresh / possibly_stale / stale)
  - Defense 3 (Offline): composite indexes scoped by tenant_id for Janitor / AutoDream
  - Defense 4 (Read-Time): metadata exposure & CRAG adjustment

Keeps the strict single-tenant PostgreSQL RLS policy installed in 023_multitenant_rls
(tenant_id = current_tenant_id) intact with zero cross-tenant leakage.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "026_memory_accuracy_defenses"
down_revision = "025_pm_scope_project"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "experiences",
        sa.Column(
            "epistemic_tier",
            sa.String(20),
            nullable=False,
            server_default="auto_extract",
        ),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "evidence_anchors",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "staleness_status",
            sa.String(20),
            nullable=False,
            server_default="fresh",
        ),
    )

    # Backfill conservative verified tier: only pinned or manual records (avoid promoting legacy Gold blindly).
    op.execute(
        sa.text(
            """
            UPDATE experiences
            SET epistemic_tier = 'verified'
            WHERE is_pinned = true OR source = 'manual'
            """
        )
    )

    op.create_index(
        "ix_experiences_tenant_staleness",
        "experiences",
        ["tenant_id", "staleness_status"],
    )
    op.create_index(
        "ix_experiences_tenant_epistemic",
        "experiences",
        ["tenant_id", "epistemic_tier"],
    )


def downgrade() -> None:
    op.drop_index("ix_experiences_tenant_epistemic", table_name="experiences")
    op.drop_index("ix_experiences_tenant_staleness", table_name="experiences")
    op.drop_column("experiences", "staleness_status")
    op.drop_column("experiences", "evidence_anchors")
    op.drop_column("experiences", "epistemic_tier")

"""Add multi-tenant llm_usage_logs table with PostgreSQL RLS.

Revision ID: 029_llm_usage_logs
Revises: 028_binary_quantize_hnsw
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "029_llm_usage_logs"
down_revision = "028_binary_quantize_hnsw"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-4000-8000-000000000001"
APP_ROLE = "tm_app_user"
TABLE_NAME = "llm_usage_logs"


def upgrade() -> None:
    op.create_table(
        TABLE_NAME,
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            server_default=sa.text(f"'{DEFAULT_TENANT_ID}'::uuid"),
        ),
        sa.Column("caller", sa.String(100), nullable=False, server_default="unknown"),
        sa.Column("provider", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("model", sa.String(128), nullable=False, server_default="unknown"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_index("ix_llm_usage_logs_tenant_created", TABLE_NAME, ["tenant_id", "created_at"])
    op.create_index("ix_llm_usage_logs_caller_created", TABLE_NAME, ["caller", "created_at"])
    op.create_index("ix_llm_usage_logs_created_at", TABLE_NAME, ["created_at"])

    # Grant table privileges to non-superuser RLS app role if present
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {TABLE_NAME} TO {APP_ROLE};
                END IF;
            END
            $$;
            """
        )
    )

    # Enable & Force Row-Level Security with strict tenant isolation policy
    op.execute(sa.text(f"ALTER TABLE {TABLE_NAME} ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text(f"ALTER TABLE {TABLE_NAME} FORCE ROW LEVEL SECURITY"))
    op.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation_{TABLE_NAME} ON {TABLE_NAME}"))
    op.execute(
        sa.text(
            f"""
            CREATE POLICY tenant_isolation_{TABLE_NAME} ON {TABLE_NAME}
            USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation_{TABLE_NAME} ON {TABLE_NAME}"))
    op.execute(sa.text(f"ALTER TABLE {TABLE_NAME} NO FORCE ROW LEVEL SECURITY"))
    op.execute(sa.text(f"ALTER TABLE {TABLE_NAME} DISABLE ROW LEVEL SECURITY"))

    op.drop_index("ix_llm_usage_logs_created_at", table_name=TABLE_NAME)
    op.drop_index("ix_llm_usage_logs_caller_created", table_name=TABLE_NAME)
    op.drop_index("ix_llm_usage_logs_tenant_created", table_name=TABLE_NAME)
    op.drop_table(TABLE_NAME)

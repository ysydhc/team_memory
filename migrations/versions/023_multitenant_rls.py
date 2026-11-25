"""Add multi-tenant schema (tenants, tenant_members, tenant_id) and PostgreSQL RLS.

Revision ID: 023_multitenant_rls
Revises: 022_telemetry_and_bitemporal
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "023_multitenant_rls"
down_revision = "022_telemetry_and_bitemporal"
branch_labels = None
depends_on = None

DEFAULT_TENANT_ID = "00000000-0000-4000-8000-000000000001"
APP_ROLE = "tm_app_user"

# All 13 tables that receive tenant_id NOT NULL
ALL_TENANT_TABLES = (
    "experiences",
    "experience_feedbacks",
    "archives",
    "archive_experience_links",
    "archive_attachments",
    "archive_upload_failures",
    "document_tree_nodes",
    "personal_memories",
    "api_keys",
    "search_logs",
    "entities",
    "relationships",
    "experience_entities",
)

# 12 business tables protected by PostgreSQL Row-Level Security
# (api_keys is queried during pre-tenant authentication and therefore not RLS-gated)
RLS_TABLES = (
    "experiences",
    "experience_feedbacks",
    "archives",
    "archive_experience_links",
    "archive_attachments",
    "archive_upload_failures",
    "document_tree_nodes",
    "personal_memories",
    "search_logs",
    "entities",
    "relationships",
    "experience_entities",
)


def upgrade() -> None:
    # 1. Create tenants & tenant_members tables
    op.create_table(
        "tenants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "settings_json",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_table(
        "tenant_members",
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
        ),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("tenant_id", "user_id", name="uq_tenant_members_tenant_user"),
    )
    op.create_index("ix_tenant_members_user_id", "tenant_members", ["user_id"])
    op.create_index("ix_tenant_members_tenant_id", "tenant_members", ["tenant_id"])

    # 2. Seed deterministic default tenant
    op.execute(
        sa.text(
            f"""
            INSERT INTO tenants (id, slug, name, settings_json)
            VALUES ('{DEFAULT_TENANT_ID}'::uuid, 'default', 'Default Workspace', '{{}}'::jsonb)
            ON CONFLICT (slug) DO NOTHING
            """
        )
    )

    # 3. Add tenant_id to all 13 tables, backfill existing rows to default tenant, set NOT NULL + FK
    for tbl in ALL_TENANT_TABLES:
        op.add_column(
            tbl,
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        op.execute(
            sa.text(
                f"UPDATE {tbl} SET tenant_id = '{DEFAULT_TENANT_ID}'::uuid WHERE tenant_id IS NULL"
            )
        )
        op.alter_column(
            tbl,
            "tenant_id",
            nullable=False,
            server_default=sa.text(f"'{DEFAULT_TENANT_ID}'::uuid"),
        )
        op.create_foreign_key(
            f"fk_{tbl}_tenant_id",
            tbl,
            "tenants",
            ["tenant_id"],
            ["id"],
            ondelete="CASCADE",
        )

    # 4. Composite indexes & constraint adjustments
    op.create_index(
        "ix_experiences_tenant_project", "experiences", ["tenant_id", "project", "is_deleted"]
    )
    op.create_index("ix_experiences_tenant_status", "experiences", ["tenant_id", "exp_status"])
    op.create_index(
        "ix_exp_feedbacks_tenant_exp", "experience_feedbacks", ["tenant_id", "experience_id"]
    )
    op.create_index("ix_archives_tenant_project", "archives", ["tenant_id", "project"])
    op.create_index(
        "ix_archive_exp_links_tenant", "archive_experience_links", ["tenant_id", "archive_id"]
    )
    op.create_index(
        "ix_archive_attachments_tenant", "archive_attachments", ["tenant_id", "archive_id"]
    )
    op.create_index(
        "ix_archive_upload_failures_tenant",
        "archive_upload_failures",
        ["tenant_id", "archive_id"],
    )
    op.create_index(
        "ix_doc_tree_nodes_tenant", "document_tree_nodes", ["tenant_id", "archive_id"]
    )
    op.create_index(
        "ix_personal_memories_tenant_user", "personal_memories", ["tenant_id", "user_id"]
    )
    op.create_index("ix_search_logs_tenant_created", "search_logs", ["tenant_id", "created_at"])
    op.create_index(
        "ix_relationships_tenant_src", "relationships", ["tenant_id", "source_entity_id"]
    )
    op.create_index(
        "ix_exp_entities_tenant", "experience_entities", ["tenant_id", "entity_id"]
    )

    # entities: replace (name, project) unique constraint with (tenant_id, name, project)
    op.execute(sa.text("ALTER TABLE entities DROP CONSTRAINT IF EXISTS uq_entity_name_project"))
    op.create_unique_constraint(
        "uq_entity_tenant_name_project", "entities", ["tenant_id", "name", "project"]
    )
    op.create_index("ix_entities_tenant_project", "entities", ["tenant_id", "project"])

    # api_keys: drop global UNIQUE(user_name) to allow per-tenant keys per user, add label
    op.add_column("api_keys", sa.Column("label", sa.String(100), nullable=True))
    op.execute(sa.text("ALTER TABLE api_keys DROP CONSTRAINT IF EXISTS api_keys_user_name_key"))
    op.create_index("ix_api_keys_tenant_user", "api_keys", ["tenant_id", "user_name"])

    # 5. Backfill tenant_members from existing api_keys
    op.execute(
        sa.text(
            """
            INSERT INTO tenant_members (id, tenant_id, user_id, role, created_at)
            SELECT gen_random_uuid(),
                   tenant_id,
                   user_name,
                   CASE
                       WHEN role = 'editor' THEN 'member'
                       WHEN role IN ('admin', 'member', 'viewer') THEN role
                       ELSE 'member'
                   END,
                   COALESCE(created_at, NOW())
            FROM api_keys
            ON CONFLICT (tenant_id, user_id) DO NOTHING
            """
        )
    )

    # 6. Create non-superuser NOBYPASSRLS role tm_app_user for transaction-scoped RLS enforcement
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                    CREATE ROLE {APP_ROLE} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
                ELSE
                    ALTER ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS;
                END IF;
            END
            $$;
            """
        )
    )
    op.execute(sa.text(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}"))
    op.execute(
        sa.text(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {APP_ROLE}"
        )
    )
    op.execute(
        sa.text(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    )
    op.execute(
        sa.text(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {APP_ROLE}"
        )
    )
    op.execute(
        sa.text(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT USAGE, SELECT ON SEQUENCES TO {APP_ROLE}"
        )
    )

    # 7. Enable + Force RLS and install strict tenant isolation policies on all 12 business tables
    for tbl in RLS_TABLES:
        op.execute(sa.text(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY"))
        op.execute(sa.text(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY"))
        op.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation_{tbl} ON {tbl}"))
        op.execute(
            sa.text(
                f"""
                CREATE POLICY tenant_isolation_{tbl} ON {tbl}
                USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
                WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
                """
            )
        )


def downgrade() -> None:
    # 1. Disable and drop RLS policies
    for tbl in reversed(RLS_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation_{tbl} ON {tbl}"))
        op.execute(sa.text(f"ALTER TABLE {tbl} NO FORCE ROW LEVEL SECURITY"))
        op.execute(sa.text(f"ALTER TABLE {tbl} DISABLE ROW LEVEL SECURITY"))

    # 2. Restore api_keys unique constraint and drop label
    op.drop_index("ix_api_keys_tenant_user", table_name="api_keys")
    op.execute(
        sa.text(
            """
            DELETE FROM api_keys a
            USING api_keys b
            WHERE a.user_name = b.user_name AND a.id > b.id
            """
        )
    )
    op.create_unique_constraint("api_keys_user_name_key", "api_keys", ["user_name"])
    op.drop_column("api_keys", "label")

    # 3. Restore entities unique constraint
    op.drop_index("ix_entities_tenant_project", table_name="entities")
    op.drop_constraint("uq_entity_tenant_name_project", "entities", type_="unique")
    op.create_unique_constraint("uq_entity_name_project", "entities", ["name", "project"])

    # 4. Drop composite indexes
    op.drop_index("ix_exp_entities_tenant", table_name="experience_entities")
    op.drop_index("ix_relationships_tenant_src", table_name="relationships")
    op.drop_index("ix_search_logs_tenant_created", table_name="search_logs")
    op.drop_index("ix_personal_memories_tenant_user", table_name="personal_memories")
    op.drop_index("ix_doc_tree_nodes_tenant", table_name="document_tree_nodes")
    op.drop_index("ix_archive_upload_failures_tenant", table_name="archive_upload_failures")
    op.drop_index("ix_archive_attachments_tenant", table_name="archive_attachments")
    op.drop_index("ix_archive_exp_links_tenant", table_name="archive_experience_links")
    op.drop_index("ix_archives_tenant_project", table_name="archives")
    op.drop_index("ix_exp_feedbacks_tenant_exp", table_name="experience_feedbacks")
    op.drop_index("ix_experiences_tenant_status", table_name="experiences")
    op.drop_index("ix_experiences_tenant_project", table_name="experiences")

    # 5. Drop tenant_id foreign keys & columns from all 13 tables
    for tbl in reversed(ALL_TENANT_TABLES):
        op.drop_constraint(f"fk_{tbl}_tenant_id", tbl, type_="foreignkey")
        op.drop_column(tbl, "tenant_id")

    # 6. Drop tenant_members and tenants
    op.drop_index("ix_tenant_members_tenant_id", table_name="tenant_members")
    op.drop_index("ix_tenant_members_user_id", table_name="tenant_members")
    op.drop_table("tenant_members")
    op.drop_table("tenants")

    # 7. Revoke privileges and drop tm_app_user role
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                    ALTER DEFAULT PRIVILEGES IN SCHEMA public
                        REVOKE ALL ON TABLES FROM {APP_ROLE};
                    ALTER DEFAULT PRIVILEGES IN SCHEMA public
                        REVOKE ALL ON SEQUENCES FROM {APP_ROLE};
                    REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {APP_ROLE};
                    REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM {APP_ROLE};
                    REVOKE USAGE ON SCHEMA public FROM {APP_ROLE};
                    DROP ROLE {APP_ROLE};
                END IF;
            END
            $$;
            """
        )
    )

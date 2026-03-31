"""MVP cleanup: add group_key, drop deleted columns & tables.

Revision ID: 002_mvp_cleanup
Revises: f1g2h3i4j5k6
Create Date: 2026-03-29

Bridges existing DB schema to the simplified MVP model.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "002_mvp_cleanup"
down_revision = "001_initial_mvp"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- 1. Add group_key column to experiences (idempotent: 001 may already have it) --
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'experiences'
                  AND column_name = 'group_key'
            ) THEN
                ALTER TABLE experiences ADD COLUMN group_key VARCHAR(200);
                CREATE INDEX ix_experiences_group_key ON experiences (group_key);
            END IF;
        END $$;
    """)

    # -- 2. Drop deleted tables (order: dependents first) --
    for table in [
        "personal_tasks",       # FK → task_groups, experiences
        "task_dependencies",    # FK → personal_tasks
        "task_messages",        # FK → personal_tasks
        "task_groups",
        "experience_artifacts",
        "experience_reflections",
        "experience_versions",
        "experience_links",
        "experience_file_locations",
        "user_expansion_configs",
        "custom_installable_contents",
        "audit_logs",
        "query_logs",
        "tool_usage_logs",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    # -- 3. Drop deleted columns from experiences --
    for col in [
        "programming_language",
        "framework",
        "code_snippets",
        "source_context",
        "root_cause",
        "severity",
        "category",
        "progress_status",
        "structured_data",
        "git_refs",
        "related_links",
        "quality_score",
        "pinned",
        "last_decay_date",
        "fts_title_text",
        "fts_desc_text",
        "fts_solution_text",
        "summary",
        "embedding_dim",
        "embedding_status",
        "last_used_at",
        "view_count",
        "avg_rating",
    ]:
        op.execute(f"ALTER TABLE experiences DROP COLUMN IF EXISTS {col}")

    # -- 4. Drop orphaned indexes (columns no longer exist) --
    for idx in [
        "idx_exp_category",
        "idx_exp_severity",
        "idx_exp_progress",
        "ix_experiences_embedding_status",
        "ix_experiences_last_used_at",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {idx}")

    # -- 5. Add project+status composite index if missing --
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_experiences_project_status
        ON experiences (project, exp_status)
    """)

    # -- 6. Migrate document_tree_nodes FK from experience_id to archive_id --
    # Check if the old FK column exists
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'document_tree_nodes'
                AND column_name = 'experience_id'
            ) AND NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'document_tree_nodes'
                AND column_name = 'archive_id'
            ) THEN
                ALTER TABLE document_tree_nodes
                    ADD COLUMN archive_id uuid;
                ALTER TABLE document_tree_nodes
                    ADD CONSTRAINT document_tree_nodes_archive_id_fkey
                    FOREIGN KEY (archive_id) REFERENCES archives(id) ON DELETE CASCADE;
                CREATE INDEX ix_document_tree_nodes_archive_id
                    ON document_tree_nodes(archive_id);
                -- Drop old FK
                ALTER TABLE document_tree_nodes
                    DROP CONSTRAINT IF EXISTS document_tree_nodes_experience_id_fkey;
                ALTER TABLE document_tree_nodes
                    DROP COLUMN experience_id;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # This is a one-way migration; downgrade is not supported.
    raise NotImplementedError("Downgrade not supported for MVP cleanup migration")

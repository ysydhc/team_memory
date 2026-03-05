"""Phase 1-3 model enhancements: root_cause, FTS, review workflow, soft delete, query logs.

Revision ID: c2d3e4f5a6b7
Revises: b1a2c3d4e5f6
Create Date: 2026-02-10
"""

import sqlalchemy as sa
from alembic import op

revision = "c2d3e4f5a6b7"
down_revision = "b1a2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- 1. Add root_cause column ---
    op.add_column("experiences", sa.Column("root_cause", sa.Text(), nullable=True))

    # --- 2. Add review/publish workflow columns ---
    op.add_column(
        "experiences",
        sa.Column(
            "publish_status",
            sa.String(20),
            nullable=False,
            server_default="published",
        ),
    )
    op.add_column(
        "experiences",
        sa.Column(
            "review_status",
            sa.String(20),
            nullable=False,
            server_default="approved",
        ),
    )
    op.add_column(
        "experiences", sa.Column("reviewed_by", sa.String(100), nullable=True)
    )
    op.add_column("experiences", sa.Column("review_note", sa.Text(), nullable=True))

    # --- 3. Add soft delete columns ---
    op.add_column(
        "experiences",
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "experiences",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- 4. Add FTS tsvector column + trigger ---
    op.add_column(
        "experiences",
        sa.Column("fts", sa.dialects.postgresql.TSVECTOR(), nullable=True),
    )

    # Populate FTS for existing rows
    op.execute("""
        UPDATE experiences SET fts =
            to_tsvector('english',
                coalesce(title, '') || ' ' ||
                coalesce(description, '') || ' ' ||
                coalesce(root_cause, '') || ' ' ||
                coalesce(solution, '') || ' ' ||
                coalesce(array_to_string(tags, ' '), '')
            )
    """)

    # Create trigger to auto-update FTS on insert/update
    op.execute("""
        CREATE OR REPLACE FUNCTION experiences_fts_update() RETURNS trigger AS $$
        BEGIN
            NEW.fts := to_tsvector('english',
                coalesce(NEW.title, '') || ' ' ||
                coalesce(NEW.description, '') || ' ' ||
                coalesce(NEW.root_cause, '') || ' ' ||
                coalesce(NEW.solution, '') || ' ' ||
                coalesce(array_to_string(NEW.tags, ' '), '')
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_experiences_fts_update
        BEFORE INSERT OR UPDATE ON experiences
        FOR EACH ROW EXECUTE FUNCTION experiences_fts_update();
    """)

    # GIN index for FTS
    op.execute("""
        CREATE INDEX idx_exp_fts ON experiences USING gin (fts);
    """)

    # Index for soft delete + publish status queries
    op.execute("""
        CREATE INDEX idx_exp_published_active ON experiences (publish_status, is_deleted)
        WHERE publish_status = 'published' AND is_deleted = false;
    """)

    # --- 5. Create query_logs table ---
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("user_name", sa.String(100), nullable=False),
        sa.Column("source", sa.String(50), server_default="mcp"),
        sa.Column("result_count", sa.Integer(), server_default="0"),
        sa.Column("search_type", sa.String(20), server_default="vector"),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("query_logs")
    op.execute("DROP TRIGGER IF EXISTS trg_experiences_fts_update ON experiences")
    op.execute("DROP FUNCTION IF EXISTS experiences_fts_update()")
    op.execute("DROP INDEX IF EXISTS idx_exp_fts")
    op.execute("DROP INDEX IF EXISTS idx_exp_published_active")
    op.drop_column("experiences", "fts")
    op.drop_column("experiences", "deleted_at")
    op.drop_column("experiences", "is_deleted")
    op.drop_column("experiences", "review_note")
    op.drop_column("experiences", "reviewed_by")
    op.drop_column("experiences", "review_status")
    op.drop_column("experiences", "publish_status")
    op.drop_column("experiences", "root_cause")

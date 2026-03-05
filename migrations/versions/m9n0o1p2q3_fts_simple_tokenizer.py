"""Switch FTS from english to simple tokenizer for Chinese support.

The 'english' text search configuration ignores Chinese characters entirely.
The 'simple' configuration tokenizes by whitespace/punctuation. Combined with
application-layer jieba tokenization (stored in fts_text), this enables
Chinese full-text search.

Migration strategy (safe, with rollback):
1. Add fts_text column for pre-tokenized text
2. Update trigger to use 'simple' on fts_text (with fallback to raw fields)
3. Rebuild FTS for existing rows

Revision ID: m9n0o1p2q3f
Revises: m9n0o1p2q3 (tool_usage_logs)
"""
import sqlalchemy as sa
from alembic import op

revision = "m9n0o1p2q3f"
down_revision = "m9n0o1p2q3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Add fts_text column for application-layer pre-tokenized text
    op.add_column(
        "experiences",
        sa.Column("fts_text", sa.Text(), nullable=True),
    )

    # Step 2: Update trigger to prefer fts_text when available, fallback to raw
    op.execute("""
        CREATE OR REPLACE FUNCTION experiences_fts_update() RETURNS trigger AS $$
        BEGIN
            IF NEW.fts_text IS NOT NULL AND NEW.fts_text != '' THEN
                NEW.fts := to_tsvector('simple', NEW.fts_text);
            ELSE
                NEW.fts := to_tsvector('simple',
                    coalesce(NEW.title, '') || ' ' ||
                    coalesce(NEW.description, '') || ' ' ||
                    coalesce(NEW.root_cause, '') || ' ' ||
                    coalesce(NEW.solution, '') || ' ' ||
                    coalesce(array_to_string(NEW.tags, ' '), '')
                );
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Step 3: Rebuild FTS for existing rows (using raw text + simple tokenizer)
    op.execute("""
        UPDATE experiences SET fts =
            to_tsvector('simple',
                coalesce(title, '') || ' ' ||
                coalesce(description, '') || ' ' ||
                coalesce(root_cause, '') || ' ' ||
                coalesce(solution, '') || ' ' ||
                coalesce(array_to_string(tags, ' '), '')
            )
    """)


def downgrade() -> None:
    # Revert trigger to english tokenizer
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
        UPDATE experiences SET fts =
            to_tsvector('english',
                coalesce(title, '') || ' ' ||
                coalesce(description, '') || ' ' ||
                coalesce(root_cause, '') || ' ' ||
                coalesce(solution, '') || ' ' ||
                coalesce(array_to_string(tags, ' '), '')
            )
    """)

    op.drop_column("experiences", "fts_text")

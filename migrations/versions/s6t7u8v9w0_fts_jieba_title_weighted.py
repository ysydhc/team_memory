"""FTS scheme C: jieba tokenized columns with title-weighted ranking.

Adds fts_title_text, fts_desc_text, fts_solution_text for application-layer
jieba tokenization. Trigger uses setweight (A=title, B=desc, C=solution)
for ts_rank_cd. Fallback to fts_text or raw when new columns are null.

Revision ID: s6t7u8v9w0
Revises: r5s6t7u8v9
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "s6t7u8v9w0"
down_revision: Union[str, Sequence[str], None] = "r5s6t7u8v9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiences",
        sa.Column("fts_title_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("fts_desc_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "experiences",
        sa.Column("fts_solution_text", sa.Text(), nullable=True),
    )

    op.execute("""
        CREATE OR REPLACE FUNCTION experiences_fts_update() RETURNS trigger AS $$
        BEGIN
            IF (NEW.fts_title_text IS NOT NULL AND NEW.fts_title_text != '')
               OR (NEW.fts_desc_text IS NOT NULL AND NEW.fts_desc_text != '')
               OR (NEW.fts_solution_text IS NOT NULL AND NEW.fts_solution_text != '') THEN
                NEW.fts :=
                    setweight(to_tsvector('simple', coalesce(NEW.fts_title_text, '')), 'A') ||
                    setweight(to_tsvector('simple', coalesce(NEW.fts_desc_text, '')), 'B') ||
                    setweight(to_tsvector('simple', coalesce(NEW.fts_solution_text, '')), 'C');
            ELSIF NEW.fts_text IS NOT NULL AND NEW.fts_text != '' THEN
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


def downgrade() -> None:
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
    op.drop_column("experiences", "fts_solution_text")
    op.drop_column("experiences", "fts_desc_text")
    op.drop_column("experiences", "fts_title_text")

"""Fix FTS trigger: remove fts_text reference for DBs without that column.

Some DBs (e.g. created via different migration path) lack fts_text. The trigger
in s6t7u8v9w0 references NEW.fts_text in ELSIF, causing "record has no field
fts_text" on INSERT. This migration updates the trigger to only use
fts_title_text/desc/solution or raw fallback.

Revision ID: t7u8v9w0x1
Revises: s6t7u8v9w0
"""
from typing import Sequence, Union

from alembic import op

revision: str = "t7u8v9w0x1"
down_revision: Union[str, Sequence[str], None] = "s6t7u8v9w0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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
    # Revert to trigger that includes fts_text (for DBs that have it)
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

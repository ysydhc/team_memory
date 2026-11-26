"""Preserve application-tokenized NEW.fts in experiences_fts_update trigger.

Revision ID: 027_fts_trigger_preserve
Revises: 026_memory_accuracy_defenses

When ExperienceRepository writes/updates an Experience, it populates NEW.fts
using team_memory.utils.tokenizer.tokenize() (Chinese jieba + technical
identifier splitting for camelCase, snake_case, hyphenated codes, versions).
This trigger update ensures the DB trigger preserves application-supplied
NEW.fts instead of overwriting it with raw PostgreSQL to_tsvector('simple', ...),
while keeping the SQL fallback intact for direct raw SQL inserts/updates.
"""

from __future__ import annotations

from alembic import op

revision = "027_fts_trigger_preserve"
down_revision = "026_memory_accuracy_defenses"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.experiences_fts_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $function$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.fts IS NULL THEN
                    NEW.fts :=
                        setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
                        setweight(
                            to_tsvector(
                                'simple',
                                coalesce(NEW.description, '') || ' ' || coalesce(array_to_string(NEW.tags, ' '), '')
                            ),
                            'B'
                        ) ||
                        setweight(to_tsvector('simple', coalesce(NEW.solution, '')), 'C');
                END IF;
            ELSIF TG_OP = 'UPDATE' THEN
                IF NEW.fts IS NULL OR NEW.fts IS NOT DISTINCT FROM OLD.fts THEN
                    NEW.fts :=
                        setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
                        setweight(
                            to_tsvector(
                                'simple',
                                coalesce(NEW.description, '') || ' ' || coalesce(array_to_string(NEW.tags, ' '), '')
                            ),
                            'B'
                        ) ||
                        setweight(to_tsvector('simple', coalesce(NEW.solution, '')), 'C');
                END IF;
            END IF;
            RETURN NEW;
        END;
        $function$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.experiences_fts_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $function$
        BEGIN
            NEW.fts :=
                setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
                setweight(
                    to_tsvector(
                        'simple',
                        coalesce(NEW.description, '') || ' ' || coalesce(array_to_string(NEW.tags, ' '), '')
                    ),
                    'B'
                ) ||
                setweight(to_tsvector('simple', coalesce(NEW.solution, '')), 'C');
            RETURN NEW;
        END;
        $function$;
        """
    )

"""Merge fragmented project-name variants into canonical slugs.

Historically, experiences were written via the MCP server (which normalized
names to snake_case) while archives came through Web/CLI paths that stored the
raw client value. The result: the same project lived under several names
("adx-sdk-android", "gensoft-ipcs-mobile/adx-sdk-android", "agent/team_doc"...),
and since recall filters by exact project match, those records were invisible
to the project's searches.

This migration rewrites the known historical variants to their canonical slug
across experiences, archives, and search_logs. Going forward, all write paths
share team_memory.project_resolver.canonicalize_project_name + alias resolution,
so new variants should not appear.

Revision ID: 021_merge_project_name_variants
Revises: 020_search_log_noise_flag
"""

from __future__ import annotations

from alembic import op

revision = "021_merge_project_name_variants"
down_revision = "020_search_log_noise_flag"
branch_labels = None
depends_on = None

# variant → canonical
_MERGE_MAP = {
    "adx-sdk-android": "adx_sdk_android",
    "gensoft-ipcs-mobile-adx-sdk-android": "adx_sdk_android",
    "gensoft-ipcs-mobile/adx-sdk-android": "adx_sdk_android",
    "agent/team_doc": "team_memory",
    "team_doc": "team_memory",
    "team_memory_project": "team_memory",
    "litellm": "litellm_claude_code",
    "MyGuguGaga": "mygugugaga",
    "key-map": "key_map",
}

_TABLES = ("experiences", "archives", "search_logs")


def upgrade() -> None:
    conn = op.get_bind()
    from sqlalchemy import text

    for table in _TABLES:
        for variant, canonical in _MERGE_MAP.items():
            conn.execute(
                text(f"UPDATE {table} SET project = :canonical WHERE project = :variant"),
                {"canonical": canonical, "variant": variant},
            )


def downgrade() -> None:
    # Irreversible: the original per-record variant is not retained.
    pass

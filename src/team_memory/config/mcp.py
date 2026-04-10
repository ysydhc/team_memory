"""MCP server output control configuration."""

from __future__ import annotations

from pydantic import BaseModel


class MCPConfig(BaseModel):
    """MCP server output control."""

    max_output_tokens: int = 4000
    truncate_solution_at: int = 6250
    profile_max_strings_per_side: int = 20
    max_content_chars: int = 200_000  # max chars for memory_save content parameter
    # Stricter than HTTP ArchiveCreateRequest (64k) if set lower via config YAML.
    max_archive_solution_doc_chars: int = 64_000
    max_tags: int = 20
    max_tag_length: int = 50

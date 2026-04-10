"""Web server and uploads configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebConfig(BaseModel):
    """Web server configuration."""

    host: str = "0.0.0.0"
    port: int = 9111
    ssl_keyfile: str | None = None
    ssl_certfile: str | None = None
    max_request_body_bytes: int = 20_971_520  # 20 MB
    max_text_field_bytes: int = 64_000  # 64 KB for solution_doc; other fields smaller
    max_raw_conversation_bytes: int = 640_000  # 640 KB for raw_conversation
    rate_limit_per_minute: int = 200


class UploadsConfig(BaseModel):
    """Local multipart uploads for archive attachments (MVP disk storage)."""

    enabled: bool = True
    root_dir: str = ".tmp/uploads"
    max_bytes: int = 52_428_800  # ~50 MiB
    # Empty list = allow any non-empty extension; non-empty = lowercase suffix whitelist
    allowed_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt", ".json", ".pdf"])

"""Tests for Archive-related ORM models."""

from team_memory.storage.models import (
    Archive,
    ArchiveAttachment,
    ArchiveExperienceLink,
)


def test_archive_model_has_required_columns():
    assert hasattr(Archive, "title")
    assert hasattr(Archive, "solution_doc")
    assert hasattr(Archive, "overview")
    assert hasattr(Archive, "status")
    assert hasattr(ArchiveExperienceLink, "archive_id")
    assert hasattr(ArchiveAttachment, "kind")

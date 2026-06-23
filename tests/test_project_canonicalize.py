"""Unit tests for project-name canonicalization.

Canonicalization is the single source of truth shared by every write path, so
"adx-sdk-android", a full path, and "MyGuguGaga" all collapse to one stable
slug — otherwise recall (which filters by exact project match) fragments.
"""

from __future__ import annotations

import pytest

from team_memory.project_resolver import canonicalize_project_name


@pytest.mark.unit
class TestCanonicalizeProjectName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("adx-sdk-android", "adx_sdk_android"),
            ("adx_sdk_android", "adx_sdk_android"),
            ("gensoft-ipcs-mobile/adx-sdk-android", "adx_sdk_android"),
            ("gensoft-ipcs-mobile\\adx-sdk-android", "adx_sdk_android"),
            ("agent/team_doc", "team_doc"),
            ("MyGuguGaga", "mygugugaga"),
            ("key-map", "key_map"),
            ("  Team Memory  ", "team_memory"),
            ("a--b__c", "a_b_c"),
            ("trailing-_", "trailing"),
            ("UPPER_Case", "upper_case"),
        ],
    )
    def test_canonical(self, raw, expected):
        assert canonicalize_project_name(raw) == expected

    def test_empty_and_none(self):
        assert canonicalize_project_name("") == ""
        assert canonicalize_project_name(None) == ""
        assert canonicalize_project_name("   ") == ""

    def test_idempotent(self):
        once = canonicalize_project_name("gensoft-ipcs-mobile/adx-sdk-android")
        assert canonicalize_project_name(once) == once

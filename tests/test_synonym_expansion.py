"""Tests for query synonym expansion in the search pipeline."""

from team_memory.services.search_pipeline import _expand_query_synonyms


class TestExpandQuerySynonyms:
    def test_empty_query_returns_empty(self):
        assert _expand_query_synonyms("", {"数据库": "PostgreSQL"}) == ""
        assert _expand_query_synonyms("  ", {"数据库": "PostgreSQL"}) == ""

    def test_empty_synonyms_returns_query_unchanged(self):
        assert _expand_query_synonyms("数据库 优化", {}) == "数据库 优化"
        assert _expand_query_synonyms("数据库", None) == "数据库"

    def test_key_in_query_appends_value(self):
        synonyms = {"数据库": "PostgreSQL", "PG": "PostgreSQL"}
        assert _expand_query_synonyms("数据库 优化", synonyms) == "数据库 优化 PostgreSQL"

    def test_value_in_query_appends_key(self):
        synonyms = {"数据库": "PostgreSQL"}
        assert _expand_query_synonyms("PostgreSQL 连接", synonyms) == "PostgreSQL 连接 数据库"

    def test_both_key_and_value_no_duplicate(self):
        synonyms = {"UI": "界面"}
        # "UI" in query -> add "界面"; "界面" not in query. So result has 界面 once.
        assert _expand_query_synonyms("UI 优化", synonyms) == "UI 优化 界面"

    def test_no_match_returns_unchanged(self):
        synonyms = {"数据库": "PostgreSQL"}
        assert _expand_query_synonyms("前端 性能", synonyms) == "前端 性能"

    def test_deduplicate_added_terms(self):
        synonyms = {"数据库": "PostgreSQL", "PG": "PostgreSQL"}
        # "数据库" and "PG" both map to PostgreSQL; add once
        out = _expand_query_synonyms("数据库 PG", synonyms)
        assert out == "数据库 PG PostgreSQL"
        assert out.count("PostgreSQL") == 1

    def test_skip_empty_key_or_value(self):
        synonyms = {"": "PostgreSQL", "数据库": ""}
        assert _expand_query_synonyms("数据库", synonyms) == "数据库"

    def test_skip_key_equals_value(self):
        synonyms = {"same": "same"}
        assert _expand_query_synonyms("same", synonyms) == "same"

"""Tests for Chinese-English technical tokenizer and FTS alignment.

Covers:
- Technical identifier expansion (camelCase, ACRONYM+CamelCase, snake_case,
  kebab-case/error codes, versions, alphanumeric boundaries, file paths)
- Chinese segmentation (jieba + CJK bigram fallback)
- Mixed Chinese/English text
- Punctuation filtering, order-preserving deduplication, edge cases
- Write-side (`_build_fts_tsvector_expr`) and read-side (`_build_hybrid_tsquery`)
  consistency for FTS recall
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from team_memory.utils.tokenizer import (
    _tokenize_simple,
    load_tech_dict,
    load_tech_dict_from_tags,
    tokenize,
)


class TestTokenizeTechnicalIdentifiers:
    """Verified test matrix for camelCase, acronyms, snake_case, kebab-case, versions, paths."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Standard PascalCase / camelCase
            ("AdLifecycleMonitor", "adlifecyclemonitor ad lifecycle monitor"),
            ("getUserProfile", "getuserprofile get user profile"),
            # Acronym + CamelCase
            ("ADXManager", "adxmanager adx manager"),
            ("HTTPServer", "httpserver http server"),
            ("parseXMLResponse", "parsexmlresponse parse xml response"),
            ("iOSDevice", "iosdevice os device"),
            # snake_case & SCREAMING_SNAKE_CASE & extensions
            ("use_worktree", "use_worktree use worktree"),
            ("use_worktree.sh", "use_worktree.sh use_worktree use worktree sh"),
            ("MAX_RETRY_COUNT", "max_retry_count max retry count"),
            # Hyphenated / error codes
            ("adx-sdk-518", "adx-sdk-518 adx sdk 518"),
            ("ERR-404-TIMEOUT", "err-404-timeout err 404 timeout"),
            # Version numbers
            ("v0.3.5", "v0.3.5 0.3.5"),
            ("PostgreSQL 16.4", "postgresql postgre sql 16.4"),
            # Alphanumeric boundaries
            ("Android16 异步日志", "android16 android 16 异步 日志"),
            # File paths
            ("src/team_memory/tokenizer.py", "src team_memory team memory tokenizer.py tokenizer py"),
        ],
    )
    def test_technical_identifier_matrix(self, raw: str, expected: str) -> None:
        assert tokenize(raw) == expected

    def test_simple_fallback_matches_identifier_expansion(self) -> None:
        assert _tokenize_simple("AdLifecycleMonitor") == "adlifecyclemonitor ad lifecycle monitor"
        assert _tokenize_simple("ADXManager") == "adxmanager adx manager"
        assert _tokenize_simple("adx-sdk-518") == "adx-sdk-518 adx sdk 518"
        assert _tokenize_simple("use_worktree.sh") == "use_worktree.sh use_worktree use worktree sh"
        assert _tokenize_simple("v0.3.5") == "v0.3.5 0.3.5"


class TestTokenizeChinese:
    def test_chinese_segmentation(self) -> None:
        result = tokenize("集成测试误删生产数据库经验数据")
        assert "数据库" in result
        assert "集成" in result
        assert "测试" in result

    def test_chinese_search_term(self) -> None:
        result = tokenize("数据库")
        assert "数据库" in result

    def test_chinese_ui_term(self) -> None:
        result = tokenize("UI优化")
        assert "ui" in result.split()
        assert "优化" in result.split()


class TestTokenizeEnglish:
    def test_english_preserved_lowercased(self) -> None:
        result = tokenize("Docker container networking")
        assert result == "docker container networking"

    def test_technical_terms(self) -> None:
        result = tokenize("PostgreSQL FastAPI React")
        tokens = result.split()
        assert "postgresql" in tokens
        assert "postgre" in tokens
        assert "sql" in tokens
        assert "fastapi" in tokens
        assert "fast" in tokens
        assert "api" in tokens
        assert "react" in tokens


class TestTokenizeMixed:
    def test_chinese_english_mixed(self) -> None:
        result = tokenize("Docker容器网络配置问题")
        tokens = result.split()
        assert "docker" in tokens
        assert len(tokens) >= 3

    def test_mixed_camel_and_chinese(self) -> None:
        result = tokenize("修复AdLifecycleMonitor内存泄漏")
        assert result == "修复 adlifecyclemonitor ad lifecycle monitor 内存 泄漏"

    def test_mixed_hyphen_script_and_chinese(self) -> None:
        result = tokenize("排查adx-sdk-518与use_worktree.sh冲突")
        assert (
            result
            == "排查 adx-sdk-518 adx sdk 518 与 use_worktree.sh use_worktree use worktree sh 冲突"
        )

    def test_dedup_and_symbol_filtering(self) -> None:
        result = tokenize("问题：HTTPServer 与 httpserver？(v0.3.5)")
        assert result == "问题 httpserver http server 与 v0.3.5 0.3.5"


class TestTokenizeEdgeCases:
    def test_empty_string(self) -> None:
        assert tokenize("") == ""

    def test_whitespace_only(self) -> None:
        assert tokenize("   ") == ""

    def test_single_char(self) -> None:
        result = tokenize("a")
        assert result.strip() == "a"

    def test_punctuation_stripped(self) -> None:
        result = tokenize("问题：如何解决？")
        assert "问题" in result
        assert "解决" in result
        assert "：" not in result
        assert "？" not in result


class TestTechDictionary:
    def test_load_tech_dict(self) -> None:
        load_tech_dict(["PostgreSQL", "FastAPI", "团队经验库"])
        result = tokenize("团队经验库的PostgreSQL配置")
        tokens = result.split()
        assert "团队经验库" in tokens
        assert "postgresql" in tokens

    def test_load_from_tags(self) -> None:
        tag_synonyms = {"PG": "PostgreSQL", "JS": "JavaScript"}
        tags = ["docker", "kubernetes", "团队经验"]
        load_tech_dict_from_tags(tag_synonyms, tags)
        result = tokenize("PostgreSQL数据库")
        tokens = result.split()
        assert "postgresql" in tokens
        assert "数据库" in tokens


class TestFtsWriteReadAlignment:
    """Verify write-side tsvector expression and read-side tsquery share identical lexemes."""

    def test_partial_keyword_matches_indexed_tokens(self) -> None:
        indexed_doc = set(tokenize("修复 AdLifecycleMonitor 与 use_worktree.sh 在 adx-sdk-518 的冲突").split())
        for query in ("Lifecycle", "Monitor", "worktree", "518", "adx-sdk-518", "AdLifecycleMonitor"):
            query_tokens = set(tokenize(query).split())
            assert query_tokens & indexed_doc, f"query {query!r} ({query_tokens}) failed to overlap {indexed_doc}"

    def test_build_fts_tsvector_expr_compiles_with_weights(self) -> None:
        from sqlalchemy.dialects import postgresql

        from team_memory.storage.repository import _build_fts_tsvector_expr

        expr = _build_fts_tsvector_expr(
            title="AdLifecycleMonitor 修复",
            description="处理 adx-sdk-518 与 use_worktree.sh 异常",
            solution="升级到 v0.3.5",
            tags=["ADXManager"],
        )
        compiled = expr.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        params = compiled.params
        assert "setweight" in sql
        assert "to_tsvector" in sql
        param_values = " ".join(str(v) for v in params.values())
        assert "lifecycle" in param_values
        assert "518" in param_values
        assert "worktree" in param_values
        assert "adxmanager" in param_values
        assert "0.3.5" in param_values

    def test_hybrid_tsquery_accepts_expanded_technical_queries(self) -> None:
        from sqlalchemy.dialects import postgresql

        from team_memory.storage.query_builders import _build_hybrid_tsquery

        for raw_query in ("Lifecycle", "adx-sdk-518", "AdLifecycleMonitor", "use_worktree.sh 报错"):
            ts_q = _build_hybrid_tsquery(tokenize(raw_query))
            compiled = ts_q.compile(dialect=postgresql.dialect())
            assert compiled is not None
            assert str(compiled)

    def test_repository_is_postgres_detection(self) -> None:
        from team_memory.storage.repository import ExperienceRepository

        pg_session = MagicMock()
        pg_session.sync_session.bind.dialect.name = "postgresql"
        assert ExperienceRepository(pg_session)._is_postgres() is True

        sqlite_session = MagicMock()
        sqlite_session.sync_session.bind.dialect.name = "sqlite"
        assert ExperienceRepository(sqlite_session)._is_postgres() is False

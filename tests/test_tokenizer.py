"""Tests for Step 3b: jieba tokenizer module.

Covers Chinese segmentation, English preservation, mixed text,
edge cases, and tech dictionary loading.
"""

from __future__ import annotations

from team_memory.services.tokenizer import (
    load_tech_dict,
    load_tech_dict_from_tags,
    tokenize,
)


class TestTokenizeChinese:
    def test_chinese_segmentation(self):
        result = tokenize("集成测试误删生产数据库经验数据")
        assert "数据库" in result
        assert "集成" in result
        assert "测试" in result

    def test_chinese_search_term(self):
        result = tokenize("数据库")
        assert "数据库" in result

    def test_chinese_ui_term(self):
        result = tokenize("UI优化")
        assert "UI" in result
        assert "优化" in result


class TestTokenizeEnglish:
    def test_english_preserved(self):
        result = tokenize("Docker container networking")
        assert "Docker" in result
        assert "container" in result
        assert "networking" in result

    def test_technical_terms(self):
        result = tokenize("PostgreSQL FastAPI React")
        assert "PostgreSQL" in result
        assert "FastAPI" in result


class TestTokenizeMixed:
    def test_chinese_english_mixed(self):
        result = tokenize("Docker容器网络配置问题")
        assert "Docker" in result
        tokens = result.split()
        assert len(tokens) >= 3


class TestTokenizeEdgeCases:
    def test_empty_string(self):
        assert tokenize("") == ""

    def test_whitespace_only(self):
        assert tokenize("   ") == ""

    def test_none_like(self):
        assert tokenize("") == ""

    def test_single_char(self):
        result = tokenize("a")
        assert result.strip() == "a"

    def test_punctuation_stripped(self):
        result = tokenize("问题：如何解决？")
        assert "问题" in result
        assert "解决" in result


class TestTechDictionary:
    def test_load_tech_dict(self):
        load_tech_dict(["PostgreSQL", "FastAPI", "团队经验库"])
        result = tokenize("团队经验库的PostgreSQL配置")
        assert "PostgreSQL" in result

    def test_load_from_tags(self):
        tag_synonyms = {"PG": "PostgreSQL", "JS": "JavaScript"}
        tags = ["docker", "kubernetes", "团队经验"]
        load_tech_dict_from_tags(tag_synonyms, tags)
        result = tokenize("PostgreSQL数据库")
        assert "PostgreSQL" in result
        assert "数据库" in result

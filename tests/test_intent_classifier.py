"""Tests for daemon.intent_classifier — keyword-based persona classification."""

from __future__ import annotations

from team_memory.daemon.integration.intent_classifier import classify_persona


class TestClassifyPersona:
    """Unit tests for classify_persona()."""

    # -- debugging --
    def test_error_keyword(self):
        assert classify_persona("这个报错怎么解决") == "debugging"

    def test_english_error(self):
        assert classify_persona("I got an error when running the script") == "debugging"

    def test_bug_keyword(self):
        assert classify_persona("there's a bug in the login flow") == "debugging"

    def test_exception_keyword(self):
        assert classify_persona("收到一个异常") == "debugging"

    def test_fix_keyword(self):
        assert classify_persona("fix the crash on startup") == "debugging"

    def test_http_status_code(self):
        assert classify_persona("服务返回502了") == "debugging"

    def test_timeout_keyword(self):
        assert classify_persona("接口超时了") == "debugging"

    def test_fail_keyword(self):
        assert classify_persona("deploy failed") == "debugging"

    # -- coding --
    def test_code_keyword(self):
        assert classify_persona("帮我写一段代码") == "coding"

    def test_implement_keyword(self):
        assert classify_persona("implement a new feature") == "coding"

    def test_function_keyword(self):
        assert classify_persona("写一个函数来处理这个") == "coding"

    def test_class_keyword(self):
        assert classify_persona("create a new class for auth") == "coding"

    def test_refactor_keyword(self):
        assert classify_persona("重构一下这个模块") == "coding"

    def test_deploy_keyword(self):
        assert classify_persona("部署到生产环境") == "coding"

    def test_pytest_keyword(self):
        assert classify_persona("add pytest for this module") == "coding"

    def test_api_keyword(self):
        assert classify_persona("设计一个API接口") == "coding"

    # -- writing --
    def test_doc_keyword(self):
        assert classify_persona("帮我写个文档") == "writing"

    def test_readme_keyword(self):
        assert classify_persona("update the README") == "writing"

    def test_article_keyword(self):
        assert classify_persona("写一篇文章") == "writing"

    def test_translate_keyword(self):
        assert classify_persona("翻译一下这段话") == "writing"

    def test_changelog_keyword(self):
        assert classify_persona("generate a CHANGELOG entry") == "writing"

    # -- research --
    def test_research_keyword(self):
        assert classify_persona("调研一下这个方案") == "research"

    def test_compare_keyword(self):
        assert classify_persona("compare React vs Vue") == "research"

    def test_analyze_keyword(self):
        assert classify_persona("分析一下性能瓶颈") == "research"

    def test_architecture_keyword(self):
        assert classify_persona("设计一个新的架构") == "research"

    def test_evaluate_keyword(self):
        assert classify_persona("evaluate these options") == "research"

    # -- general (fallback) --
    def test_general_no_keywords(self):
        assert classify_persona("你好") == "general"

    def test_general_casual(self):
        assert classify_persona("what's the weather like") == "general"

    def test_empty_string(self):
        assert classify_persona("") == "general"

    # -- priority: debugging wins over coding --
    def test_debugging_over_coding(self):
        """'error' should match debugging before any coding keywords."""
        assert classify_persona("test code throws error") == "debugging"

    def test_fix_test_is_debugging(self):
        """'fix' matches debugging before 'test' matches coding."""
        assert classify_persona("fix the failing test") == "debugging"

    # -- case insensitivity --
    def test_case_insensitive(self):
        assert classify_persona("IMPLEMENT a new CLASS") == "coding"
        assert classify_persona("BUG in production") == "debugging"
        assert classify_persona("Write DOCUMENTATION") == "writing"
        assert classify_persona("RESEARCH alternatives") == "research"

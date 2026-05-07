"""Tests for RefinementWorker — draft refinement and learning card detection / 草稿精炼与学习卡片检测测试

Covers:
- _is_learning_card title detection / 标题检测
- _is_learning_card content detection / 内容检测
- Non-learning cards should NOT be skipped / 非学习卡片不应跳过
"""
from __future__ import annotations

import pytest

from daemon.refinement_worker import RefinementWorker

pytestmark = pytest.mark.unit


class TestIsLearningCard:
    """Tests for _is_learning_card static method / _is_learning_card 静态方法测试"""

    # ── Title-based detection / 基于标题的检测 ──

    def test_card_prefix_chinese(self):
        """卡片- prefix should be detected / 卡片- 前缀应被检测"""
        assert RefinementWorker._is_learning_card("卡片-iOS数据存储") is True

    def test_topic_prefix_chinese(self):
        """主题- prefix should be detected / 主题- 前缀应被检测"""
        assert RefinementWorker._is_learning_card("主题-SwiftUI基础") is True

    def test_layer_prefix(self):
        """Layer N prefix should be detected / Layer N 前缀应被检测"""
        assert RefinementWorker._is_learning_card("Layer 0 iOS开发概览") is True
        assert RefinementWorker._is_learning_card("Layer 2 UIKit核心") is True

    def test_study_plan_prefix(self):
        """学习计划 prefix should be detected / 学习计划 前缀应被检测"""
        assert RefinementWorker._is_learning_card("学习计划-2026Q2") is True

    # ── Content-based detection / 基于内容的检测 ──

    def test_learning_objective_in_content(self):
        """学习目标 in content should be detected / 内容中包含学习目标应被检测"""
        assert RefinementWorker._is_learning_card("My Note", "## 学习目标\n掌握基础") is True

    def test_prerequisite_in_content(self):
        """前置知识 in content should be detected / 内容中包含前置知识应被检测"""
        assert RefinementWorker._is_learning_card("Setup", "前置知识: Python基础") is True

    def test_scan_content_in_content(self):
        """扫盲 in content should be detected / 内容中包含扫盲应被检测"""
        assert RefinementWorker._is_learning_card("Quick Guide", "iOS扫盲笔记") is True

    def test_layer_in_content(self):
        """Layer N in content should be detected / 内容中包含 Layer N 应被检测"""
        assert RefinementWorker._is_learning_card("Dev Guide", "这是 Layer 1 的内容") is True

    # ── Negative cases / 反例 ──

    def test_normal_experience_not_learning(self):
        """Normal experience should NOT be detected as learning / 普通经验不应被检测为学习卡片"""
        assert RefinementWorker._is_learning_card(
            "Docker PostgreSQL连接失败",
            "使用host.docker.internal连接PostgreSQL"
        ) is False

    def test_problem_solution_not_learning(self):
        """Problem-solution format should NOT be learning / 问题-解决方案格式不应为学习卡片"""
        assert RefinementWorker._is_learning_card(
            "OOM crash fix",
            "Problem: App crashes with OOM\nSolution: Use lazy loading"
        ) is False

    def test_empty_title_and_content(self):
        """Empty title and content should NOT be learning / 空标题和内容不应为学习卡片"""
        assert RefinementWorker._is_learning_card("", "") is False

    def test_partial_match_not_learning(self):
        """Partial match without clear pattern should NOT be learning / 部分匹配但无明确模式不应为学习卡片"""
        assert RefinementWorker._is_learning_card(
            "Docker学习笔记",
            "Docker容器使用指南"
        ) is False

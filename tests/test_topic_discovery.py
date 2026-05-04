"""Tests for topic_discovery service."""
from __future__ import annotations

import pytest

from team_memory.services.topic_discovery import (
    Topic,
    TopicResult,
    _simple_kmeans,
)


class TestSimpleKmeans:
    def test_basic_clustering(self):
        """Two well-separated clusters should be correctly identified."""
        import numpy as np
        rng = np.random.RandomState(42)
        # Cluster 1: centered around [1, 0, 0]
        c1 = rng.randn(10, 3) * 0.1 + np.array([1, 0, 0])
        # Cluster 2: centered around [0, 1, 0]
        c2 = rng.randn(10, 3) * 0.1 + np.array([0, 1, 0])
        vectors = {}
        for i in range(10):
            vectors[f"a{i}"] = c1[i].tolist()
        for i in range(10):
            vectors[f"b{i}"] = c2[i].tolist()
        clusters = _simple_kmeans(vectors, k=2)
        assert len(clusters) == 2
        # Each cluster should have ~10 items
        sizes = sorted(len(v) for v in clusters.values())
        assert sizes[0] >= 5
        assert sizes[1] >= 5

    def test_single_cluster(self):
        """k=1 should put everything in one cluster."""
        vectors = {f"e{i}": [float(i)] * 3 for i in range(5)}
        clusters = _simple_kmeans(vectors, k=1)
        assert len(clusters) == 1
        assert len(list(clusters.values())[0]) == 5

    def test_empty_input(self):
        """Empty vectors should return empty."""
        assert _simple_kmeans({}, k=3) == {}

    def test_k_larger_than_n(self):
        """If k >= n, each item gets its own cluster."""
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        clusters = _simple_kmeans(vectors, k=5)
        assert len(clusters) == 2
        # Each cluster has exactly 1 item
        for items in clusters.values():
            assert len(items) == 1


class TestTopic:
    def test_topic_defaults(self):
        t = Topic(id="abc", name="Test Topic")
        assert t.experience_ids == []
        assert t.tags == []
        assert t.entities == []

    def test_topic_with_data(self):
        t = Topic(
            id="abc",
            name="MCP / team_doc",
            experience_ids=["1", "2", "3"],
            tags=["docker", "mcp"],
            entities=["MCP", "PostgreSQL"],
            center=[0.1, 0.2, 0.3],
        )
        assert len(t.experience_ids) == 3
        assert t.center == [0.1, 0.2, 0.3]


class TestTopicResult:
    def test_empty_result(self):
        r = TopicResult()
        assert r.topics == []
        assert r.total_experiences == 0
        assert r.unclustered == 0

    def test_with_topics(self):
        t = Topic(id="x", name="X")
        r = TopicResult(topics=[t], total_experiences=10, unclustered=2)
        assert len(r.topics) == 1
        assert r.total_experiences == 10

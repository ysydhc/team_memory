"""Tests for FaithfulnessJudge — LLM-based faithfulness evaluator / 忠实度评估器测试

Covers:
- FaithfulnessResult dataclass / 结果数据类
- _parse_result JSON parsing / JSON 解析
- evaluate with empty response / 空回复处理
- _call_llm mocked / 模拟 LLM 调用
- Learning card detection / 学习卡片检测
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from daemon.faithfulness_judge import FaithfulnessJudge, FaithfulnessResult

pytestmark = pytest.mark.unit


class TestFaithfulnessResult:
    """Tests for FaithfulnessResult dataclass / 结果数据类测试"""

    def test_to_dict_returns_expected_fields(self):
        """to_dict should return all fields / to_dict 应返回所有字段"""
        result = FaithfulnessResult(
            score=0.75,
            total_claims=4,
            supported_claims=3,
            claims_detail=[
                {"claim": "A", "supported": True, "reason": "found"},
                {"claim": "B", "supported": False, "reason": "not found"},
            ],
            raw_response="...",
        )
        d = result.to_dict()
        assert d["score"] == 0.75
        assert d["total_claims"] == 4
        assert d["supported_claims"] == 3
        assert len(d["claims_detail"]) == 2
        # raw_response should NOT be in to_dict
        assert "raw_response" not in d

    def test_to_dict_empty_claims(self):
        """to_dict with zero claims / 无 claims 时的 to_dict"""
        result = FaithfulnessResult(
            score=1.0, total_claims=0, supported_claims=0,
            claims_detail=[], raw_response="",
        )
        d = result.to_dict()
        assert d["score"] == 1.0
        assert d["total_claims"] == 0


class TestParseResult:
    """Tests for _parse_result JSON parsing / JSON 解析测试"""

    def setup_method(self):
        self.judge = FaithfulnessJudge(
            base_url="http://fake:4000/v1",
            model="test-model",
            api_key="test-key",
        )

    def test_parse_valid_json(self):
        """Valid JSON should parse correctly / 有效 JSON 应正确解析"""
        raw = json.dumps({
            "claims": [
                {"claim": "Docker uses containers", "supported": True, "reason": "found in context"},
                {"claim": "K8s is easy", "supported": False, "reason": "not in context"},
            ],
            "score": 0.5,
        })
        result = self.judge._parse_result(raw)
        assert result.score == 0.5
        assert result.total_claims == 2
        assert result.supported_claims == 1

    def test_parse_json_with_markdown_code_block(self):
        """JSON wrapped in markdown code block should parse / 包裹在 markdown 代码块中的 JSON 应解析"""
        raw = '```json\n{"claims": [{"claim": "test", "supported": true, "reason": "ok"}], "score": 1.0}\n```'
        result = self.judge._parse_result(raw)
        assert result.score == 1.0
        assert result.total_claims == 1

    def test_parse_json_embedded_in_text(self):
        """JSON embedded in surrounding text should be extracted / 嵌入在文本中的 JSON 应被提取"""
        raw = 'Here is the result:\n{"claims": [{"claim": "x", "supported": false, "reason": "no"}], "score": 0.0}\nEnd.'
        result = self.judge._parse_result(raw)
        assert result.score == 0.0
        assert result.total_claims == 1

    def test_parse_invalid_json_returns_negative_score(self):
        """Invalid JSON should return score=-1 / 无效 JSON 应返回 score=-1"""
        result = self.judge._parse_result("not json at all")
        assert result.score == -1.0
        assert result.total_claims == 0

    def test_parse_empty_claims_returns_score_1(self):
        """No claims extracted should default to score 1.0 / 无 claims 时默认 score=1.0"""
        raw = json.dumps({"claims": [], "score": 1.0})
        result = self.judge._parse_result(raw)
        assert result.score == 1.0
        assert result.total_claims == 0

    def test_parse_missing_score_computed_from_claims(self):
        """Missing score field should be computed from claims / 缺少 score 字段应从 claims 计算"""
        raw = json.dumps({
            "claims": [
                {"claim": "A", "supported": True, "reason": "ok"},
                {"claim": "B", "supported": True, "reason": "ok"},
                {"claim": "C", "supported": False, "reason": "no"},
            ],
        })
        result = self.judge._parse_result(raw)
        assert result.total_claims == 3
        assert result.supported_claims == 2
        assert abs(result.score - 2 / 3) < 0.01


class TestEvaluate:
    """Tests for evaluate method / evaluate 方法测试"""

    def setup_method(self):
        self.judge = FaithfulnessJudge(
            base_url="http://fake:4000/v1",
            model="test-model",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self):
        """Empty response should return score=0 / 空回复应返回 score=0"""
        result = await self.judge.evaluate(
            query="test", contexts=["some context"], response=""
        )
        assert result.score == 0.0
        assert result.total_claims == 0

    @pytest.mark.asyncio
    async def test_evaluate_whitespace_response(self):
        """Whitespace-only response should return score=0 / 仅空白回复应返回 score=0"""
        result = await self.judge.evaluate(
            query="test", contexts=["ctx"], response="   \n  "
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_llm_call_mocked(self):
        """Evaluate should call LLM and parse result / evaluate 应调用 LLM 并解析结果"""
        mock_response = json.dumps({
            "claims": [
                {"claim": "PostgreSQL uses pgvector", "supported": True, "reason": "in context"},
            ],
            "score": 1.0,
        })

        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await self.judge.evaluate(
                query="pgvector setup",
                contexts=["pgvector extends PostgreSQL with vector similarity search"],
                response="PostgreSQL supports vector search via pgvector extension",
            )
            assert result.score == 1.0
            assert result.total_claims == 1
            assert result.supported_claims == 1

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure_returns_negative(self):
        """LLM failure should return score=-1 / LLM 失败应返回 score=-1"""
        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await self.judge.evaluate(
                query="test", contexts=["ctx"], response="some response"
            )
            assert result.score == -1.0

    @pytest.mark.asyncio
    async def test_evaluate_no_contexts(self):
        """No contexts should use placeholder text / 无检索结果应使用占位文本"""
        mock_response = json.dumps({
            "claims": [{"claim": "test", "supported": False, "reason": "no context"}],
            "score": 0.0,
        })

        with patch.object(self.judge, "_call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            result = await self.judge.evaluate(
                query="test", contexts=[], response="some response"
            )
            # Verify the prompt includes placeholder
            call_args = mock_llm.call_args[0][0]
            assert "无检索结果" in call_args
            assert result.score == 0.0


class TestCallLlm:
    """Tests for _call_llm HTTP interaction / _call_llm HTTP 交互测试"""

    @pytest.mark.asyncio
    async def test_call_llm_uses_correct_endpoint(self):
        """_call_llm should POST to /chat/completions / _call_llm 应 POST 到 /chat/completions"""
        judge = FaithfulnessJudge(
            base_url="http://fake:4000/v1",
            model="glm-4-flash",
            api_key="sk-test",
        )

        mock_data = {"choices": [{"message": {"content": "result"}}]}
        mock_resp = AsyncMock()
        mock_resp.json = lambda: mock_data
        mock_resp.raise_for_status = lambda: None

        async def fake_post(url, **kwargs):
            assert "/chat/completions" in url
            assert kwargs["json"]["model"] == "glm-4-flash"
            assert kwargs["headers"]["Authorization"] == "Bearer sk-test"
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post = fake_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            result = await judge._call_llm("test prompt")
            assert result == "result"

"""
Unit tests for GeminiJudge wrapper and Groq permission enforcement (tests/test_judge.py).
Tests token ledger, caching, rate limits, schema validation, and budget enforcement.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

sys.path.insert(0, ".")
from src.evaluation.judge import GeminiJudge, GroqJudge, BudgetExceededError, TokenAwareRateLimiter
import src.config as cfg


class SampleSchema(BaseModel):
    verdict: str
    reason: str


def test_groq_key_forbidden_when_provider_not_groq():
    """Verify Rule R3: GroqJudge raises PermissionError unless JUDGE_PROVIDER=groq."""
    with patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test123"}):
        with pytest.raises(PermissionError) as exc_info:
            GroqJudge(model_name="qwen/qwen3.8-27b", metric="test_metric")
        assert "Groq key must not be used unless JUDGE_PROVIDER=groq" in str(exc_info.value)


def test_gemini_judge_initialization():
    """Verify GeminiJudge initialization with api key and model name."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKey12345"}):
        with patch("google.genai.Client"):
            judge = GeminiJudge(model_name=cfg.JUDGE_MODEL_PRIMARY, metric="test_metric")
            assert judge.get_model_name() == cfg.JUDGE_MODEL_PRIMARY
            assert judge.metric == "test_metric"


def test_gemini_judge_disk_caching(tmp_path):
    """Verify that cached entries avoid external API calls."""
    cache_file = tmp_path / "judge_cache.json"
    cache_file.write_text("{}", encoding="utf-8")

    with patch("src.evaluation.judge.CACHE_PATH", cache_file), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKey12345"}):
        with patch("google.genai.Client"):
            judge = GeminiJudge(model_name=cfg.JUDGE_MODEL_PRIMARY, metric="faithfulness")
            
            with patch.object(judge, "_call_api_with_retry", return_value=('{"verdict": "yes", "reason": "Accurate"}', 10, 5)) as mock_api:
                # First call: calls API
                out1 = judge.generate("prompt 1")
                assert "verdict" in out1
                assert mock_api.call_count == 1

                # Second call: returns cached result without API call
                out2 = judge.generate("prompt 1")
                assert out2 == out1
                assert mock_api.call_count == 1


def test_rate_limiter_budget_enforcement(tmp_path):
    """Verify that daily request budget enforcement raises BudgetExceededError cleanly."""
    ledger_file = tmp_path / "judge_usage.jsonl"
    limiter = TokenAwareRateLimiter(daily_budget=2)

    with patch("src.evaluation.judge.USAGE_LEDGER_PATH", ledger_file), \
         patch.object(limiter, "get_today_request_count", return_value=3):
        with pytest.raises(BudgetExceededError) as exc_info:
            limiter.acquire("Test prompt")
        assert "JUDGE_DAILY_REQUEST_BUDGET" in str(exc_info.value)


def test_rate_limiter_custom_budget_stricter_than_model_cap(tmp_path):
    """Verify a small custom budget (e.g. 5) overrides a much higher model default (14400)."""
    ledger_file = tmp_path / "judge_usage.jsonl"
    limiter = TokenAwareRateLimiter(daily_budget=5)

    with patch("src.evaluation.judge.USAGE_LEDGER_PATH", ledger_file):
        # Non-exhausted: 4 requests done today
        with patch.object(limiter, "get_today_request_count", return_value=4):
            tokens, wait_time = limiter.acquire("Test prompt", model="gemma-4-31b-it")
            assert tokens > 0

        # Exact boundary: 5 requests done today -> exhausted
        with patch.object(limiter, "get_today_request_count", return_value=5):
            with pytest.raises(BudgetExceededError) as exc_info:
                limiter.acquire("Test prompt", model="gemma-4-31b-it")
            assert "JUDGE_DAILY_REQUEST_BUDGET=5" in str(exc_info.value)
            assert "gemma-4-31b-it" in str(exc_info.value)


def test_rate_limiter_model_default_budget(tmp_path):
    """Verify model default caps apply when instance daily_budget is None."""
    ledger_file = tmp_path / "judge_usage.jsonl"
    limiter = TokenAwareRateLimiter(daily_budget=None)

    with patch("src.evaluation.judge.USAGE_LEDGER_PATH", ledger_file):
        # For gemini-3.5-flash-lite, default is cfg.JUDGE_DAILY_REQUEST_BUDGET (450)
        with patch.object(limiter, "get_today_request_count", return_value=449):
            tokens, _ = limiter.acquire("Test prompt", model="gemini-3.5-flash-lite")
            assert tokens > 0

        with patch.object(limiter, "get_today_request_count", return_value=450):
            with pytest.raises(BudgetExceededError) as exc_info:
                limiter.acquire("Test prompt", model="gemini-3.5-flash-lite")
            assert "JUDGE_DAILY_REQUEST_BUDGET=450" in str(exc_info.value)

        # For gemma-4-26b-a4b-it, default is cfg.GEMMA_RPD_LIMIT (14400)
        with patch.object(limiter, "get_today_request_count", return_value=14400):
            with pytest.raises(BudgetExceededError) as exc_info:
                limiter.acquire("Test prompt", model="gemma-4-26b-a4b-it")
            assert "14400" in str(exc_info.value)


def test_rate_limiter_stricter_limit_chosen_when_custom_budget_higher(tmp_path):
    """Verify that if custom budget is higher than model cap, the stricter model cap is respected."""
    ledger_file = tmp_path / "judge_usage.jsonl"
    limiter = TokenAwareRateLimiter(daily_budget=1000)

    with patch("src.evaluation.judge.USAGE_LEDGER_PATH", ledger_file):
        # Model cap is 450. Since 450 < 1000, effective cap is 450.
        with patch.object(limiter, "get_today_request_count", return_value=450):
            with pytest.raises(BudgetExceededError) as exc_info:
                limiter.acquire("Test prompt", model="gemini-3.5-flash-lite")
            assert "JUDGE_DAILY_REQUEST_BUDGET=450" in str(exc_info.value)



def test_gemini_judge_schema_validation(tmp_path):
    """Verify schema parsing and validation into Pydantic model instance."""
    cache_file = tmp_path / "judge_cache.json"
    cache_file.write_text("{}", encoding="utf-8")

    with patch("src.evaluation.judge.CACHE_PATH", cache_file), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyFakeKey12345"}):
        with patch("google.genai.Client"):
            judge = GeminiJudge(model_name=cfg.JUDGE_MODEL_PRIMARY, metric="test_schema")
            
            valid_json = '{"verdict": "yes", "reason": "Fully supported"}'
            with patch.object(judge, "_call_api_with_retry", return_value=(valid_json, 15, 10)):
                res = judge.generate("Evaluate claim", schema=SampleSchema)
                assert isinstance(res, SampleSchema)
                assert res.verdict == "yes"
                assert res.reason == "Fully supported"


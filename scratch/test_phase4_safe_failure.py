"""
Phase 4: Safe Failure Behavior Verification without Paid Calls
Simulates:
1. Missing Gemini credentials
2. Gemini timeout
3. Invalid judge response
4. DeepEval metric exception
Verifies all fail safely (verified=False, status='not_verified') and never contact Gemini or Groq.
"""

import sys
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

sys.path.insert(0, ".")
import src.config as cfg
from src.evaluation.quality_gate import verify_generation_faithfulness
from src.evaluation.judge import GeminiJudge


def test_missing_gemini_credentials():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": ""}), \
         patch("google.genai.Client", side_effect=AssertionError("Live Gemini contacted!")):
        res = verify_generation_faithfulness(
            question="What is the section?",
            context_chunks=[{"text": "Section 12 applies."}],
            answer="Section 12 applies.",
        )
        assert res["verified"] is False, f"Expected verified=False, got {res['verified']}"
        assert res["status"] == "not_verified"
        assert res["score"] == 0.0
        assert "GEMINI_API_KEY missing" in res["reason"]
        print("  [OK] Missing credentials safely degraded to not_verified (verified=False)")


def test_gemini_timeout():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key"}), \
         patch("src.evaluation.quality_gate.GeminiJudge") as mock_judge_cls, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls:
        
        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        mock_f = MagicMock()
        mock_f.measure.side_effect = TimeoutError("Deadline exceeded: 504 Gateway Timeout")
        mock_f_cls.return_value = mock_f

        res = verify_generation_faithfulness(
            question="What is the section?",
            context_chunks=[{"text": "Section 12 applies."}],
            answer="Section 12 applies.",
        )
        assert res["verified"] is False, f"Expected verified=False, got {res['verified']}"
        assert res["status"] == "not_verified"
        assert res["score"] == 0.0
        assert "Deadline exceeded: 504 Gateway Timeout" in res["reason"]
        print("  [OK] Gemini timeout safely degraded to not_verified (verified=False)")


def test_invalid_judge_response():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key"}), \
         patch("src.evaluation.quality_gate.GeminiJudge") as mock_judge_cls, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls:
        
        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        mock_f = MagicMock()
        mock_f.measure.side_effect = ValueError("JSONDecodeError: Expecting value: line 1 column 1 (char 0)")
        mock_f_cls.return_value = mock_f

        res = verify_generation_faithfulness(
            question="What is the section?",
            context_chunks=[{"text": "Section 12 applies."}],
            answer="Section 12 applies.",
        )
        assert res["verified"] is False, f"Expected verified=False, got {res['verified']}"
        assert res["status"] == "not_verified"
        assert res["score"] == 0.0
        assert "JSONDecodeError" in res["reason"]
        print("  [OK] Invalid judge response safely degraded to not_verified (verified=False)")


def test_deepeval_metric_exception():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key"}), \
         patch("src.evaluation.quality_gate.GeminiJudge") as mock_judge_cls, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls:
        
        mock_judge = MagicMock()
        mock_judge_cls.return_value = mock_judge

        mock_f = MagicMock()
        mock_f.measure.side_effect = RuntimeError("DeepEval internal metric execution failed: schema mismatch")
        mock_f_cls.return_value = mock_f

        res = verify_generation_faithfulness(
            question="What is the section?",
            context_chunks=[{"text": "Section 12 applies."}],
            answer="Section 12 applies.",
        )
        assert res["verified"] is False, f"Expected verified=False, got {res['verified']}"
        assert res["status"] == "not_verified"
        assert res["score"] == 0.0
        assert "DeepEval internal metric execution failed" in res["reason"]
        print("  [OK] DeepEval metric exception safely degraded to not_verified (verified=False)")


if __name__ == "__main__":
    print("Running Phase 4 Safe Failure Behavior Tests:")
    test_missing_gemini_credentials()
    test_gemini_timeout()
    test_invalid_judge_response()
    test_deepeval_metric_exception()
    print("All Phase 4 Safe Failure Behavior Tests Passed Successfully!")

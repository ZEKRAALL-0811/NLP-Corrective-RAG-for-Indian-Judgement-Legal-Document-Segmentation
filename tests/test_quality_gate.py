"""
Unit tests for Online Quality Gate (tests/test_quality_gate.py).
Tests regeneration on failure, graceful degradation when offline, and pass/fail routing with judge mocked.
"""

import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")
from src.evaluation.quality_gate import (
    verify_generation_faithfulness,
    run_quality_gated_generation,
)



def test_quality_gate_disabled():
    with patch("src.config.ENABLE_ONLINE_GATE", False):
        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is True
        assert res["status"] == "gate_disabled"


def test_quality_gate_degrades_when_gemini_key_missing():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is False
        assert res["status"] == "not_verified"
        assert "GEMINI_API_KEY missing" in res["reason"]


def test_quality_gate_degrades_when_groq_key_missing():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "groq"), \
         patch.dict("os.environ", {"GROQ_API_KEY": ""}):
        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is False
        assert res["status"] == "not_verified"
        assert "GROQ_API_KEY missing" in res["reason"]


def test_quality_gate_degrades_when_unsupported_provider():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "unknown_provider"):
        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is False
        assert res["status"] == "not_verified"
        assert "Unsupported JUDGE_PROVIDER" in res["reason"]


def test_quality_gate_instantiates_configured_provider_judges():
    # 1. Gemini judge instantiation
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "fake_gemini_key"}), \
         patch("src.evaluation.quality_gate.GeminiJudge") as mock_gemini_cls, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
         patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
        
        mock_f = MagicMock()
        mock_f.is_successful.return_value = True
        mock_f.score = 1.0
        mock_f_cls.return_value = mock_f

        mock_r = MagicMock()
        mock_r.is_successful.return_value = True
        mock_r.score = 1.0
        mock_r_cls.return_value = mock_r

        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is True
        mock_gemini_cls.assert_called_once_with(metric="quality_gate")

    # 2. Groq judge instantiation
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "groq"), \
         patch.dict("os.environ", {"GROQ_API_KEY": "fake_groq_key"}), \
         patch("src.evaluation.quality_gate.GroqJudge") as mock_groq_cls, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
         patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:

        mock_f = MagicMock()
        mock_f.is_successful.return_value = True
        mock_f.score = 1.0
        mock_f_cls.return_value = mock_f

        mock_r = MagicMock()
        mock_r.is_successful.return_value = True
        mock_r.score = 1.0
        mock_r_cls.return_value = mock_r

        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is True
        mock_groq_cls.assert_called_once_with(metric="quality_gate")


def test_quality_gate_abstention_skips_judge():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch.dict("os.environ", {"GROQ_API_KEY": "dummy_key"}):
        res = verify_generation_faithfulness(
            question="What is the penalty?",
            context_chunks=[{"text": "Some text"}],
            answer="The retrieved parts of this judgment do not answer this question.",
        )
        assert res["verified"] is True
        assert res["status"] == "abstained"


def test_quality_gate_pass_with_mocked_judge():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch.dict("os.environ", {"GROQ_API_KEY": "dummy_key"}):
        
        mock_judge = MagicMock()
        with patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
             patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
            mock_f = MagicMock()
            mock_f.score = 1.0
            mock_f.is_successful.return_value = True
            mock_f.reason = "Faithful to context"
            mock_f_cls.return_value = mock_f

            mock_r = MagicMock()
            mock_r.score = 0.95
            mock_r.is_successful.return_value = True
            mock_r.reason = "Highly relevant to question"
            mock_r_cls.return_value = mock_r
            
            res = verify_generation_faithfulness(
                question="When did the sale occur?",
                context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
                answer="The sale occurred on 01.07.2021.",
                judge=mock_judge,
                threshold=0.7,
            )
            assert res["verified"] is True
            assert res["status"] == "passed"
            assert res["score"] == 1.0
            assert res["faithfulness_score"] == 1.0
            assert res["relevancy_score"] == 0.95
            assert "Faithfulness: Faithful to context" in res["reason"]
            assert "Relevancy: Highly relevant to question" in res["reason"]


def test_quality_gate_fails_when_faithfulness_fails_even_if_relevancy_passes():
    with patch("src.config.ENABLE_ONLINE_GATE", True):
        mock_judge = MagicMock()
        with patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
             patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
            mock_f = MagicMock()
            mock_f.score = 0.3
            mock_f.is_successful.return_value = False
            mock_f.reason = "Hallucinated claim"
            mock_f_cls.return_value = mock_f

            mock_r = MagicMock()
            mock_r.score = 0.9
            mock_r.is_successful.return_value = True
            mock_r.reason = "Relevant"
            mock_r_cls.return_value = mock_r

            res = verify_generation_faithfulness(
                question="When did the sale occur?",
                context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
                answer="The sale occurred in 1999.",
                judge=mock_judge,
            )
            assert res["verified"] is False
            assert res["status"] == "failed"
            assert res["faithfulness_score"] == 0.3
            assert res["relevancy_score"] == 0.9


def test_quality_gate_fails_when_relevancy_fails_even_if_faithfulness_passes():
    with patch("src.config.ENABLE_ONLINE_GATE", True):
        mock_judge = MagicMock()
        with patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
             patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
            mock_f = MagicMock()
            mock_f.score = 1.0
            mock_f.is_successful.return_value = True
            mock_f.reason = "Fully grounded"
            mock_f_cls.return_value = mock_f

            mock_r = MagicMock()
            mock_r.score = 0.4
            mock_r.is_successful.return_value = False
            mock_r.reason = "Off-topic response"
            mock_r_cls.return_value = mock_r

            res = verify_generation_faithfulness(
                question="When did the sale occur?",
                context_chunks=[{"text": "Sale occurred on 01.07.2021. The judge was Justice Rao."}],
                answer="Justice Rao was the presiding judge.",
                judge=mock_judge,
            )
            assert res["verified"] is False
            assert res["status"] == "failed"
            assert res["faithfulness_score"] == 1.0
            assert res["relevancy_score"] == 0.4


def test_quality_gated_generation_regenerates_on_failure():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch.dict("os.environ", {"GROQ_API_KEY": "dummy_key"}):
        
        mock_judge = MagicMock()
        # Mock generate_grounded_answer
        gen_outputs = [
            {"answer": "Initial unfaithful answer", "citations": [1], "status": "ready"},
            {"answer": "Second faithful answer", "citations": [1], "status": "ready"},
        ]
        
        with patch("src.evaluation.quality_gate.generate_grounded_answer", side_effect=gen_outputs), \
             patch("src.evaluation.quality_gate.verify_generation_faithfulness") as mock_verify:
            
            # First verification fails, second verification passes
            mock_verify.side_effect = [
                {"verified": False, "score": 0.2, "status": "failed", "reason": "Contradiction"},
                {"verified": True, "score": 1.0, "status": "passed", "reason": "Faithful"},
            ]
            
            res = run_quality_gated_generation(
                question="Test question",
                context_chunks=[{"text": "Context passage"}],
                judge=mock_judge,
                max_regenerations=1,
            )
            
            assert res["verification_status"] == "verified"
            assert res["faithfulness_score"] == 1.0
            assert res["flag_unverified"] is False
            assert res["answer"] == "Second faithful answer"


def test_quality_gated_generation_flags_after_persistent_failure():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch.dict("os.environ", {"GROQ_API_KEY": "dummy_key"}):
        
        mock_judge = MagicMock()
        gen_outputs = [
            {"answer": "Unfaithful 1", "citations": [1], "status": "ready"},
            {"answer": "Unfaithful 2", "citations": [1], "status": "ready"},
        ]
        
        with patch("src.evaluation.quality_gate.generate_grounded_answer", side_effect=gen_outputs), \
             patch("src.evaluation.quality_gate.verify_generation_faithfulness") as mock_verify:
            
            # Both attempts fail
            mock_verify.side_effect = [
                {"verified": False, "score": 0.2, "status": "failed", "reason": "Contradiction 1"},
                {"verified": False, "score": 0.3, "status": "failed", "reason": "Contradiction 2"},
            ]
            
            res = run_quality_gated_generation(
                question="Test question",
                context_chunks=[{"text": "Context passage"}],
                judge=mock_judge,
                max_regenerations=1,
            )
            
            assert res["verification_status"] == "could not be fully verified"
            assert res["flag_unverified"] is True


def test_quality_gate_fails_when_both_metrics_fail():
    with patch("src.config.ENABLE_ONLINE_GATE", True):
        mock_judge = MagicMock()
        with patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
             patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
            mock_f = MagicMock()
            mock_f.score = 0.2
            mock_f.is_successful.return_value = False
            mock_f.reason = "Unfaithful to context"
            mock_f_cls.return_value = mock_f

            mock_r = MagicMock()
            mock_r.score = 0.3
            mock_r.is_successful.return_value = False
            mock_r.reason = "Completely off-topic"
            mock_r_cls.return_value = mock_r

            res = verify_generation_faithfulness(
                question="When did the sale occur?",
                context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
                answer="The weather is sunny in Paris.",
                judge=mock_judge,
            )
            assert res["verified"] is False
            assert res["status"] == "failed"
            assert res["faithfulness_score"] == 0.2
            assert res["relevancy_score"] == 0.3


def test_quality_gate_degrades_on_judge_exception():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_key"}):
        mock_judge = MagicMock()
        with patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls:
            mock_f = MagicMock()
            mock_f.measure.side_effect = RuntimeError("Simulated Gemini API connection timeout")
            mock_f_cls.return_value = mock_f

            res = verify_generation_faithfulness(
                question="When did the sale occur?",
                context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
                answer="The sale occurred on 01.07.2021.",
                judge=mock_judge,
            )
            assert res["verified"] is False
            assert res["status"] == "not_verified"
            assert res["score"] == 0.0
            assert "Simulated Gemini API connection timeout" in res["reason"]


def test_quality_gate_gemini_only_enforcement():
    with patch("src.config.ENABLE_ONLINE_GATE", True), \
         patch("src.config.JUDGE_PROVIDER", "gemini"), \
         patch.dict("os.environ", {"GEMINI_API_KEY": "valid_gemini_key"}), \
         patch("src.evaluation.quality_gate.GeminiJudge") as mock_gemini, \
         patch("src.evaluation.quality_gate.GroqJudge") as mock_groq, \
         patch("src.evaluation.quality_gate.FaithfulnessMetric") as mock_f_cls, \
         patch("src.evaluation.quality_gate.AnswerRelevancyMetric") as mock_r_cls:
        
        mock_f = MagicMock()
        mock_f.is_successful.return_value = True
        mock_f.score = 1.0
        mock_f_cls.return_value = mock_f

        mock_r = MagicMock()
        mock_r.is_successful.return_value = True
        mock_r.score = 1.0
        mock_r_cls.return_value = mock_r

        res = verify_generation_faithfulness(
            question="When did the sale occur?",
            context_chunks=[{"text": "Sale occurred on 01.07.2021."}],
            answer="The sale occurred on 01.07.2021.",
        )
        assert res["verified"] is True
        assert mock_gemini.call_count == 1
        assert mock_groq.call_count == 0

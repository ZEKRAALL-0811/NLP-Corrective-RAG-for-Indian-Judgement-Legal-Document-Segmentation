"""
Module 10: Online Quality Gate (src/evaluation/quality_gate.py)

Per PRD:
- After generation, if ENABLE_ONLINE_GATE is True and GROQ_API_KEY is present:
  runs a FaithfulnessMetric check on the generated answer against the context.
- On failure (score < threshold), regenerates once.
- If it fails again, attaches flag 'could not be fully verified'.
- Defaults to OFF (ENABLE_ONLINE_GATE = False).
- Degrades gracefully to 'not verified' if offline, key missing, or judge unavailable.
"""

import os
from typing import List, Dict, Any, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM

import src.config as cfg
from src.rag.generator import generate_grounded_answer, check_abstention_phrasing, ABSTENTION_TEXT
from src.evaluation.judge import GeminiJudge, GroqJudge


def verify_generation_faithfulness(
    question: str,
    context_chunks: List[Dict[str, Any]],
    answer: str,
    judge: Optional[DeepEvalBaseLLM] = None,
    threshold: float = 0.7,
    relevancy_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates faithfulness and answer relevancy using DeepEval FaithfulnessMetric and AnswerRelevancyMetric.
    Respects configured JUDGE_PROVIDER ('gemini' or 'groq').
    Degrades to 'not_verified' if offline, credentials missing, or unsupported provider.
    """
    if not cfg.ENABLE_ONLINE_GATE:
        return {
            "verified": True,
            "score": 1.0,
            "faithfulness_score": 1.0,
            "relevancy_score": 1.0,
            "status": "gate_disabled",
            "reason": "Online quality gate is disabled in configuration.",
        }

    provider = getattr(cfg, "JUDGE_PROVIDER", "gemini").lower().strip()

    # Credential & provider checks if judge is not explicitly provided
    if judge is None:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                return {
                    "verified": False,
                    "score": 0.0,
                    "faithfulness_score": 0.0,
                    "relevancy_score": 0.0,
                    "status": "not_verified",
                    "reason": "GEMINI_API_KEY missing; degraded to unverified.",
                }
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                return {
                    "verified": False,
                    "score": 0.0,
                    "faithfulness_score": 0.0,
                    "relevancy_score": 0.0,
                    "status": "not_verified",
                    "reason": "GROQ_API_KEY missing; degraded to unverified.",
                }
        else:
            return {
                "verified": False,
                "score": 0.0,
                "faithfulness_score": 0.0,
                "relevancy_score": 0.0,
                "status": "not_verified",
                "reason": f"Unsupported JUDGE_PROVIDER '{provider}'; degraded to unverified.",
            }

    # Abstentions pass automatically without LLM verification
    if not answer or answer == ABSTENTION_TEXT or check_abstention_phrasing(answer):
        return {
            "verified": True,
            "score": 1.0,
            "faithfulness_score": 1.0,
            "relevancy_score": 1.0,
            "status": "abstained",
            "reason": "Abstention response; verification skipped.",
        }

    context_texts = [c.get("text", "") for c in context_chunks if c.get("text")]
    if not context_texts:
        return {
            "verified": False,
            "score": 0.0,
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
            "status": "not_verified",
            "reason": "Empty context; cannot verify faithfulness.",
        }

    try:
        if judge is not None:
            eval_judge = judge
        elif provider == "gemini":
            eval_judge = GeminiJudge(metric="quality_gate")
        elif provider == "groq":
            eval_judge = GroqJudge(metric="quality_gate")
        else:
            return {
                "verified": False,
                "score": 0.0,
                "faithfulness_score": 0.0,
                "relevancy_score": 0.0,
                "status": "not_verified",
                "reason": f"Unsupported JUDGE_PROVIDER '{provider}'; degraded to unverified.",
            }

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context_texts,
        )
        faith_metric = FaithfulnessMetric(
            threshold=threshold,
            model=eval_judge,
            verbose_mode=False,
            async_mode=False,
        )
        faith_metric.measure(test_case)

        rel_thresh = relevancy_threshold if relevancy_threshold is not None else threshold
        rel_metric = AnswerRelevancyMetric(
            threshold=rel_thresh,
            model=eval_judge,
            verbose_mode=False,
            async_mode=False,
        )
        rel_metric.measure(test_case)

        is_passed = bool(faith_metric.is_successful() and rel_metric.is_successful())
        f_score = float(faith_metric.score) if faith_metric.score is not None else 0.0
        r_score = float(rel_metric.score) if rel_metric.score is not None else 0.0

        reasons = []
        f_reason = getattr(faith_metric, "reason", "")
        r_reason = getattr(rel_metric, "reason", "")
        if f_reason:
            reasons.append(f"Faithfulness: {f_reason}")
        if r_reason:
            reasons.append(f"Relevancy: {r_reason}")

        return {
            "verified": is_passed,
            "score": f_score,
            "faithfulness_score": f_score,
            "relevancy_score": r_score,
            "status": "passed" if is_passed else "failed",
            "reason": "; ".join(reasons),
        }
    except Exception as e:
        # Graceful degradation if offline or error occurs
        return {
            "verified": False,
            "score": 0.0,
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
            "status": "not_verified",
            "reason": f"Quality gate evaluation error: {str(e)}",
        }


def run_quality_gated_generation(
    question: str,
    context_chunks: List[Dict[str, Any]],
    fallback_chunks: Optional[List[Dict[str, Any]]] = None,
    is_web_fallback: bool = False,
    judge: Optional[DeepEvalBaseLLM] = None,
    max_regenerations: int = 1,
    threshold: float = 0.7,
    relevancy_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generates answer and subjects it to online faithfulness and relevancy quality gate if enabled.
    Regenerates once on failure, then flags as 'could not be fully verified'.
    """
    # 1. Initial generation
    gen_res = generate_grounded_answer(
        question=question,
        verified_chunks=context_chunks,
        fallback_chunks=fallback_chunks,
        is_web_fallback=is_web_fallback,
    )

    if not cfg.ENABLE_ONLINE_GATE:
        gen_res["verification_status"] = "gate_disabled"
        gen_res["flag_unverified"] = False
        return gen_res

    # 2. First verification
    ver_res = verify_generation_faithfulness(
        question=question,
        context_chunks=context_chunks,
        answer=gen_res["answer"],
        judge=judge,
        threshold=threshold,
        relevancy_threshold=relevancy_threshold,
    )

    if ver_res["status"] == "passed" or ver_res["status"] == "abstained":
        gen_res["verification_status"] = "verified"
        gen_res["faithfulness_score"] = ver_res.get("faithfulness_score", ver_res["score"])
        gen_res["relevancy_score"] = ver_res.get("relevancy_score", 1.0)
        gen_res["flag_unverified"] = False
        return gen_res

    if ver_res["status"] == "not_verified":
        gen_res["verification_status"] = "not_verified"
        gen_res["faithfulness_score"] = ver_res.get("faithfulness_score", 0.0)
        gen_res["relevancy_score"] = ver_res.get("relevancy_score", 0.0)
        gen_res["flag_unverified"] = True
        return gen_res

    # 3. Failed -> Regenerate once
    if max_regenerations > 0 and ver_res["status"] == "failed":
        regen_res = generate_grounded_answer(
            question=question,
            verified_chunks=context_chunks,
            fallback_chunks=fallback_chunks,
            is_web_fallback=is_web_fallback,
        )
        ver_res_2 = verify_generation_faithfulness(
            question=question,
            context_chunks=context_chunks,
            answer=regen_res["answer"],
            judge=judge,
            threshold=threshold,
            relevancy_threshold=relevancy_threshold,
        )

        if ver_res_2["status"] == "passed":
            regen_res["verification_status"] = "verified"
            regen_res["faithfulness_score"] = ver_res_2.get("faithfulness_score", ver_res_2["score"])
            regen_res["relevancy_score"] = ver_res_2.get("relevancy_score", 1.0)
            regen_res["flag_unverified"] = False
            return regen_res
        else:
            regen_res["verification_status"] = "could not be fully verified"
            regen_res["faithfulness_score"] = ver_res_2.get("faithfulness_score", ver_res_2["score"])
            regen_res["relevancy_score"] = ver_res_2.get("relevancy_score", 0.0)
            regen_res["flag_unverified"] = True
            return regen_res

    gen_res["verification_status"] = "could not be fully verified"
    gen_res["faithfulness_score"] = ver_res.get("faithfulness_score", ver_res["score"])
    gen_res["relevancy_score"] = ver_res.get("relevancy_score", 0.0)
    gen_res["flag_unverified"] = True
    return gen_res

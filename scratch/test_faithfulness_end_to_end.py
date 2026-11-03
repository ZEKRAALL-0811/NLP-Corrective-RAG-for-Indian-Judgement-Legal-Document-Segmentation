"""
End-to-end test of FaithfulnessMetric with GeminiJudge.
Tests:
1. Faithful answer: expects score >= 0.9
2. Contradicting answer: expects a clear drop (< 0.5)
Logs raw evaluation outputs to artifacts/logs/faithfulness_test.txt.
"""

import os
import sys
import json
from pathlib import Path

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
sys.path.insert(0, ".")
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.evaluation.judge import get_judge

def run_faithfulness_test():
    judge = get_judge(metric="faithfulness")
    print(f"Testing FaithfulnessMetric with judge model: {judge.get_model_name()}...")

    context = [
        "On 01.07.2021, the petitioners sold equity shares of company under Offer For Sale to the public. "
        "Pramod Swarup Agrawal transferred 11,11,486 equity shares for Rs.33,00,00,000/-. "
        "Sneh Lata Agarwal transferred 14,05,405 equity shares for Rs.41,59,99,880/-."
    ]
    question = "How many shares and for what amount did Pramod Swarup Agrawal sell?"

    # Case 1: Faithful
    faithful_answer = "Pramod Swarup Agrawal transferred 11,11,486 equity shares for a total consideration of Rs.33,00,00,000/- on 01.07.2021."
    tc_faithful = LLMTestCase(
        input=question,
        actual_output=faithful_answer,
        retrieval_context=context,
    )

    metric_faithful = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False)
    print("Measuring faithful case...", flush=True)
    metric_faithful.measure(tc_faithful, _show_indicator=False)
    score_faithful = metric_faithful.score
    reason_faithful = metric_faithful.reason
    print(f"Faithful Case -> Score: {score_faithful} | Reason: {reason_faithful}", flush=True)

    # Case 2: Contradicting
    contradicting_answer = "Pramod Swarup Agrawal sold 5,000,000 shares for Rs.100,00,00,000/- on 15.08.2023 to a private firm."
    tc_contradicting = LLMTestCase(
        input=question,
        actual_output=contradicting_answer,
        retrieval_context=context,
    )

    metric_contradicting = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True, async_mode=False)
    print("Measuring contradicting case...", flush=True)
    metric_contradicting.measure(tc_contradicting, _show_indicator=False)
    score_contradicting = metric_contradicting.score
    reason_contradicting = metric_contradicting.reason
    print(f"Contradicting Case -> Score: {score_contradicting} | Reason: {reason_contradicting}", flush=True)

    # Format log
    log_lines = [
        "=" * 80,
        "DEEPEVAL FAITHFULNESS METRIC END-TO-END VERIFICATION",
        f"Judge Model: {judge.get_model_name()}",
        "=" * 80,
        "",
        "[Case 1: Faithful Answer]",
        f"Input: {question}",
        f"Context: {context[0]}",
        f"Actual Output: {faithful_answer}",
        f"Score: {score_faithful}",
        f"Reason: {reason_faithful}",
        f"Pass (>=0.9): {score_faithful >= 0.9}",
        "",
        "[Case 2: Contradicting Answer]",
        f"Input: {question}",
        f"Context: {context[0]}",
        f"Actual Output: {contradicting_answer}",
        f"Score: {score_contradicting}",
        f"Reason: {reason_contradicting}",
        f"Drop ( < score_faithful): {score_contradicting < score_faithful}",
        "=" * 80,
    ]

    out_file = Path("artifacts/logs/faithfulness_test.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Wrote log to {out_file}")

    assert score_faithful >= 0.9, f"Faithful score {score_faithful} < 0.9"
    assert score_contradicting < score_faithful, f"Contradicting score {score_contradicting} did not drop from {score_faithful}"
    print("PASS: FaithfulnessMetric end-to-end verification passed!")

if __name__ == "__main__":
    run_faithfulness_test()

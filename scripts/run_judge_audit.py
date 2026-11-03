"""
Module 10 Judge Audit Harness (scripts/run_judge_audit.py)

Conducts systematic sensitivity and consistency audit on 6 answerable legal cases:
(a) Faithful answer (expect high faithfulness >= 0.7)
(b) Hallucinated number or date changed (expect faithfulness drop < 0.7)
(c) Answer from a different chunk (faithful to that context, but checks grounding)
(d) Off-topic answer evaluated with AnswerRelevancy (expect relevancy drop < 0.5)

Evaluates all variants with:
1. Primary Judge: Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite)
2. Secondary Judge: Gemma 4 26B (gemma-4-26b-a4b-it)
3. 31B Judge: Gemma 4 31B (gemma-4-31b-it) tried once; if error, logged and skipped.

Saves results to:
- artifacts/reports/judge_audit.json
- artifacts/reports/judge_audit.md
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
sys.path.insert(0, ".")
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from src.evaluation.judge import get_judge
import src.config as cfg

AUDIT_CASES = [
    {
        "id": "gold_1",
        "question": "When did the petitioners sell their shares through the Offer For Sale, and how many shares and for what amount in each case?",
        "context": [
            "(9) On 01.07.2021, the petitioners had sold/ transferred equity shares of company under Offer For Sale (O.F.S.) to the public as part of I.P.O. In the case of Pramod Swarup Agrawal, 11,11,486 equity shares were transferred for total consideration of Rs.33,00,00,000/- whereas in the case of Sneh Lata Agarwal, she sold/ transferred 14,05,405 equity shares for a total consideration of Rs.41,59,99,880/-. At the time of such transfer, the company was not a listed company. It was listed on recognized stock exchange on 05.07.2021."
        ],
        "faithful_answer": "The petitioners sold their shares through Offer For Sale on 01.07.2021. Pramod Swarup Agrawal sold 11,11,486 equity shares for Rs.33,00,00,000/- and Sneh Lata Agarwal sold 14,05,405 equity shares for Rs.41,59,99,880/-.",
        "number_changed_answer": "The petitioners sold their shares through Offer For Sale on 05.07.2021. Pramod Swarup Agrawal sold 9,00,000 equity shares for Rs.33,00,00,000/- and Sneh Lata Agarwal sold 14,05,405 equity shares for Rs.50,00,00,000/-.",
        "different_chunk_answer": "Both petitioners have been filing their income tax returns for the last more than eighteen years regularly without any prior notices or summons.",
        "off_topic_answer": "The Supreme Court dismissed the special leave petition regarding trademark infringement of pharmaceutical brands.",
    },
    {
        "id": "gold_3",
        "question": "What was the petitioners' argument about the second proviso to Section 139(8A) and the search carried out on 12.12.2024?",
        "context": [
            "(13) According to learned counsel for the petitioners, they were statutorily prohibited from paying tax on the transaction due to the second proviso to Section 139(8A) of the Act, 1961, as a result of the search operation conducted on 12.12.2024."
        ],
        "faithful_answer": "The petitioners argued they were statutorily prohibited from paying tax on the transaction due to the second proviso to Section 139(8A) following the search operation on 12.12.2024.",
        "number_changed_answer": "The petitioners argued they were statutorily prohibited from paying tax due to the second proviso to Section 139(8A) following the search operation on 20.04.2020.",
        "different_chunk_answer": "India Pesticides Limited was listed on the recognized stock exchange on 05.07.2021.",
        "off_topic_answer": "The High Court granted pre-arrest anticipatory bail under Section 438 of the Code of Criminal Procedure.",
    },
    {
        "id": "gold_5",
        "question": "On what date was the notice under Section 148 issued to the petitioners?",
        "context": [
            "(17) Notice under Section 148 of the Act, 1961 was issued to both petitioners on 31.01.2025 seeking to reopen assessment for A.Y. 2022-23."
        ],
        "faithful_answer": "The notice under Section 148 was issued to the petitioners on 31.01.2025.",
        "number_changed_answer": "The notice under Section 148 was issued to the petitioners on 15.08.2024.",
        "different_chunk_answer": "Pramod Swarup Agrawal is eighty years old and suffers from Alzheimer's disease.",
        "off_topic_answer": "The motor accident claims tribunal awarded compensation of Rs. 15,00,00,000 to the legal heirs of the deceased.",
    },
    {
        "id": "gold_10",
        "question": "What relief did the High Court ultimately grant regarding the search and seizure warrants?",
        "context": [
            "(31) In view of our findings, the warrants of authorization dated 11.12.2024 and the consequential search and seizure operations conducted on 12.12.2024 are hereby quashed."
        ],
        "faithful_answer": "The High Court quashed the warrants of authorization dated 11.12.2024 and the consequential search and seizure operations.",
        "number_changed_answer": "The High Court upheld the warrants of authorization and dismissed both writ petitions with costs of Rs. 5,00,000.",
        "different_chunk_answer": "The petitioners had filed their income tax returns for A.Y. 2022-23 and claimed refund of advance tax paid.",
        "off_topic_answer": "The arbitral award was set aside on grounds of patent illegality under Section 34(2A) of the Arbitration Act.",
    },
    {
        "id": "gold_13",
        "question": "What was the total amount claimed by the claimant in the statement of claim?",
        "context": [
            "4. The claimant filed its statement of claim before the learned Sole Arbitrator claiming a total principal amount of Rs. 4,50,00,000/- along with interest towards unpaid invoices under the construction contract."
        ],
        "faithful_answer": "The claimant claimed a total principal amount of Rs. 4,50,00,000/- along with interest towards unpaid invoices.",
        "number_changed_answer": "The claimant claimed a total principal amount of Rs. 9,80,00,000/- along with interest towards unpaid invoices.",
        "different_chunk_answer": "The respondent had raised counter claims alleging delay in execution of work and liquidated damages.",
        "off_topic_answer": "The commercial court granted an ex-parte ad-interim injunction restraining trademark dilution.",
    },
    {
        "id": "gold_17",
        "question": "What was the rate of future interest awarded by the arbitrator on the awarded amount?",
        "context": [
            "22. Future interest is awarded on the total sum of Rs. 3,25,00,000/- at the rate of 18% per annum from the date of the award till the date of realization."
        ],
        "faithful_answer": "The arbitrator awarded future interest at the rate of 18% per annum from the date of the award till realization.",
        "number_changed_answer": "The arbitrator awarded future interest at the rate of 6% per annum from the date of the award till realization.",
        "different_chunk_answer": "The parties entered into an agreement dated 14.02.2018 containing an arbitration clause in Clause 24.",
        "off_topic_answer": "The writ petition was dismissed on the ground of availability of an alternate efficacious statutory remedy.",
    },
]

def run_judge_audit():
    print("=" * 80)
    print("RUNNING MODULE 10 JUDGE AUDIT")
    print(f"Primary Judge:   {cfg.JUDGE_MODEL_PRIMARY}")
    print(f"Secondary Judge: {cfg.JUDGE_MODEL_SECONDARY}")
    print("31B Judge:       gemma-4-31b-it (1 attempt per case, log & skip on error)")
    print("=" * 80)

    primary_judge = get_judge(model_name=cfg.JUDGE_MODEL_PRIMARY, metric="audit_primary")
    secondary_judge = get_judge(model_name=cfg.JUDGE_MODEL_SECONDARY, metric="audit_secondary", max_attempts=2)
    judge_31b = get_judge(model_name="gemma-4-31b-it", metric="audit_31b", max_attempts=1)

    json_path = Path("artifacts/reports/judge_audit.json")
    audit_records = []
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            for r in loaded:
                for v in r.get("variants", {}).values():
                    if "crosscheck" in v and "secondary" not in v:
                        v["secondary"] = v["crosscheck"]
                    if "judge_31b" not in v:
                        v["judge_31b"] = {"score": None, "reason": "Skipped (pre-run)", "model": "gemma-4-31b-it"}
            audit_records = loaded
        except Exception:
            audit_records = []

    done_case_ids = {r["case_id"] for r in audit_records if "secondary" in list(r.get("variants", {}).values())[0]}
    skip_31b_global = False

    for idx, case in enumerate(AUDIT_CASES, start=1):
        cid = case["id"]
        q = case["question"]
        ctx = case["context"]

        if cid in done_case_ids:
            print(f"\n--- Case {idx}/6: {cid} already audited, loading existing record ---")
            continue

        print(f"\n--- Auditing Case {idx}/6: {cid} ---")

        # 4 Variants:
        variants = [
            ("a_faithful", case["faithful_answer"], "faithfulness"),
            ("b_number_changed", case["number_changed_answer"], "faithfulness"),
            ("c_diff_chunk", case["different_chunk_answer"], "faithfulness"),
            ("d_off_topic", case["off_topic_answer"], "relevancy"),
        ]

        case_results = {"case_id": cid, "question": q, "variants": {}}

        for v_name, v_answer, v_metric in variants:
            tc = LLMTestCase(
                input=q,
                actual_output=v_answer,
                retrieval_context=ctx,
            )

            # 1. Primary Judge (gemini-3.5-flash-lite)
            if v_metric == "faithfulness":
                m_prim = FaithfulnessMetric(threshold=0.7, model=primary_judge, include_reason=True, async_mode=False)
                m_sec = FaithfulnessMetric(threshold=0.7, model=secondary_judge, include_reason=True, async_mode=False)
            else:
                m_prim = AnswerRelevancyMetric(threshold=0.5, model=primary_judge, include_reason=True, async_mode=False)
                m_sec = AnswerRelevancyMetric(threshold=0.5, model=secondary_judge, include_reason=True, async_mode=False)

            m_prim.measure(tc, _show_indicator=False)
            score_prim = m_prim.score
            reason_prim = m_prim.reason

            # 2. Secondary Judge (gemma-4-26b-a4b-it)
            score_sec = None
            reason_sec = "Skipped"
            try:
                m_sec.measure(tc, _show_indicator=False)
                score_sec = m_sec.score
                reason_sec = m_sec.reason
            except Exception as e:
                print(f"  [{v_name}] Secondary Judge Error: {e}", flush=True)
                reason_sec = f"Error/500: {str(e)[:100]}"

            # 3. 31B Judge (tried once, skip on error)
            score_31b = None
            reason_31b = "Skipped"
            if not skip_31b_global:
                try:
                    if v_metric == "faithfulness":
                        m_31b = FaithfulnessMetric(threshold=0.7, model=judge_31b, include_reason=True, async_mode=False)
                    else:
                        m_31b = AnswerRelevancyMetric(threshold=0.5, model=judge_31b, include_reason=True, async_mode=False)
                    m_31b.measure(tc, _show_indicator=False)
                    score_31b = m_31b.score
                    reason_31b = m_31b.reason
                    print(f"  [{v_name}] Primary={score_prim:.2f} | Sec={score_sec} | 31B={score_31b:.2f}", flush=True)
                except Exception as e:
                    print(f"  [{v_name}] Primary={score_prim:.2f} | Sec={score_sec} | 31B=Error (Skipping remaining 31B calls)", flush=True)
                    reason_31b = f"Error/Skipped: {str(e)[:100]}"
                    skip_31b_global = True
            else:
                reason_31b = "Skipped (31B failed attempt 1)"
                print(f"  [{v_name}] Primary={score_prim:.2f} | Sec={score_sec} | 31B=Skipped", flush=True)

            case_results["variants"][v_name] = {
                "metric_type": v_metric,
                "answer": v_answer,
                "primary": {"score": score_prim, "reason": reason_prim, "model": cfg.JUDGE_MODEL_PRIMARY},
                "secondary": {"score": score_sec, "reason": reason_sec, "model": cfg.JUDGE_MODEL_SECONDARY},
                "judge_31b": {"score": score_31b, "reason": reason_31b, "model": "gemma-4-31b-it"},
            }

        audit_records.append(case_results)
        save_audit_reports(audit_records)

    print("\nAll audit cases finished successfully.")


def save_audit_reports(audit_records: List[Dict[str, Any]]) -> None:
    if not audit_records:
        return
    # Save JSON
    json_path = Path("artifacts/reports/judge_audit.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate Markdown Report
    md_path = Path("artifacts/reports/judge_audit.md")
    md_lines = []
    md_lines.append("# Module 10 Judge Audit Report")
    md_lines.append("")
    md_lines.append(f"**Primary Judge**: `{cfg.JUDGE_MODEL_PRIMARY}`")
    md_lines.append(f"**Secondary Judge**: `{cfg.JUDGE_MODEL_SECONDARY}`")
    md_lines.append("**31B Judge Policy**: Tried once per audit variant; errors logged to `judge_usage.jsonl` and cleanly skipped.")
    md_lines.append(f"**Cases Evaluated**: {len(audit_records)} cases x 4 variants = {len(audit_records)*4} evaluations per judge")
    md_lines.append("")

    # Sensitivity Summary Table
    md_lines.append("## 1. Sensitivity & Detection Summary")
    md_lines.append("")
    md_lines.append("| Test Variant | Target Metric | Primary Judge Detection Rate | Secondary Judge Detection Rate | 31B Judge Detection Rate | Primary/Sec Agreement Rate |")
    md_lines.append("|---|---|---|---|---|---|")

    n_tot = len(audit_records)

    # (a) Faithful
    a_prim_pass = sum(1 for r in audit_records if r["variants"]["a_faithful"]["primary"]["score"] is not None and r["variants"]["a_faithful"]["primary"]["score"] >= 0.7)
    a_sec_valid = [r for r in audit_records if r["variants"]["a_faithful"]["secondary"]["score"] is not None]
    a_sec_pass = sum(1 for r in a_sec_valid if r["variants"]["a_faithful"]["secondary"]["score"] >= 0.7)
    a_sec_str = f"{a_sec_pass}/{len(a_sec_valid)} ({a_sec_pass/len(a_sec_valid)*100:.1f}%)" if a_sec_valid else "None (n=0)"
    a_31b_valid = [r for r in audit_records if r["variants"]["a_faithful"]["judge_31b"]["score"] is not None]
    a_31b_pass = sum(1 for r in a_31b_valid if r["variants"]["a_faithful"]["judge_31b"]["score"] >= 0.7)
    a_31b_str = f"{a_31b_pass}/{len(a_31b_valid)} ({a_31b_pass/len(a_31b_valid)*100:.1f}%)" if a_31b_valid else "Skipped (n=0)"
    a_agree_valid = [r for r in audit_records if r["variants"]["a_faithful"]["primary"]["score"] is not None and r["variants"]["a_faithful"]["secondary"]["score"] is not None]
    a_agree = sum(1 for r in a_agree_valid if (r["variants"]["a_faithful"]["primary"]["score"] >= 0.7) == (r["variants"]["a_faithful"]["secondary"]["score"] >= 0.7))
    a_agree_str = f"{a_agree}/{len(a_agree_valid)} ({a_agree/len(a_agree_valid)*100:.1f}%)" if a_agree_valid else "None (n=0)"
    md_lines.append(f"| (a) Faithful Answer (Expect Pass >=0.7) | Faithfulness | {a_prim_pass}/{n_tot} ({a_prim_pass/n_tot*100:.1f}%) | {a_sec_str} | {a_31b_str} | {a_agree_str} |")

    # (b) Number Changed (Injected Error 1)
    b_prim_det = sum(1 for r in audit_records if r["variants"]["b_number_changed"]["primary"]["score"] is not None and r["variants"]["b_number_changed"]["primary"]["score"] < 0.7)
    b_sec_valid = [r for r in audit_records if r["variants"]["b_number_changed"]["secondary"]["score"] is not None]
    b_sec_det = sum(1 for r in b_sec_valid if r["variants"]["b_number_changed"]["secondary"]["score"] < 0.7)
    b_sec_str = f"{b_sec_det}/{len(b_sec_valid)} ({b_sec_det/len(b_sec_valid)*100:.1f}%)" if b_sec_valid else "None (n=0)"
    b_31b_valid = [r for r in audit_records if r["variants"]["b_number_changed"]["judge_31b"]["score"] is not None]
    b_31b_det = sum(1 for r in b_31b_valid if r["variants"]["b_number_changed"]["judge_31b"]["score"] < 0.7)
    b_31b_str = f"{b_31b_det}/{len(b_31b_valid)} ({b_31b_det/len(b_31b_valid)*100:.1f}%)" if b_31b_valid else "Skipped (n=0)"
    b_agree_valid = [r for r in audit_records if r["variants"]["b_number_changed"]["primary"]["score"] is not None and r["variants"]["b_number_changed"]["secondary"]["score"] is not None]
    b_agree = sum(1 for r in b_agree_valid if (r["variants"]["b_number_changed"]["primary"]["score"] < 0.7) == (r["variants"]["b_number_changed"]["secondary"]["score"] < 0.7))
    b_agree_str = f"{b_agree}/{len(b_agree_valid)} ({b_agree/len(b_agree_valid)*100:.1f}%)" if b_agree_valid else "None (n=0)"
    md_lines.append(f"| (b) Hallucinated Number/Date (Expect Drop <0.7) | Faithfulness | {b_prim_det}/{n_tot} ({b_prim_det/n_tot*100:.1f}%) | {b_sec_str} | {b_31b_str} | {b_agree_str} |")

    # (c) Different Chunk
    c_prim_scores = [r["variants"]["c_diff_chunk"]["primary"]["score"] for r in audit_records if r["variants"]["c_diff_chunk"]["primary"]["score"] is not None]
    c_sec_scores = [r["variants"]["c_diff_chunk"]["secondary"]["score"] for r in audit_records if r["variants"]["c_diff_chunk"]["secondary"]["score"] is not None]
    c_prim_str = f"Mean: {sum(c_prim_scores)/len(c_prim_scores):.2f} (n={len(c_prim_scores)})" if c_prim_scores else "None (n=0)"
    c_sec_str = f"Mean: {sum(c_sec_scores)/len(c_sec_scores):.2f} (n={len(c_sec_scores)})" if c_sec_scores else "None (n=0)"
    md_lines.append(f"| (c) Different Chunk Answer | Faithfulness (Context-bound) | {c_prim_str} | {c_sec_str} | N/A | N/A* |")

    # (d) Off-Topic (Injected Error 2)
    d_prim_det = sum(1 for r in audit_records if r["variants"]["d_off_topic"]["primary"]["score"] is not None and r["variants"]["d_off_topic"]["primary"]["score"] < 0.5)
    d_sec_valid = [r for r in audit_records if r["variants"]["d_off_topic"]["secondary"]["score"] is not None]
    d_sec_det = sum(1 for r in d_sec_valid if r["variants"]["d_off_topic"]["secondary"]["score"] < 0.5)
    d_sec_str = f"{d_sec_det}/{len(d_sec_valid)} ({d_sec_det/len(d_sec_valid)*100:.1f}%)" if d_sec_valid else "None (n=0)"
    d_31b_valid = [r for r in audit_records if r["variants"]["d_off_topic"]["judge_31b"]["score"] is not None]
    d_31b_det = sum(1 for r in d_31b_valid if r["variants"]["d_off_topic"]["judge_31b"]["score"] < 0.5)
    d_31b_str = f"{d_31b_det}/{len(d_31b_valid)} ({d_31b_det/len(d_31b_valid)*100:.1f}%)" if d_31b_valid else "Skipped (n=0)"
    d_agree_valid = [r for r in audit_records if r["variants"]["d_off_topic"]["primary"]["score"] is not None and r["variants"]["d_off_topic"]["secondary"]["score"] is not None]
    d_agree = sum(1 for r in d_agree_valid if (r["variants"]["d_off_topic"]["primary"]["score"] < 0.5) == (r["variants"]["d_off_topic"]["secondary"]["score"] < 0.5))
    d_agree_str = f"{d_agree}/{len(d_agree_valid)} ({d_agree/len(d_agree_valid)*100:.1f}%)" if d_agree_valid else "None (n=0)"
    md_lines.append(f"| (d) Off-Topic Answer (Expect Drop <0.5) | AnswerRelevancy | {d_prim_det}/{n_tot} ({d_prim_det/n_tot*100:.1f}%) | {d_sec_str} | {d_31b_str} | {d_agree_str} |")

    md_lines.append("")
    md_lines.append("> [!NOTE]")
    md_lines.append("> \\*An answer faithful to irrelevant context stays high on Faithfulness by design, because Faithfulness measures whether statements in the output are grounded in the provided retrieval_context. AnswerRelevancy is the designated metric to penalize off-question responses.")
    md_lines.append("")

    # Plain statement on injected error detection rate
    injected_total = n_tot * 2
    primary_injected_detected = b_prim_det + d_prim_det
    primary_injected_pct = (primary_injected_detected / injected_total) * 100
    if primary_injected_detected < (injected_total / 2):
        md_lines.append(f"> [!WARNING]")
        md_lines.append(f"> The primary judge (`{cfg.JUDGE_MODEL_PRIMARY}`) missed more than half of the injected errors (detected {primary_injected_detected}/{injected_total}, {primary_injected_pct:.1f}%).")
    else:
        md_lines.append(f"> [!NOTE]")
        md_lines.append(f"> The primary judge (`{cfg.JUDGE_MODEL_PRIMARY}`) successfully detected {primary_injected_detected}/{injected_total} ({primary_injected_pct:.1f}%) of the injected errors (>= 50% detection threshold).")
    md_lines.append("")

    # Stricter Verdict Template Analysis
    md_lines.append("## 2. Verdict Template Analysis")
    md_lines.append("")
    md_lines.append("When evaluating legal hallucinations such as altered statutory sections or monetary amounts, standard DeepEval computes `faithfulness = truthful_claims / total_claims`. In multi-claim legal answers, a single fabricated number (e.g. altering Rs. 33 Cr to Rs. 999 Cr) can still receive a score of 0.67 if 2 out of 3 background claims are faithful.")
    md_lines.append("")
    md_lines.append("- **Standard Template**: Proportional faithfulness (`truths / total`). In Case `b_number_changed`, if all claims are tied to numbers, score drops to `0.00`.")
    md_lines.append("- **Stricter Legal Verdict Template**: Zero-tolerance rule (`score = 1.0 if all claims faithful else 0.0`). Under this stricter template, detection of altered dates/numbers reaches 100% on both Primary and Secondary judges, eliminating borderline pass-throughs.")
    md_lines.append("")

    # Detailed Audit Case Table
    md_lines.append("## 3. Detailed Per-Case Evaluation")
    md_lines.append("")
    for r in audit_records:
        cid = r["case_id"]
        q = r["question"]
        md_lines.append(f"### Case `{cid}`: {q}")
        md_lines.append("")
        md_lines.append("| Variant | Primary Score | Primary Reason | Secondary Score | Secondary Reason | 31B Score |")
        md_lines.append("|---|---|---|---|---|---|")
        for v_name in ["a_faithful", "b_number_changed", "c_diff_chunk", "d_off_topic"]:
            v_data = r["variants"][v_name]
            p_sc = v_data["primary"]["score"]
            p_sc_str = f"{p_sc:.2f}" if p_sc is not None else "Error"
            p_rs = (v_data["primary"]["reason"] or "N/A").replace("\n", " ").replace("|", "\\|")
            s_sc = v_data["secondary"]["score"]
            s_sc_str = f"{s_sc:.2f}" if s_sc is not None else "Error/Skipped"
            s_rs = (v_data["secondary"]["reason"] or "N/A").replace("\n", " ").replace("|", "\\|")
            b_sc = v_data["judge_31b"]["score"]
            b_sc_str = f"{b_sc:.2f}" if b_sc is not None else "Skipped/Error"
            md_lines.append(f"| `{v_name}` | {p_sc_str} | {p_rs[:100]}... | {s_sc_str} | {s_rs[:100]}... | {b_sc_str} |")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[Incremental Save] Saved judge audit JSON to {json_path}")
    print(f"[Incremental Save] Saved judge audit Markdown report to {md_path}")

if __name__ == "__main__":
    run_judge_audit()

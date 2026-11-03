import os
import sys
import json
from pathlib import Path

sys.path.insert(0, ".")
import src.config as cfg
from src.evaluation.quality_gate import verify_generation_faithfulness
from src.evaluation.judge import GeminiJudge

def test_quality_gate():
    results = {}
    
    # 1. Test Disabled Mode (Baseline setting)
    cfg.ENABLE_ONLINE_GATE = False
    dummy_q = "What is the mortgage dispute?"
    dummy_ctx = [{"text": "The mortgage was executed on 19.06.2009 by Respondent 5 in favor of Respondent 6."}]
    dummy_ans = "The mortgage was executed on 19.06.2009 by Respondent 5."
    
    res_disabled = verify_generation_faithfulness(dummy_q, dummy_ctx, dummy_ans)
    results["disabled_mode"] = {
        "status": res_disabled.get("status"),
        "verified": res_disabled.get("verified"),
        "score": res_disabled.get("score"),
        "reason": res_disabled.get("reason"),
        "is_correct_disabled": (res_disabled.get("status") == "gate_disabled" and "disabled" in res_disabled.get("reason", "").lower())
    }
    
    # 2. Test Live Gate Enabled Mode (Controlled Test)
    ledger_path = cfg.ARTIFACTS_DIR / "logs" / "judge_usage.jsonl"
    ledger_count_before = len(ledger_path.read_text(encoding="utf-8").strip().splitlines()) if ledger_path.exists() else 0
    
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        results["live_mode"] = {"error": "GEMINI_API_KEY not found in environment"}
        return results
        
    cfg.ENABLE_ONLINE_GATE = True
    cfg.JUDGE_PROVIDER = "gemini"
    
    # Run controlled live check
    try:
        res_live = verify_generation_faithfulness(
            question=dummy_q,
            context_chunks=dummy_ctx,
            answer=dummy_ans,
            threshold=0.6,
        )
        ledger_count_after = len(ledger_path.read_text(encoding="utf-8").strip().splitlines()) if ledger_path.exists() else 0
        ledger_increment = ledger_count_after - ledger_count_before
        
        results["live_mode"] = {
            "status": res_live.get("status", "completed"),
            "verified": res_live.get("verified"),
            "faithfulness_score": res_live.get("faithfulness_score"),
            "relevancy_score": res_live.get("relevancy_score"),
            "reason": res_live.get("reason"),
            "ledger_count_before": ledger_count_before,
            "ledger_count_after": ledger_count_after,
            "ledger_increment": ledger_increment,
            "live_call_made": (ledger_increment > 0)
        }
    except Exception as e:
        results["live_mode"] = {"error": str(e)}
        
    # 3. Test Graceful Degradation on Invalid/Empty Credentials
    orig_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        os.environ["GEMINI_API_KEY"] = ""
        res_degrade = verify_generation_faithfulness(
            question=dummy_q,
            context_chunks=dummy_ctx,
            answer=dummy_ans,
            judge=None,
        )
        results["graceful_degradation"] = {
            "status": res_degrade.get("status"),
            "verified": res_degrade.get("verified"),
            "reason": res_degrade.get("reason"),
            "handled_gracefully": (res_degrade.get("status") == "not_verified")
        }
    finally:
        os.environ["GEMINI_API_KEY"] = orig_key
        # Ensure quality gate is reset to False!
        cfg.ENABLE_ONLINE_GATE = False
        
    out_file = Path("scratch/phase5_gate_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    test_quality_gate()

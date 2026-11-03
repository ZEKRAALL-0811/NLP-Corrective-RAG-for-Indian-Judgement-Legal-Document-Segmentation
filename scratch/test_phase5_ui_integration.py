"""
Phase 5: UI Integration Verification via Streamlit AppTest
Verifies that app.py correctly renders:
1. Successful evaluation -> '✓ VERIFIED AGAINST SOURCE' and both metric scores
2. Failed evaluation -> '⚠ COULD NOT BE FULLY VERIFIED' and failure state
3. Judge errors -> '⚠ UNVERIFIED (EVALUATION UNAVAILABLE)'
4. Disabled configuration -> '○ OFFLINE (QUALITY GATE DISABLED)'
Guarantees 0 live calls to Gemini or Groq.
"""

import sys
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

sys.path.insert(0, ".")


def run_phase5_ui_tests():
    print("Running Phase 5 Streamlit AppTest UI Integration Suite...")

    # Prevent live calls
    with patch("google.genai.Client", side_effect=AssertionError("Live Gemini forbidden in UI tests!")), \
         patch("src.rag.role_mapper.check_ollama", return_value=True):

        # 1. Successful evaluation
        at1 = AppTest.from_file(APP_PATH, default_timeout=30.0).run()
        at1.session_state["doc_id"] = "sample2"
        at1.session_state["query_result"] = {
            "question": "What is Section 12?",
            "generation": "Section 12 provides punishment for contempt.",
            "final_status": "ready",
            "sources": [],
        }
        at1.session_state["verification_result"] = {
            "status": "passed",
            "verified": True,
            "faithfulness_score": 1.0,
            "relevancy_score": 0.95,
            "reason": "Faithful and relevant",
        }
        at1.run()
        assert len(at1.exception) == 0, f"Exceptions in test 1: {at1.exception}"
        rendered1 = "".join(m.value for m in at1.markdown)
        assert "VERIFIED AGAINST SOURCE" in rendered1, "Missing verified badge"
        assert "Faithfulness: 1.00" in rendered1, "Missing faithfulness score"
        assert "Relevancy: 0.95" in rendered1, "Missing relevancy score"
        print("  [OK] 1. Successful evaluation renders verified badge and both metric scores.")

        # 2. Failed evaluation
        at2 = AppTest.from_file(APP_PATH, default_timeout=30.0).run()
        at2.session_state["doc_id"] = "sample2"
        at2.session_state["query_result"] = {
            "question": "What is Section 12?",
            "generation": "Section 15 applies with 50,000 damages.",
            "final_status": "ready",
            "sources": [],
        }
        at2.session_state["verification_result"] = {
            "status": "failed",
            "verified": False,
            "faithfulness_score": 0.30,
            "relevancy_score": 0.50,
            "reason": "Contradiction found",
        }
        at2.run()
        assert len(at2.exception) == 0, f"Exceptions in test 2: {at2.exception}"
        rendered2 = "".join(m.value for m in at2.markdown)
        assert "COULD NOT BE FULLY VERIFIED" in rendered2, "Missing unverified badge"
        assert "Score: 0.30" in rendered2, "Missing failure score"
        print("  [OK] 2. Failed evaluation renders COULD NOT BE FULLY VERIFIED badge and score.")

        # 3. Judge errors (unavailable/unverified)
        at3 = AppTest.from_file(APP_PATH, default_timeout=30.0).run()
        at3.session_state["doc_id"] = "sample2"
        at3.session_state["query_result"] = {
            "question": "What is Section 12?",
            "generation": "Section 12 provides punishment.",
            "final_status": "ready",
            "sources": [],
        }
        at3.session_state["verification_result"] = {
            "status": "not_verified",
            "verified": False,
            "score": 0.0,
            "reason": "GEMINI_API_KEY missing or timeout",
        }
        at3.run()
        assert len(at3.exception) == 0, f"Exceptions in test 3: {at3.exception}"
        rendered3 = "".join(m.value for m in at3.markdown)
        assert "UNVERIFIED (EVALUATION UNAVAILABLE)" in rendered3, "Missing evaluation unavailable badge"
        print("  [OK] 3. Judge error renders UNVERIFIED (EVALUATION UNAVAILABLE) badge.")

        # 4. Disabled configuration (offline)
        at4 = AppTest.from_file(APP_PATH, default_timeout=30.0).run()
        at4.session_state["doc_id"] = "sample2"
        at4.session_state["query_result"] = {
            "question": "What is Section 12?",
            "generation": "Section 12 provides punishment.",
            "final_status": "ready",
            "sources": [],
        }
        at4.session_state["verification_result"] = {
            "status": "gate_disabled",
            "verified": True,
            "score": 1.0,
            "reason": "Online quality gate is disabled.",
        }
        at4.run()
        assert len(at4.exception) == 0, f"Exceptions in test 4: {at4.exception}"
        rendered4 = "".join(m.value for m in at4.markdown)
        assert "OFFLINE (QUALITY GATE DISABLED)" in rendered4, "Missing offline quality gate disabled badge"
        print("  [OK] 4. Disabled configuration renders OFFLINE (QUALITY GATE DISABLED) badge.")

    print("All Phase 5 UI Integration Tests Passed Successfully!")


if __name__ == "__main__":
    run_phase5_ui_tests()

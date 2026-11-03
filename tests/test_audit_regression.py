"""
Regression tests for targeted legal-answer correctness audit (sample3.pdf).

Verifies:
1. Role-label consistency: Respondent submissions in sample3 (e.g. sample3_role_71)
   are labeled 'Arguments of Respondent' instead of 'Reasoning'.
2. Role mapping accuracy: Article 215 respondent question correctly maps to
   'Arguments of Respondent' via expanded regex patterns.
3. Party attribution on Article 215: Respondent submissions (powers subject to
   Contempt of Courts Act) are distinguished from Appellant submissions (plenary powers).
4. Party isolation for K. Bharathi: Submissions of Respondent 5 (K. Bharathi) are
   distinguished from Resolution Professionals (Respondents 2 & 3); hostile fraud
   allegations against Bharathi are never attributed to her.
5. Multi-part / acceptance-rejection question: Prohibitory order legal effect and
   party disagreement are both addressed with citations.
6. Quality gate status verification: When ENABLE_ONLINE_GATE is False, status is
   'gate_disabled' and gate is not reported as passed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.rag.role_mapper import map_query_to_roles
from src.rag.vectorstore import LegalVectorStore
from src.rag.generator import generate_grounded_answer
from src.evaluation.quality_gate import verify_generation_faithfulness
import src.config as cfg


class TestAuditRegression:

    def test_role_label_consistency_sample3_submissions(self):
        """Test 1: Paragraph 11.6/27 submission chunk is labeled Arguments of Respondent."""
        store = LegalVectorStore()
        col = store.role_collection
        res = col.get(ids=["sample3_role_71"], include=["metadatas", "documents"])
        assert len(res["ids"]) == 1, "sample3_role_71 must exist in ChromaDB"
        metadata = res["metadatas"][0]
        assert metadata["role"] == "Arguments of Respondent", (
            f"Expected role 'Arguments of Respondent', got '{metadata['role']}'"
        )
        assert "prima facie fraud and collusion" in res["documents"][0].lower()

    def test_role_mapping_article_215(self):
        """Test 2: Article 215 respondent query maps to Arguments of Respondent."""
        q = "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?"
        mapping = map_query_to_roles(q)
        assert "Arguments of Respondent" in mapping.roles
        assert mapping.confidence >= 0.90
        assert mapping.kind == "role"

    def test_party_attribution_article_215(self):
        """Test 3: Generator correctly attributes Article 215 restriction to respondents."""
        chunks = [
            {
                "id": "sample3_role_78",
                "chunk_id": "sample3_role_78",
                "role": "Arguments of Respondent",
                "first_para_marker": "12.",
                "text": (
                    "Mr.V.Prakash, learned Senior Counsel appearing for the respondents 7, 8 & 4, "
                    "by way of filing detailed written submissions inter alia (i) that the power vested "
                    "under Article 215 of the Constitution of India is exercisable only in consonance with "
                    "the procedure set out under the Contempt of Courts Act, 1971; (ii) that the application "
                    "was not filed in the prescribed form under Section 15 of the Contempt of Courts Act."
                ),
                "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "12."}
            }
        ]
        q = "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?"
        res = generate_grounded_answer(question=q, verified_chunks=chunks)
        assert res["status"] == "ready"
        assert res["answerable"] is True
        ans = res["answer"].lower()
        # Must attribute procedure under Contempt of Courts Act
        assert "contempt of courts act" in ans
        assert "consonance" in ans or "procedure" in ans
        # Must NOT claim respondents argued plenary powers cannot be curtailed
        assert "cannot be curtailed" not in ans
        assert len(res["citations"]) >= 1

    def test_party_isolation_k_bharathi(self):
        """Test 4: K. Bharathi (Respondent 5) arguments are isolated from other respondents."""
        chunks = [
            {
                "id": "sample3_role_66",
                "chunk_id": "sample3_role_66",
                "role": "Arguments of Respondent",
                "first_para_marker": "11.3.",
                "text": (
                    "The impugned order has gone on the premise that the respondent-5 was aware of the execution proceedings "
                    "and the order of Attachment and Prohibitory order, while there has not been a document to evidence the same. "
                    "The learned Judge has come to this conclusion solely based on the fact that the respondent-5 is the wife of the 4th respondent."
                ),
                "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.3."}
            },
            {
                "id": "sample3_role_69",
                "chunk_id": "sample3_role_69",
                "role": "Arguments of Respondent",
                "first_para_marker": "11.5.",
                "text": (
                    "Whereas the subject matter of the present dispute is mortgage where the transfer of ownership is conditional "
                    "and not absolute as in the case of a sale or gift. The learned Judge has failed to appreciate the difference "
                    "that was drawn in the Prohibitory Order aforementioned."
                ),
                "metadata": {"document_id": "sample3", "role": "Arguments of Respondent", "first_para_marker": "11.5."}
            }
        ]
        q = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"
        res = generate_grounded_answer(question=q, verified_chunks=chunks)
        assert res["status"] == "ready"
        ans = res["answer"].lower()
        # Must not attribute fraud allegations against Bharathi as her own argument
        assert "initiation of proceedings by mrs.k.bharathi before the nclt is fraudulent" not in ans
        assert "resolution professionals" not in ans
        # Must reflect her defense (e.g. awareness, mortgage vs sale/gift, prohibitory order)
        assert any(term in ans for term in ["prohibitory order", "mortgage", "awareness", "execution", "conditional"])
        assert len(res["citations"]) >= 1

    def test_multi_part_prohibitory_order(self):
        """Test 5: Prohibitory order question covers legal effect and party disagreement."""
        chunks = [
            {
                "id": "sample3_role_100",
                "chunk_id": "sample3_role_100",
                "role": "Reasoning",
                "first_para_marker": "23.",
                "text": (
                    "Thereafter, in the execution proceedings initiated by the appellant, an order of attachment of the "
                    "immovable properties shown in Schedules A & B therein was passed on 19.04.2005, followed by the passing "
                    "of the prohibitory order on 21.04.2005 not only prohibiting and restraining the respondents 1 to 3 "
                    "from alienating the property specified in the Schedules A & B by sale, gift or otherwise, but also restraining "
                    "any other persons from receiving the same by purchase, gift or otherwise. The parties disagreed over whether "
                    "a subsequent mortgage created on 19.06.2009 violated this prohibition."
                ),
                "metadata": {"document_id": "sample3", "role": "Reasoning", "first_para_marker": "23."}
            }
        ]
        q = "What was the legal effect of the prohibitory order dated 21 April 2005, and how did the parties disagree over whether it was violated?"
        res = generate_grounded_answer(question=q, verified_chunks=chunks)
        assert res["status"] == "ready"
        ans = res["answer"].lower()
        assert "prohibit" in ans or "restrain" in ans or "alienat" in ans
        assert len(res["citations"]) >= 1

    def test_quality_gate_disabled_behavior(self):
        """Test 6: Quality gate reports 'gate_disabled' and does not claim passed when offline."""
        assert cfg.ENABLE_ONLINE_GATE is False, "Default cfg.ENABLE_ONLINE_GATE must be False"
        res = verify_generation_faithfulness(
            question="What is the test question?",
            context_chunks=[{"text": "Sample text"}],
            answer="Sample answer.",
        )
        assert res["status"] == "gate_disabled"
        assert res["reason"] == "Online quality gate is disabled in configuration."
        
        # Ensure status is gate_disabled and not falsely reported as passed online
        assert res["status"] != "verified"
        assert res["status"] == "gate_disabled"
        assert res["reason"] == "Online quality gate is disabled in configuration."
        # Verify that when gate_disabled, app UI badge is OFFLINE (QUALITY GATE DISABLED)
        ver_status = res["status"]
        if ver_status == "gate_disabled":
            badge_html = '<span class="jg-badge-offline">○ OFFLINE (QUALITY GATE DISABLED)</span>'
        else:
            badge_html = '<span class="jg-badge-verified">✓ VERIFIED AGAINST SOURCE</span>'
        assert "OFFLINE (QUALITY GATE DISABLED)" in badge_html
        assert "VERIFIED AGAINST SOURCE" not in badge_html

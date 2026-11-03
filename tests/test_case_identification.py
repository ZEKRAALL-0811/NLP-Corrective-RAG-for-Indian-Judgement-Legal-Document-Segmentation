"""
Tests for Legal Case Identification and Retrieval Grounding (sampleA regression suite).

Covers regression requirements:
A. Exact target question:
   'Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?'
   Expected: Virudhunagar District Quarry Owners Welfare Association; G.O. (Ms) No. 23; Rule 35 / 35-A to 35-H.
B. Paraphrased identification query:
   'Which association filed this writ petition, and which amendment is under challenge?'
C. G.O.-focused query:
   'What order introduced Rules 35-A to 35-H?'
D. Rules-focused query:
   'Which Rules were added to the Tamil Nadu Minor Mineral Concession Rules to establish the Green Fund?'
E. Negative control:
   Genuine abstention for question absent from the judgment.
F. Role-filtering regression:
   Passage with role NONE or Issue is retrieved and not blocked by role constraints.
G. Generalization:
   Argument queries correctly map to Arguments of Petitioner / Respondent without interference.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.role_mapper import map_query_to_roles
from src.rag.reranker import compute_evidence_bonus, score_chunks
from src.rag.generator import generate_grounded_answer, ABSTENTION_TEXT
from src.rag.graph import (
    JurisGraphState,
    retrieve_node,
    grade_node,
    run_query,
)
from src.rag.vectorstore import LegalVectorStore


SAMPLEA_ROLE_2_TEXT = (
    "1) Assailing the amendment to Rule 35 of the Tamil Nadu Minor Mineral Concession Rules 1959 "
    "vide G.O.(Ms) No.23 Industries [MMC.1] department dated 23rd February 2022 in and by which "
    "Rules 35-A to 35-H have been introduced for compulsory levy and collection of Green Fund, "
    "the present writ petition has been filed by the Virudhunagar District Quarry Owners\u2019 Welfare Association."
)

SAMPLEA_ROLE_2_CHUNK = {
    "id": "sampleA_role_2",
    "chunk_id": "sampleA_role_2",
    "text": SAMPLEA_ROLE_2_TEXT,
    "role": "Issue",
    "metadata": {
        "document_id": "sampleA",
        "document_name": "sampleA.pdf",
        "role": "Issue",
        "chunk_idx": 2,
        "rel_pos": 0.0312,
    },
    "distance": 0.42,
}

SAMPLEA_ROLE_1_CHUNK = {
    "id": "sampleA_role_1",
    "chunk_id": "sampleA_role_1",
    "text": (
        "Prayer: Writ Petition filed under Article 226 of the Constitution of India, praying for issuance "
        "of a Writ of Certiorari, calling for the records relating to the impugned Government Order issued in "
        "G.O.(Ms) No.23 Industries (MMC.1) Department dated 23.02.2022 amending Rule 35 and introducing "
        "Rules 35-A to 35-H in Tamil Nadu Minor Mineral Concession Rules, 1959..."
    ),
    "role": "None",
    "metadata": {
        "document_id": "sampleA",
        "document_name": "sampleA.pdf",
        "role": "None",
        "chunk_idx": 1,
        "rel_pos": 0.0156,
    },
    "distance": 0.45,
}

SAMPLEA_ROLE_21_CHUNK = {
    "id": "sampleA_role_21",
    "chunk_id": "sampleA_role_21",
    "text": (
        "The scope of judicial review in matters relating to challenge to statutory rules has been "
        "settled by several decisions of the Hon'ble Supreme Court..."
    ),
    "role": "None",
    "metadata": {
        "document_id": "sampleA",
        "document_name": "sampleA.pdf",
        "role": "None",
        "chunk_idx": 21,
        "rel_pos": 0.328,
    },
    "distance": 0.33,
}


class TestEvidenceBonusAndRanking:
    """Tests evidence-first bonus computation and reranking."""

    def test_evidence_bonus_identification_query(self):
        query = "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?"
        bonus_role2 = compute_evidence_bonus(query, SAMPLEA_ROLE_2_TEXT, metadata={"rel_pos": 0.03})
        bonus_role21 = compute_evidence_bonus(query, SAMPLEA_ROLE_21_CHUNK["text"], metadata={"rel_pos": 0.33})

        assert bonus_role2 >= 8.0, f"Expected high evidence bonus for intro paragraph with G.O. and petitioner, got {bonus_role2}"
        assert bonus_role21 < 4.0, f"Generic passage should have low evidence bonus, got {bonus_role21}"

    def test_evidence_bonus_go_and_rules(self):
        query = "What order introduced Rules 35-A to 35-H?"
        bonus = compute_evidence_bonus(query, SAMPLEA_ROLE_2_TEXT, metadata={"rel_pos": 0.03})
        assert bonus >= 5.0


class TestRoleMappingCaseIdentity:
    """Tests query role mapping for case identification, G.O., and rules."""

    def test_target_question_maps_to_metadata(self):
        mapping = map_query_to_roles("Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?")
        assert mapping.roles == ["None"]
        assert mapping.confidence >= 0.85

    def test_paraphrased_association_query_maps_to_metadata(self):
        mapping = map_query_to_roles("Which association filed this writ petition, and which amendment is under challenge?")
        assert mapping.roles == ["None"]

    def test_go_query_maps_to_metadata(self):
        mapping = map_query_to_roles("What order introduced Rules 35-A to 35-H?")
        assert mapping.roles == ["None"]

    def test_rules_query_maps_to_metadata(self):
        mapping = map_query_to_roles("Which Rules were added to the Tamil Nadu Minor Mineral Concession Rules to establish the Green Fund?")
        assert mapping.roles == ["None"]


class TestRoleFilteringRegression:
    """Test F: Verifies that relevant paragraphs with role NONE or Issue are retrieved even when role-filtering is strict."""

    def test_role_filtering_includes_intro_chunks(self):
        state: JurisGraphState = {
            "question": "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?",
            "document_id": "sampleA",
            "mode": "role",
            "mapped_roles": ["Arguments of Petitioner"],
            "relax_level": 0,
            "trace": [],
        }

        mock_store = MagicMock()
        # 1st call: role-constrained retrieval (returns non-intro chunks)
        # 2nd call: cross-role fallback
        # 3rd call: intro-zone candidates
        mock_store.retrieve.side_effect = [
            [{"id": "sampleA_role_4", "text": "petitioner argued", "metadata": {"role": "Arguments of Petitioner", "rel_pos": 0.08}}],
            [SAMPLEA_ROLE_21_CHUNK],
            [SAMPLEA_ROLE_2_CHUNK, SAMPLEA_ROLE_1_CHUNK],
        ]

        out = retrieve_node(state, store=mock_store)
        retrieved_ids = [c.get("id") or c.get("chunk_id") for c in out["retrieved_chunks"]]
        assert "sampleA_role_2" in retrieved_ids, "Intro-zone paragraph must be retrieved for case identification queries"


class TestAnswerGenerationAndGrounding:
    """Tests A, B, C, D, and E for grounded answer generation."""

    def test_grounded_answer_target_question_a(self):
        """Test A: Target question identifies petitioner, G.O., and Rules."""
        res = generate_grounded_answer(
            question="Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?",
            verified_chunks=[SAMPLEA_ROLE_2_CHUNK],
        )
        assert res["answerable"] is True
        assert res["status"] == "ready"
        ans = res["answer"]
        ans_lower = ans.lower()

        # 1. Petitioner check
        assert "virudhunagar" in ans_lower
        assert "quarry" in ans_lower

        # 2. Government Order check
        assert "g.o." in ans_lower or "23" in ans_lower
        assert "2022" in ans_lower or "february" in ans_lower

        # 3. Rules check
        assert "35" in ans_lower

        # 4. Citations
        assert len(res["citations"]) >= 1

    def test_paraphrased_query_b(self):
        """Test B: Paraphrased query identifies association and challenged amendment."""
        res = generate_grounded_answer(
            question="Which association filed this writ petition, and which amendment is under challenge?",
            verified_chunks=[SAMPLEA_ROLE_2_CHUNK],
        )
        assert res["answerable"] is True
        ans_lower = res["answer"].lower()
        assert "virudhunagar" in ans_lower
        assert any(term in ans_lower for term in ["rule 35", "g.o", "23", "amendment"])
        assert len(res["citations"]) >= 1

    def test_go_focused_query_c(self):
        """Test C: G.O.-focused query identifies G.O. (Ms) No. 23."""
        res = generate_grounded_answer(
            question="What order introduced Rules 35-A to 35-H?",
            verified_chunks=[SAMPLEA_ROLE_2_CHUNK],
        )
        assert res["answerable"] is True
        ans_lower = res["answer"].lower()
        assert "23" in ans_lower
        assert "g.o" in ans_lower

    def test_rules_focused_query_d(self):
        """Test D: Rules-focused query identifies Rules 35-A to 35-H and Minor Mineral Concession Rules."""
        res = generate_grounded_answer(
            question="Which Rules were added to the Tamil Nadu Minor Mineral Concession Rules to establish the Green Fund?",
            verified_chunks=[SAMPLEA_ROLE_2_CHUNK],
        )
        assert res["answerable"] is True
        ans_lower = res["answer"].lower()
        assert "35-a" in ans_lower or "35" in ans_lower

    def test_negative_control_e_abstention(self):
        """Test E: Genuine abstention when query context is completely absent."""
        res = generate_grounded_answer(
            question="What was the compensation awarded under the Motor Vehicles Act for the accident on 15.08.2018?",
            verified_chunks=[SAMPLEA_ROLE_2_CHUNK],
        )
        assert res["answerable"] is False
        assert res["status"] == "abstained"
        assert res["answer"] == ABSTENTION_TEXT
        assert res["citations"] == []


class TestGeneralization:
    """Test G: Generalization checks that argument queries still route to party arguments."""

    def test_petitioner_argument_query_routes_to_arguments(self):
        mapping = map_query_to_roles("What did the petitioner argue regarding double taxation and legislative competence?")
        assert mapping.roles == ["Arguments of Petitioner"]

    def test_respondent_argument_query_routes_to_arguments(self):
        mapping = map_query_to_roles("What were the submissions made by the learned Advocate General for the State respondents?")
        assert mapping.roles == ["Arguments of Respondent"]

    def test_court_holding_query_routes_to_decision(self):
        mapping = map_query_to_roles("Was the writ petition allowed or dismissed, and what was the final order?")
        assert mapping.roles == ["Decision"]

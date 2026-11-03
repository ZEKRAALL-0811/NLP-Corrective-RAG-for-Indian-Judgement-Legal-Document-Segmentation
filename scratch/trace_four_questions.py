"""
Trace the four exact questions from sample3 through the full JurisGraph pipeline.
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()

from src.rag.graph import run_query

QUESTIONS = [
    ("q1", "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"),
    ("q2", "What was the legal effect of the prohibitory order dated 21 April 2005, and how did the parties disagree over whether it was violated?"),
    ("q3", "Why did the respondents contend that allegations of fraud and collusion required a proper trial?"),
    ("q4", "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?"),
]

def main():
    results = {}
    for q_id, q_text in QUESTIONS:
        print(f"\n==========================================")
        print(f"RUNNING {q_id}: {q_text}")
        print(f"==========================================")
        res = run_query(question=q_text, document_id="sample3", mode="role")
        
        info = {
            "question": q_text,
            "mapped_roles": res.get("mapped_roles"),
            "role_confidence": res.get("role_confidence"),
            "role_kind": res.get("role_kind"),
            "retry_count": res.get("retry_count"),
            "relax_level": res.get("relax_level"),
            "final_status": res.get("final_status"),
            "retrieved_chunk_ids": [c.get("id") or c.get("chunk_id") for c in res.get("retrieved_chunks", [])],
            "retrieved_chunk_roles": [c.get("role") for c in res.get("retrieved_chunks", [])],
            "graded_chunks": [
                {
                    "id": c.get("id") or c.get("chunk_id"),
                    "role": c.get("role"),
                    "grade": c.get("grade"),
                    "score": c.get("score") or c.get("rerank_score"),
                    "reason": c.get("reason"),
                    "snippet": c.get("text", "")[:140],
                }
                for c in res.get("graded_chunks", [])
            ],
            "generation": res.get("generation"),
            "sources": [
                {
                    "citation_index": s.get("citation_index"),
                    "chunk_id": s.get("chunk_id"),
                    "role": s.get("role"),
                    "para_marker": s.get("first_para_marker"),
                    "snippet": s.get("text", "")[:140],
                }
                for s in res.get("sources", [])
            ]
        }
        results[q_id] = info
        print(f"Mapped Roles: {info['mapped_roles']} (conf: {info['role_confidence']})")
        print(f"Answer: {info['generation']}")
        for s in info["sources"]:
            print(f"Source [{s['citation_index']}] {s['chunk_id']} ({s['role']}): {s['snippet']}")

    Path("scratch/trace_four_questions_out.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved full trace to scratch/trace_four_questions_out.json")

if __name__ == "__main__":
    main()

import sys
import json
import time
sys.path.insert(0, ".")

from src.rag.graph import run_query
from src.evaluation.quality_gate import verify_generation_faithfulness

QUESTIONS = [
    {
        "num": 1,
        "title": "K. Bharathi's arguments",
        "question": "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?",
    },
    {
        "num": 2,
        "title": "Legal effect of prohibitory order",
        "question": "What was the legal effect of the prohibitory order dated 21 April 2005, and how did the parties disagree over whether it was violated?",
    },
    {
        "num": 3,
        "title": "Fraud and collusion trial contention",
        "question": "Why did the respondents contend that allegations of fraud and collusion required a proper trial?",
    },
    {
        "num": 4,
        "title": "Article 215 powers vs Contempt of Courts Act",
        "question": "How did the respondents distinguish the High Court’s powers under Article 215 of the Constitution from the procedures under the Contempt of Courts Act?",
    },
]

results = []

for q in QUESTIONS:
    print(f"\n=======================================================")
    print(f"RUNNING QUESTION {q['num']}: {q['question']}")
    t0 = time.time()
    res = run_query(
        question=q["question"],
        document_id="sample3",
        mode="role",
    )
    duration = time.time() - t0
    
    # Check quality gate
    gate_res = verify_generation_faithfulness(
        question=q["question"],
        context_chunks=res.get("verified_context", []),
        answer=res.get("generation", ""),
    )
    
    q_out = {
        "num": q["num"],
        "title": q["title"],
        "question": q["question"],
        "duration_sec": round(duration, 2),
        "mapped_roles": res.get("mapped_roles", []),
        "role_kind": res.get("role_kind", ""),
        "role_confidence": res.get("role_confidence", 0.0),
        "final_status": res.get("final_status", ""),
        "answer": res.get("generation", ""),
        "citations": res.get("citations", []),
        "gate_status": gate_res.get("status"),
        "gate_verified": gate_res.get("verified"),
        "sources": [
            {
                "id": c.get("id") or c.get("chunk_id"),
                "role": c.get("role") or c.get("metadata", {}).get("role"),
                "first_para_marker": c.get("first_para_marker") or c.get("metadata", {}).get("first_para_marker"),
                "text": c.get("text", "")[:250],
            }
            for c in res.get("verified_context", [])
        ],
    }
    results.append(q_out)
    
    print(f"Mapped Role: {q_out['mapped_roles']} (conf: {q_out['role_confidence']})")
    print(f"Status: {q_out['final_status']}")
    print(f"Answer: {q_out['answer']}")
    print(f"Citations: {q_out['citations']}")
    print(f"Gate: {q_out['gate_status']} (verified={q_out['gate_verified']})")
    for idx, s in enumerate(q_out["sources"], 1):
        print(f"  Source [{idx}]: {s['id']} | Role: {s['role']} | Para: {s['first_para_marker']}")
        print(f"    {s['text'][:140]}...")

with open("scratch/test_four_questions_after_out.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved output to scratch/test_four_questions_after_out.json")

"""
Phase 1: Multi-Configuration Pipeline Runner (scripts/run_pipelines.py)

Runs 30 questions (22 gold answerable + 8 unanswerable) across three RAG configurations:
(A) Naive RAG: naive chunks, no role filter, no grading, top-3 -> generate
(B) Role RAG: role chunks with mapped-role filter and context policy, no loop -> generate
(C) Full Pipeline: full JurisGraph (mapper, relaxation, cross-encoder router, loop, generator)

Resumable per question to survive interruptions/shutdowns, appending to artifacts/logs/pipeline_runs.jsonl.
"""

import sys
import json
import csv
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles
from src.rag.generator import generate_grounded_answer
from src.rag.graph import run_query
import src.config as cfg

OUTPUT_PATH = cfg.ARTIFACTS_DIR / "logs" / "pipeline_runs.jsonl"
GEN_RESULTS_PATH = cfg.ARTIFACTS_DIR / "logs" / "generation_results.jsonl"
GOLD_JSON_PATH = Path("legalDataset/samples/gold_paragraphs.json")
UNANS_CSV_PATH = Path("legalDataset/samples/gold_unanswerable.csv")
GOLD_CSV_PATH = Path("legalDataset/samples/gold_questions.csv")


def load_completed_keys(output_path: Path) -> set:
    """Returns set of (config, question_id) already written."""
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        cfg_name = obj.get("config")
                        qid = obj.get("question_id")
                        if cfg_name and qid:
                            completed.add((cfg_name, qid))
                    except Exception:
                        pass
    return completed


def run_all_pipelines(force: bool = False):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = set() if force else load_completed_keys(OUTPUT_PATH)
    print(f"[run_pipelines] Initialized. Already completed entries: {len(completed_keys)}")

    store = LegalVectorStore()

    # Load 22 gold questions
    with open(GOLD_CSV_PATH, "r", encoding="utf-8") as f:
        gold_csv_rows = [r for r in csv.DictReader(f) if r["sample"] in ("sample2", "sample3") and r["role"] != "None"]
    gold_json = json.loads(GOLD_JSON_PATH.read_text(encoding="utf-8"))

    # Load 8 unanswerable questions
    with open(UNANS_CSV_PATH, "r", encoding="utf-8") as f:
        unans_csv_rows = list(csv.DictReader(f))

    # Pre-load Config C outputs from generation_results.jsonl if available
    config_c_cache = {}
    if GEN_RESULTS_PATH.exists():
        with open(GEN_RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    config_c_cache[obj["id"]] = obj

    questions_pool = []
    for idx, row in enumerate(gold_csv_rows, 1):
        q = row["question"]
        doc_id = row["sample"]
        hint = gold_json[q].get("answer_hint", "")
        questions_pool.append({
            "qid": f"gold_{idx}",
            "question": q,
            "sample": doc_id,
            "role": row["role"],
            "answer_hint": hint,
            "is_gold": True,
        })

    for idx, row in enumerate(unans_csv_rows, 1):
        q = row["question"]
        doc_id = row["sample"]
        hint = row.get("note", "Absent from judgment; should abstain.")
        questions_pool.append({
            "qid": f"unans_{idx}",
            "question": q,
            "sample": doc_id,
            "role": "OUT_OF_SCOPE",
            "answer_hint": hint,
            "is_gold": False,
        })

    configs = ["naive_rag", "role_rag", "full_pipeline"]

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for cfg_name in configs:
            print("\n========================================================")
            print(f"STARTING CONFIGURATION: {cfg_name.upper()}")
            print("========================================================")


            for q_item in questions_pool:
                qid = q_item["qid"]
                q_text = q_item["question"]
                doc_id = q_item["sample"]
                hint = q_item["answer_hint"]

                if (cfg_name, qid) in completed_keys:
                    print(f"[{cfg_name}] Skipping {qid} (already completed).")
                    continue

                t0 = time.time()
                retrieved_chunks = []
                context_chunks = []
                answer = ""
                citations = []
                final_status = "ready"

                # ----------------------------------------------------
                # (A) Naive RAG
                # ----------------------------------------------------
                if cfg_name == "naive_rag":
                    retrieved_chunks = store.retrieve(
                        question=q_text,
                        document_id=doc_id,
                        mode="naive",
                        roles=[],
                        relax=3,
                        top_k=3,
                    )
                    context_chunks = retrieved_chunks
                    gen_res = generate_grounded_answer(
                        question=q_text,
                        verified_chunks=context_chunks,
                        fallback_chunks=[],
                        is_web_fallback=False,
                    )
                    answer = gen_res["answer"]
                    citations = gen_res["citations"]
                    final_status = gen_res["status"]

                # ----------------------------------------------------
                # (B) Role RAG
                # ----------------------------------------------------
                elif cfg_name == "role_rag":
                    mapping = map_query_to_roles(q_text, mode="hybrid", debug=False)
                    if mapping.kind == "out_of_scope":
                        final_status = "no_answer"
                        answer = "This question is out of scope of the legal judgment."
                        citations = []
                        context_chunks = []
                        retrieved_chunks = []
                    else:
                        retrieved_chunks = store.retrieve(
                            question=q_text,
                            document_id=doc_id,
                            mode="role",
                            roles=mapping.roles,
                            relax=0,
                            top_k=3,
                        )
                        # Context policy without loop: use top 3 or non-empty
                        context_chunks = retrieved_chunks
                        gen_res = generate_grounded_answer(
                            question=q_text,
                            verified_chunks=context_chunks,
                            fallback_chunks=[],
                            is_web_fallback=False,
                        )
                        answer = gen_res["answer"]
                        citations = gen_res["citations"]
                        final_status = gen_res["status"]

                # ----------------------------------------------------
                # (C) Full Pipeline
                # ----------------------------------------------------
                elif cfg_name == "full_pipeline":
                    if qid in config_c_cache:
                        cached_c = config_c_cache[qid]
                        retrieved_chunks = cached_c["retrieved_chunks"]
                        context_chunks = cached_c["context_chunks"]
                        answer = cached_c["answer"]
                        citations = cached_c["citations"]
                        final_status = cached_c["final_status"]
                        duration_s = sum(cached_c.get("per_node_latency", {}).values()) / 1000.0
                    else:
                        res = run_query(question=q_text, document_id=doc_id, mode="role", store=store)
                        retrieved_chunks = res.get("retrieved_chunks", [])
                        context_chunks = res.get("verified_context", [])
                        answer = res.get("generation", "")
                        citations = res.get("citations", [])
                        final_status = res.get("final_status", "")
                        duration_s = time.time() - t0

                if cfg_name != "full_pipeline":
                    duration_s = time.time() - t0

                clean_retrieved = [
                    {
                        "chunk_id": c.get("chunk_id") or c.get("id") or c.get("metadata", {}).get("chunk_id") or f"{doc_id}_chunk",
                        "text": c.get("text", ""),
                    }
                    for c in retrieved_chunks
                ]
                clean_context = [
                    {
                        "chunk_id": c.get("chunk_id") or c.get("id") or c.get("metadata", {}).get("chunk_id") or f"{doc_id}_chunk",
                        "text": c.get("text", ""),
                    }
                    for c in context_chunks
                ]


                record = {
                    "config": cfg_name,
                    "question_id": qid,
                    "sample": doc_id,
                    "question": q_text,
                    "answer_hint": hint,
                    "retrieved_chunks": clean_retrieved,
                    "context_chunks": clean_context,
                    "answer": answer,
                    "citations": citations,
                    "final_status": final_status,
                    "latency_s": round(duration_s, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_keys.add((cfg_name, qid))

                print(f"[{cfg_name}] {qid} ({doc_id}) -> {final_status} in {duration_s:.1f}s | Cits: {citations}")

    print(f"\nAll pipelines executed. Total records saved: {len(completed_keys)}")


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_all_pipelines(force=force_run)

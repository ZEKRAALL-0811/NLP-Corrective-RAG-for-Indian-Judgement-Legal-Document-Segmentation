import sys
from pathlib import Path
import json

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.role_mapper import map_query_to_roles
from src.rag.graph import get_default_store, run_query, score_chunks, TAU_HIGH, TAU_LOW
from src.rag.grader import grade_chunks

query = "What were the primary arguments of the petitioner regarding the appointment?"
doc_id = "sample3"

print("=" * 60)
print(f"DIAGNOSING QUERY: '{query}'")
print(f"DOCUMENT ID: '{doc_id}'")
print("=" * 60)

# Stage 1: Role mapping
print("\n--- STAGE 1: ROLE MAPPING ---")
rm = map_query_to_roles(query, mode="hybrid", debug=False)
print(f"Roles: {rm.roles}")
print(f"Kind: {rm.kind}")
print(f"Confidence: {rm.confidence}")
print(f"Reason: {rm.reason}")

# Stage 2: ChromaDB Retrieval at relax=0
print("\n--- STAGE 2: RETRIEVAL (relax=0) ---")
vstore = get_default_store()
chunks_r0 = vstore.retrieve(
    question=query,
    document_id=doc_id,
    mode="role",
    roles=rm.roles,
    relax=0,
    top_k=5,
)
print(f"Retrieved {len(chunks_r0)} chunks at relax=0.")
for i, c in enumerate(chunks_r0):
    meta = c.get("metadata", {})
    print(f"  Chunk {i+1} [idx={meta.get('chunk_idx')}, role={meta.get('role')}]: distance={c.get('distance'):.4f}")
    print(f"    Excerpt: {c.get('text', '')[:120]}...")

# Stage 3: Cross-Encoder Scoring
print("\n--- STAGE 3: CROSS-ENCODER SCORING ---")
scored, ce_ms = score_chunks(query, chunks_r0)
reranked = sorted(scored, key=lambda x: x.get("rerank_score", -99.0), reverse=True)
print(f"Scored in {ce_ms:.2f}ms. Router thresholds: TAU_HIGH={TAU_HIGH}, TAU_LOW={TAU_LOW}")
for i, c in enumerate(reranked):
    score = c.get("rerank_score", -99.0)
    print(f"  Reranked {i+1} [chunk_idx={c.get('chunk_idx')}, role={c.get('role')}]: rerank_score={score:.4f}")
top_score = reranked[0].get("rerank_score", -99.0) if reranked else -99.0
print(f"Top CE score: {top_score:.4f}")
if top_score >= TAU_HIGH:
    expected_route = "generate"
elif top_score >= TAU_LOW:
    expected_route = "fallback"
else:
    expected_route = "rewrite"
print(f"Cross-encoder route: {expected_route}")

# Stage 4: LLM Relevance Grading
print("\n--- STAGE 4: LLM RELEVANCE GRADING ---")
eval_chunks = reranked[:3] if reranked else chunks_r0[:3]
graded, grader_route, grade_ms = grade_chunks(query, eval_chunks, max_chars=800, use_cache=False)
print(f"Graded in {grade_ms:.2f}ms, Grader route: {grader_route}")
for i, g in enumerate(graded):
    print(f"  Graded {i+1}: grade={g.get('grade')}, conf={g.get('confidence')}, reason={g.get('reason')}")

# Stage 5: Full run_query execution
print("\n--- STAGE 5: FULL RUN_QUERY EXECUTION ---")
result = run_query(question=query, document_id=doc_id, mode="role")
print(f"Final status: {result.get('final_status')}")
print(f"Generation: {result.get('generation')}")
print(f"Retry count: {result.get('retry_count')}")
print(f"Relax level: {result.get('relax_level')}")
print(f"External context (web): {result.get('external_context')}")
print(f"Verified context count: {len(result.get('verified_context', []))}")
print(f"Trace events count: {len(result.get('trace', []))}")
print("\nTrace Timeline:")
for t in result.get("trace", []):
    print(f"  Node: {t.get('node'):<20} Route: {t.get('route'):<15} Dur: {t.get('duration_ms'):>7.1f}ms Retry: {t.get('retry_count')} Relax: {t.get('relax_level')} Details: {json.dumps(t.get('details', {}))}")

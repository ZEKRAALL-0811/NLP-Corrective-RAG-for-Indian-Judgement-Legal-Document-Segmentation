import sys
from pathlib import Path
import json

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.role_mapper import map_query_to_roles
from src.rag.graph import run_query, get_default_store

queries = [
    "What did the appellant argue regarding the Contempt of Courts Act?",
    "What was the first point the Bench framed for consideration?",
    "On what grounds did the decree holder argue that the NCLT order of 22.06.2018 should not bind it?",
]

doc_id = "sample3"

print("=" * 80)
print("VALIDATING CANDIDATE SUPPORTED QUERIES ON SAMPLE3.PDF")
print("=" * 80)

for idx, q in enumerate(queries, 1):
    print(f"\n[{idx}/3] TESTING QUERY: '{q}'")
    print("-" * 70)
    
    # 1. Direct role mapping pre-check
    rm = map_query_to_roles(q, mode="hybrid")
    print(f"Role mapping: {rm.roles} | kind: {rm.kind} | confidence: {rm.confidence:.2f}")

    # 2. Production run_query
    result = run_query(question=q, document_id=doc_id, mode="role")
    
    final_status = result.get("final_status")
    retries = result.get("retry_count", 0)
    route = result.get("route", "")
    gen = result.get("generation", "")
    sources = result.get("sources", [])
    verified_context = result.get("verified_context", [])
    trace = result.get("trace", [])
    
    print(f"Final Status: {final_status}")
    print(f"Retries: {retries}")
    print(f"Final Route: {route}")
    print(f"Verified context count: {len(verified_context)}")
    print(f"Generation reached: {bool(gen and final_status != 'no_answer')}")
    print(f"Final Answer: {gen}")
    print(f"Citations count: {len(sources)}")
    for s_idx, s in enumerate(sources, 1):
        print(f"  Citation [{s.get('citation_index', s_idx)}]: chunk_id={s.get('chunk_id')}, role={s.get('role')}")
        print(f"    Text: {s.get('text', '')[:120]}...")
    
    print("\nDetailed Trace Summary:")
    for ev in trace:
        node = ev.get("node")
        ev_route = ev.get("route")
        dur = ev.get("duration_ms")
        details = ev.get("details", {})
        print(f"  Node: {node:<20} Route: {ev_route:<15} Dur: {dur:>6.1f}ms Details: {json.dumps(details)}")

import json
from src.rag.graph import run_query

query = "Under what section was the Contempt of Courts Act invoked?"
doc_id = "sample3"

print("Running query:", query)
res = run_query(query, document_id=doc_id, mode="role")

print("\n=== EXECUTION SUMMARY ===")
print("Final Status:", res.get("final_status"))
print("Final Answer:\n", res.get("generation"))
print("Citations:", res.get("citations"))
print("\nSources:")
for s in res.get("sources", []):
    print(f"  [{s.get('citation_num')}] Chunk: {s.get('chunk_id')} | Role: {s.get('role')} | Para: {s.get('first_para_marker')}")
    print(f"      Snippet: {s.get('snippet')[:140]}...")

print("\nTrace Steps:")
for t in res.get("trace", []):
    print(f"  - Node: {t['node']} (route={t.get('route')}, duration={t.get('duration_ms', 0):.1f}ms)")
    print(f"    Details: {json.dumps(t.get('details', {}), indent=6)}")

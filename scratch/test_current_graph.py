import json
from src.rag.graph import create_jurisgraph
from src.rag.role_mapper import map_query_to_roles

graph = create_jurisgraph()

q = "Who is the petitioner, and what specific Government Order (G.O.) and Rules are being challenged?"
init_state = {
    "question": q,
    "original_question": q,
    "document_id": "sampleA",
    "mode": "role",
    "retry_count": 0,
    "relax_level": 0,
    "llm_call_count": 0,
    "trace": [],
}

final_state = graph.invoke(init_state)
print("Route:", final_state.get("route"))
print("Final Status:", final_state.get("final_status"))
print("Generation:", final_state.get("generation"))
print("Citations:", final_state.get("citations"))
print("Sources count:", len(final_state.get("sources", [])))
for s in final_state.get("sources", []):
    print("  Source:", s.get("citation_num"), s.get("chunk_id"), s.get("role"), repr(s.get("snippet", "")[:80]))

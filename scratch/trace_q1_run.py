import sys
sys.path.insert(0, ".")
from src.rag.graph import run_query

q1 = "What were K. Bharathi’s principal arguments concerning the alleged wilful disobedience of court orders and the disputed mortgage transactions?"
res = run_query(question=q1, document_id="sample3", mode="role")

print("Mapped roles:", res.get("mapped_roles"))
print("Final status:", res.get("final_status"))
print("Answer:", res.get("generation"))
print("Citations:", res.get("citations"))
print("\nVerified Context:")
for c in res.get("verified_context", []):
    print(" ", c.get("id") or c.get("chunk_id"), c.get("grade"), ":", c.get("text", "")[:120])

print("\nTrace:")
for t in res.get("trace", []):
    print(" ", t.get("node"), t.get("route"), t.get("details"))

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path("d:/NLP Project")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.role_mapper import map_query_to_roles, RoleMapping
from src.rag.graph import run_query

print("=" * 60)
print("PHASE 2: OUT-OF-SCOPE QUERY INTERCEPTION")
print("=" * 60)

query = "What is the weather forecast for tomorrow?"
doc_id = "sample3"

# 1. Test role mapper classification
rm = map_query_to_roles(query, mode="hybrid")
print(f"Role Mapper output for '{query}':")
print(f"  Kind: {rm.kind}")
print(f"  Confidence: {rm.confidence}")
print(f"  Roles: {rm.roles}")
assert rm.kind == "out_of_scope"
assert len(rm.roles) == 0

# 2. Test in graph execution with mocks
with patch("src.rag.graph.retrieve_node") as mock_retrieve, \
     patch("src.rag.graph.generate_node") as mock_generate:
    
    result = run_query(question=query, document_id=doc_id, mode="role")
    
    print("\nGraph execution output:")
    print(f"  Final status: {result.get('final_status')}")
    print(f"  Role kind: {result.get('role_kind')}")
    print(f"  Generation: '{result.get('generation')}'")
    print(f"  Retrieve node called: {mock_retrieve.called}")
    print(f"  Generate node called: {mock_generate.called}")
    print(f"  Trace nodes: {[t.get('node') for t in result.get('trace', [])]}")
    
    assert result.get("final_status") == "no_answer"
    assert result.get("role_kind") == "out_of_scope"
    assert "outside the scope" in result.get("generation")
    assert mock_retrieve.called is False, "retrieve_node was invoked for out-of-scope query!"
    assert mock_generate.called is False, "generate_node was invoked for out-of-scope query!"

# 3. Test UI state logic for out-of-scope
from streamlit.testing.v1 import AppTest
APP_PATH = str(project_root / "app.py")

with patch("src.rag.role_mapper.check_ollama", return_value=True), \
     patch("src.rag.graph.run_query") as mock_app_run_query:
    
    at = AppTest.from_file(APP_PATH, default_timeout=60.0).run()
    at.session_state["doc_id"] = "sample3"
    at.session_state["doc_name"] = "sample3.pdf"
    at.run()
    at.text_input[0].input(query)
    at.button[0].click().run()
    
    rendered = "".join(m.value for m in at.markdown)
    print("\nUI Behavior for out-of-scope query:")
    print(f"  run_query called: {mock_app_run_query.called}")
    print(f"  OUT OF LEGAL SCOPE chip displayed: {'OUT OF LEGAL SCOPE' in rendered}")
    print(f"  Out of scope message displayed: {'outside the scope of Indian legal judgments' in rendered}")
    
    assert mock_app_run_query.called is False
    assert "OUT OF LEGAL SCOPE" in rendered
    assert "outside the scope of Indian legal judgments" in rendered

print("\nPhase 2: ALL ASSERTIONS PASSED (Retrieval and Generation strictly bypassed).")

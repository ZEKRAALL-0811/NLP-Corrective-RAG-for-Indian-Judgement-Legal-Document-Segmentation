"""
Display a single row from artifacts/logs/generation_results.jsonl by ID.
Usage:
    python scripts/show_row.py <id>
Example:
    python scripts/show_row.py gold_1
"""

import sys
import json
from pathlib import Path

def show_row(row_id: str, jsonl_path: Path = Path("artifacts/logs/generation_results.jsonl")):
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist.")
        sys.exit(1)

    target_id = row_id.strip()
    found = False

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id") == target_id:
                found = True
                print("=" * 80)
                print(f"ROW ID: {row.get('id')} | Sample: {row.get('sample')} | Status: {row.get('final_status')}")
                print("=" * 80)
                print(f"Question: {row.get('question')}")
                print(f"Gold Role: {row.get('gold_role')} | Gold Markers: {row.get('gold_markers')}")
                print(f"Timestamp: {row.get('timestamp')}")
                print(f"Models Used: {row.get('models_used')}")
                print(f"Answer: {row.get('answer')}")
                print(f"Citations: {row.get('citations')}")
                print(f"Ungrounded Numbers: {row.get('ungrounded_numbers')}")
                print(f"Flags: {row.get('flags')}")
                print(f"Per-Node Latency (ms): {row.get('per_node_latency')}")
                print(f"Ollama Total Duration: {row.get('ollama_total_duration')}")
                print(f"Context Chunks ({len(row.get('context_chunks', []))}):")
                for c in row.get("context_chunks", []):
                    cid = c.get("chunk_id", "N/A")
                    role = c.get("role", "Unknown")
                    txt = c.get("text", "").replace("\n", " ")
                    snippet = txt[:120] + "..." if len(txt) > 120 else txt
                    print(f"  * [{cid}] ({role}): {snippet}")
                print("=" * 80)
                break

    if not found:
        print(f"Row with ID '{target_id}' not found in {jsonl_path}.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_row.py <id>")
        sys.exit(1)
    show_row(sys.argv[1])

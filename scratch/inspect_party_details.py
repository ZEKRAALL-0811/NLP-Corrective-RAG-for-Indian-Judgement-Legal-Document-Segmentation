import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from src.models.segmenter import segment_document

records, stats = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)
pts = [r for r in records if r.get("override") == "party_details"]
print(f"Found {len(pts)} party_details overrides:")
for r in pts:
    print(f"Index {r['sentence_idx']}:")
    print(f"  Role: {r['role']} (Raw: {r['raw_role']})")
    print(f"  Text: {repr(r['text'])}")
    print()

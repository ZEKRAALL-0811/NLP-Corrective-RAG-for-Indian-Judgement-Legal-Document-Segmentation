import sys
sys.path.insert(0, ".")
import re
import json
from pathlib import Path
from src.models.segmenter import segment_document

recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

text2 = " ".join(r["text"] for r in recs2)
text3 = " ".join(r["text"] for r in recs3)

gold_json = json.loads(Path("legalDataset/samples/gold_paragraphs.json").read_text(encoding="utf-8"))

print(f"Sample 2 text length: {len(text2)}")
print(f"Sample 3 text length: {len(text3)}")

# Test marker matching for each question in gold_paragraphs.json
for q, info in gold_json.items():
    doc_id = info["sample"]
    markers = info["markers"]
    doc_text = text2 if doc_id == "sample2" else text3
    print(f"\n[{doc_id}] Q: {q[:50]}...")
    print(f"  Markers: {markers}")
    for m in markers:
        # Find occurrences of marker
        m_str = str(m).strip()
        if m_str.startswith("(") and m_str.endswith(")"):
            num = m_str[1:-1]
            pat = re.compile(r"(?:^|\s)(?:\(" + re.escape(num) + r"\)|\b" + re.escape(num) + r"\.)(?!\d)")
        else:
            pat = re.compile(r"(?:^|\s)(?:" + re.escape(m_str) + r"[\.\)\s])(?!\d{1,2}\.\d{4})")
        matches = list(pat.finditer(doc_text))
        print(f"    Marker '{m}': found {len(matches)} matches at {[match.start() for match in matches[:3]]}")

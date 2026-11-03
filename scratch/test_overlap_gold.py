import sys
import json
import re
from pathlib import Path
sys.path.insert(0, ".")
from src.models.segmenter import segment_document

paras2 = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
paras3 = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

print(f"Sample 2 paras: {len(paras2)}, Sample 3 paras: {len(paras3)}")

gold_paras = json.loads(Path("legalDataset/samples/gold_paragraphs.json").read_text(encoding="utf-8"))
print(f"Gold paras questions: {len(gold_paras)}")

# Map markers to text
map2 = {p.marker: p.text for p in paras2}
map3 = {p.marker: p.text for p in paras3}

# Check how markers in gold_paras match
matched = 0
for q, data in gold_paras.items():
    doc = data.get("sample")
    markers = data.get("markers", [])
    m_map = map2 if doc == "sample2" else map3
    found = [m for m in markers if m in m_map]
    if len(found) == len(markers):
        matched += 1
    else:
        print(f"Unmatched markers for {doc}: {markers} vs found {found}")

print(f"Total matched questions markers: {matched}/{len(gold_paras)}")

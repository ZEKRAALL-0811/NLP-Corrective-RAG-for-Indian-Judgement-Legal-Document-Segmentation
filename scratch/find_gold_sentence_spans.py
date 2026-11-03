import sys
sys.path.insert(0, ".")
import json
import re
from pathlib import Path
from src.models.segmenter import segment_document

recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

# Build sentence character spans in joined full_text
def get_sentence_char_spans(records):
    spans = []
    curr = 0
    full_text_parts = []
    for r in records:
        text = r["text"]
        full_text_parts.append(text)
        start = curr
        end = curr + len(text)
        spans.append((start, end))
        curr = end + 1 # account for space joining
    return spans, " ".join(full_text_parts)

spans2, text2 = get_sentence_char_spans(recs2)
spans3, text3 = get_sentence_char_spans(recs3)

gold_json = json.loads(Path("legalDataset/samples/gold_paragraphs.json").read_text(encoding="utf-8"))

print("MAPPING GOLD PARAGRAPHS TO CHARACTER SPANS:")
for q, info in gold_json.items():
    doc_id = info["sample"]
    recs = recs2 if doc_id == "sample2" else recs3
    spans = spans2 if doc_id == "sample2" else spans3
    full_text = text2 if doc_id == "sample2" else text3
    markers = info["markers"]

    # Find starting sentences for the markers
    marker_start_sentences = []
    for m in markers:
        m_str = str(m).strip()
        found_idx = None
        for s_idx, r in enumerate(recs):
            t = r["text"].strip()
            # Check if sentence starts with the marker
            if m_str.startswith("(") and m_str.endswith(")"):
                num = m_str[1:-1]
                if re.match(r"^\(" + re.escape(num) + r"\)", t) or re.match(r"^" + re.escape(num) + r"\.", t):
                    found_idx = s_idx
                    break
            else:
                if re.match(r"^" + re.escape(m_str) + r"[\.\)\s]", t) or t.startswith(m_str + " "):
                    found_idx = s_idx
                    break
                # Special cases like 7(g), 7(b), 11.1, etc.
                if m_str in ["7(g)", "7(b)"] and (t.startswith(m_str) or re.search(r"\b" + re.escape(m_str) + r"\b", t)):
                    found_idx = s_idx
                    break
                if m_str == "6" and re.match(r"^6[\.\)]", t):
                    found_idx = s_idx
                    break
                if m_str == "7" and re.match(r"^7[\.\)]", t):
                    found_idx = s_idx
                    break

        if found_idx is not None:
            marker_start_sentences.append((m, found_idx))
        else:
            # Fallback: search anywhere in full text
            print(f"  [WARN] Marker '{m}' not found at sentence start for Q: {q[:30]}...")

    print(f"[{doc_id}] Q: {q[:40]}... -> Markers: {marker_start_sentences}")

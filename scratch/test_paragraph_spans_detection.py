import sys
sys.path.insert(0, ".")
import json
import re
from pathlib import Path
from src.models.segmenter import segment_document

recs2, _ = segment_document("legalDataset/samples/sample2.pdf", use_cache=True)
recs3, _ = segment_document("legalDataset/samples/sample3.pdf", use_cache=True)

def build_doc_spans(records):
    sent_spans = []
    curr = 0
    full_text_parts = []
    for r in records:
        text = r["text"]
        full_text_parts.append(text)
        start = curr
        end = curr + len(text)
        sent_spans.append((start, end))
        curr = end + 1
    full_text = " ".join(full_text_parts)
    return sent_spans, full_text

sent_spans2, text2 = build_doc_spans(recs2)
sent_spans3, text3 = build_doc_spans(recs3)

# Build map of paragraph marker -> character span in doc
def find_all_paragraphs_sample2(records, sent_spans, full_text):
    para_starts = {}
    for idx, r in enumerate(records):
        t = r["text"].strip()
        m = re.match(r"^\((\d+)\)", t)
        if m:
            p_num = int(m.group(1))
            if p_num not in para_starts:
                para_starts[p_num] = (idx, sent_spans[idx][0])
    sorted_nums = sorted(para_starts.keys())
    para_spans = {}
    for i, num in enumerate(sorted_nums):
        start_char = para_starts[num][1]
        if i + 1 < len(sorted_nums):
            next_num = sorted_nums[i + 1]
            end_char = para_starts[next_num][1]
        else:
            end_char = len(full_text)
        para_spans[f"({num})"] = (start_char, end_char)
    return para_spans

def find_all_paragraphs_sample3(records, sent_spans, full_text):
    # Known paragraphs in sample3: 1, 2, 3.1..3.8, 4..28
    markers = [
        "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8",
        "4", "5", "6", "7", "8", "9", "10", "11.1", "11.2", "11.3", "11.4",
        "11.5", "11.6", "11.7", "11.8", "11.9", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"
    ]
    para_starts = {}
    for m in markers:
        # Search at start of sentence
        found_idx = None
        for idx, r in enumerate(records):
            t = r["text"].strip()
            if re.match(r"^" + re.escape(m) + r"[\.\)\s]", t):
                found_idx = idx
                break
        if found_idx is not None:
            para_starts[m] = (found_idx, sent_spans[found_idx][0])
    
    # Sort markers by character start
    sorted_markers = sorted(para_starts.keys(), key=lambda k: para_starts[k][1])
    para_spans = {}
    for i, m in enumerate(sorted_markers):
        start_char = para_starts[m][1]
        if i + 1 < len(sorted_markers):
            next_m = sorted_markers[i + 1]
            end_char = para_starts[next_m][1]
        else:
            end_char = len(full_text)
        para_spans[m] = (start_char, end_char)
    return para_spans

p2 = find_all_paragraphs_sample2(recs2, sent_spans2, text2)
p3 = find_all_paragraphs_sample3(recs3, sent_spans3, text3)

print(f"Sample 2 found {len(p2)} paragraphs: {sorted(p2.keys())[:10]}...")
print(f"Sample 3 found {len(p3)} paragraphs: {sorted(p3.keys())[:10]}...")

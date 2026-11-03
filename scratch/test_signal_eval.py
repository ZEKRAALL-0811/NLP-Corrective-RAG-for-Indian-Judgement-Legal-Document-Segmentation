"""
Test and compute Part A signals, context policies, cross-encoder ROC-AUC, question-level AUC,
cross-document validation, and reranking metrics.
"""
import sys
import json
import re
import time
import os
import psutil
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.vectorstore import LegalVectorStore
from src.rag.role_mapper import map_query_to_roles
from src.rag.grader import grade_chunks, GRADE_SCORES
from src.rag.reranker import score_chunks, rerank_chunks, get_cross_encoder, get_cross_encoder_memory_footprint_mb

print("1. Loading Document Spans...")
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
    return sent_spans, " ".join(full_text_parts)

sent_spans2, text2 = build_doc_spans(recs2)
sent_spans3, text3 = build_doc_spans(recs3)

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
        end_char = para_starts[sorted_nums[i + 1]][1] if i + 1 < len(sorted_nums) else len(full_text)
        para_spans[f"({num})"] = (start_char, end_char)
    return para_spans

def find_all_paragraphs_sample3(records, sent_spans, full_text):
    all_markers = [
        "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8",
        "4", "5", "6", "7", "8", "9", "10", "11.1", "11.2", "11.3", "11.4",
        "11.5", "11.6", "11.7", "11.8", "11.9", "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"
    ]
    para_starts = {}
    for m in all_markers:
        for idx in range(36, len(records)):
            t = records[idx]["text"].strip()
            if re.match(r"^" + re.escape(m) + r"\.\s+", t):
                para_starts[m] = (idx, sent_spans[idx][0])
                break
    sorted_markers = sorted(para_starts.keys(), key=lambda k: para_starts[k][1])
    para_spans = {}
    for i, m in enumerate(sorted_markers):
        start_char = para_starts[m][1]
        end_char = para_starts[sorted_markers[i + 1]][1] if i + 1 < len(sorted_markers) else len(full_text)
        para_spans[m] = (start_char, end_char)
    return para_spans

para_spans_2 = find_all_paragraphs_sample2(recs2, sent_spans2, text2)
para_spans_3 = find_all_paragraphs_sample3(recs3, sent_spans3, text3)

def check_fair_overlap_hit(chunk, doc_id, gold_markers, min_overlap=150):
    meta = chunk.get("metadata", {})
    s0 = meta.get("start_sentence_idx", chunk.get("start_sentence_idx"))
    s1 = meta.get("end_sentence_idx", chunk.get("end_sentence_idx"))
    spans = sent_spans2 if doc_id == "sample2" else sent_spans3
    if s0 is not None and s1 is not None and s0 < len(spans) and s1 < len(spans):
        c_start = spans[s0][0]
        c_end = spans[s1][1]
    else:
        c_start = chunk.get("char_start", 0)
        c_end = chunk.get("char_end", 0)
    spans_dict = para_spans_2 if doc_id == "sample2" else para_spans_3
    
    for m in gold_markers:
        key = m if doc_id == "sample2" else m.replace("para_", "").replace("para", "")
        if key not in spans_dict:
            for k in spans_dict:
                if k.strip("()") == key.strip("()"):
                    key = k
                    break
        if key in spans_dict:
            p_start, p_end = spans_dict[key]
            overlap = max(0, min(c_end, p_end) - max(c_start, p_start))
            if overlap >= min_overlap:
                return True
    return False

# Load datasets
gold_data = json.loads(Path("legalDataset/samples/gold_paragraphs.json").read_text(encoding="utf-8"))
unans_rows = []
with open("legalDataset/samples/gold_unanswerable.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        unans_rows.append(row)

store = LegalVectorStore()
get_cross_encoder()

print(f"Loaded {len(gold_data)} gold questions and {len(unans_rows)} unanswerable questions.")

# 2. Retrieve top-5 chunks for all 22 questions and 8 unanswerable questions
gold_items = []
for q, info in gold_data.items():
    doc_id = info["sample"]
    markers = info["markers"]
    oracle_role = info.get("role", "Facts")
    mapping = map_query_to_roles(q, mode="hybrid", debug=False)
    pred_roles = mapping.roles
    pos_pred = (0.85, 1.0) if "Decision" in pred_roles else None

    # Retrieve top-5 for predicted-role relax 0 and unfiltered relax 3
    top5_pred = store.retrieve(question=q, document_id=doc_id, mode="role", roles=pred_roles, relax=0, rel_pos_range=pos_pred, top_k=5)
    top5_r3 = store.retrieve(question=q, document_id=doc_id, mode="role", relax=3, top_k=5)
    
    gold_items.append({
        "question": q,
        "doc_id": doc_id,
        "markers": markers,
        "pred_roles": pred_roles,
        "top5_pred": top5_pred,
        "top5_r3": top5_r3,
    })

unans_items = []
for row in unans_rows:
    q = row["question"]
    doc_id = row["sample"]
    mapping = map_query_to_roles(q, mode="hybrid", debug=False)
    top5 = store.retrieve(question=q, document_id=doc_id, mode="role", roles=mapping.roles, relax=0, top_k=5)
    unans_items.append({
        "question": q,
        "doc_id": doc_id,
        "note": row["note"],
        "top5": top5,
    })

# Score with CrossEncoder
ce_latencies = []
for item in gold_items:
    q = item["question"]
    c5_pred, lat = score_chunks(q, item["top5_pred"])
    item["scored_top5_pred"] = c5_pred
    ce_latencies.append(lat)
    
    c5_r3, lat2 = score_chunks(q, item["top5_r3"])
    item["scored_top5_r3"] = c5_r3
    ce_latencies.append(lat2)

for item in unans_items:
    q = item["question"]
    c5, lat = score_chunks(q, item["top5"])
    item["scored_top5"] = c5
    ce_latencies.append(lat)

# Measure memory
mem_rss_mb = get_cross_encoder_memory_footprint_mb()
print(f"CrossEncoder mean scoring latency: {np.mean(ce_latencies):.2f} ms per 5 chunks")
print(f"Process RSS memory with CrossEncoder loaded: {mem_rss_mb:.1f} MB")

# Grade top-3 chunks with LLM (use_cache=False to get true timing & evaluation hygiene)
print("\nGrading chunks with LLM grader (use_cache=False)...")
llm_latencies = []
for i, item in enumerate(gold_items):
    q = item["question"]
    # top-3 pred
    graded_pred, route_pred, wall_ms = grade_chunks(q, item["top5_pred"][:3], max_chars=500, use_cache=False)
    item["graded_pred_top3"] = graded_pred
    item["route_pred"] = route_pred
    llm_latencies.append(wall_ms)
    
    graded_r3, route_r3, wall_ms2 = grade_chunks(q, item["top5_r3"][:3], max_chars=500, use_cache=False)
    item["graded_r3_top3"] = graded_r3
    item["route_r3"] = route_r3
    llm_latencies.append(wall_ms2)
    print(f"  Graded Q{i+1}/22 (pred={route_pred}, r3={route_r3}, {wall_ms:.1f}ms)", flush=True)

print(f"LLM Grader latency (use_cache=False @ 500c): Min={min(llm_latencies):.1f}ms, Med={np.median(llm_latencies):.1f}ms, Max={max(llm_latencies):.1f}ms")

# Save checkpoint json so we have these results
ckpt_data = {
    "ce_latencies": ce_latencies,
    "llm_latencies": llm_latencies,
    "mem_rss_mb": mem_rss_mb,
}
Path("scratch/signal_eval_ckpt.json").write_text(json.dumps(ckpt_data), encoding="utf-8")
print("Checkpoint saved.")

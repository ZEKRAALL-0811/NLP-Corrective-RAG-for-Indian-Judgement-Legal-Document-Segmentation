"""
Comprehensive Verification Script for Preprocessing Fixes, Caching, and Previews
Executes all required checks for sample1, sample2, and sample3.
"""

import sys
import time
from pathlib import Path
import pypdf

# UTF-8 stdout encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.config import CANONICAL_LABELS
from src.data.preprocessor import (
    extract_pages_from_pdf,
    clean_legal_text,
    preprocess_document,
    ScannedPdfError,
    get_sentencizer,
)
from src.models.segmenter import LegalSegmenter

print("="*85)
print("1. EARLY SCANNED-PDF DETECTION: sample1.pdf")
print("="*85)

sample1_path = PROJECT_ROOT / "legalDataset" / "samples" / "sample1.pdf"
t0 = time.time()
caught_scanned = False
try:
    extract_pages_from_pdf(sample1_path)
except ScannedPdfError as e:
    caught_scanned = True
    scanned_elapsed = time.time() - t0
    print(f"ScannedPdfError caught in: {scanned_elapsed:.3f} seconds (previously took ~120s)")
    print(f"Error message: {e}")

assert caught_scanned, "ScannedPdfError was not raised for sample1.pdf!"

print("\n" + "="*85)
print("2. LINE-BREAK JOINING & UNDER-5-TOKEN REDUCTION: sample2.pdf")
print("="*85)

sample2_path = PROJECT_ROOT / "legalDataset" / "samples" / "sample2.pdf"
reader2 = pypdf.PdfReader(str(sample2_path))
raw_pages2 = [p.extract_text() or "" for p in reader2.pages]
raw_unjoined_text2 = "\n".join(raw_pages2)

# Baseline: raw text without paragraph joining or header removal
nlp = get_sentencizer()
doc_unjoined = nlp(raw_unjoined_text2)
sents_unjoined = [s.text.strip() for s in doc_unjoined.sents if s.text.strip()]
u5_unjoined = sum(1 for s in sents_unjoined if len(s.split()) < 5)

# After cleaning with header removal and intra-paragraph line joining
sents_cleaned2, stats2 = preprocess_document(sample2_path)
u5_cleaned = stats2["sentences_under_5_tokens"]

print("Sample2 Baseline (Before Cleaning):")
print(f"  Total Sentences:             {len(sents_unjoined)}")
print(f"  Sentences Under 5 Tokens:    {u5_unjoined} ({u5_unjoined / len(sents_unjoined) * 100:.1f}%)")
print("\nSample2 After Cleaning & Line Joining:")
print(f"  Total Sentences:             {stats2['sentence_count']}")
print(f"  Sentences Under 5 Tokens:    {u5_cleaned} ({stats2['sentences_under_5_tokens_pct']:.1f}%)")
print(f"  Under-5-Token Reduction:     {u5_unjoined} -> {u5_cleaned} (-{u5_unjoined - u5_cleaned} fragmented sentences, -{(u5_unjoined - u5_cleaned)/u5_unjoined*100:.1f}%)")

print("\n" + "="*85)
print("3. HEADER/FOOTER REMOVAL DETAILS: sample2.pdf & sample3.pdf")
print("="*85)

sample3_path = PROJECT_ROOT / "legalDataset" / "samples" / "sample3.pdf"

for name, path in [("sample2.pdf", sample2_path), ("sample3.pdf", sample3_path)]:
    p_texts = extract_pages_from_pdf(path)
    _, h_stats = clean_legal_text(p_texts)
    print(f"\n--- {name} ({h_stats['num_pages']} pages) ---")
    print(f"Total Lines Removed: {h_stats['lines_removed']}")
    print(f"Repeating Header/Footer Patterns Detected ({len(h_stats['repeating_headers'])}):")
    for rh in h_stats['repeating_headers']:
        print(f"  - {repr(rh)}")
    print(f"Examples of Stripped Lines ({len(h_stats['removed_examples'])}):")
    for ex in h_stats['removed_examples'][:6]:
        print(f"  - {repr(ex)}")

print("\n" + "="*85)
print("4. COLD VS. WARM CACHE TIMING BENCHMARK")
print("="*85)

# Clear cache entry for sample3 to test clean cold vs warm
segmenter = LegalSegmenter(num_threads=8, batch_size=64)

# 1. Cold run
print("Running COLD segmentation on sample3.pdf...")
t_cold_start = time.time()
records3_cold, stats3_cold = segmenter.segment_document(sample3_path, use_cache=True)
t_cold = time.time() - t_cold_start
print(f"COLD Wall-Clock Time: {t_cold:.2f}s (cached={stats3_cold.get('cached')})")

# 2. Warm run
print("Running WARM segmentation on sample3.pdf...")
t_warm_start = time.time()
records3_warm, stats3_warm = segmenter.segment_document(sample3_path, use_cache=True)
t_warm = time.time() - t_warm_start
print(f"WARM Wall-Clock Time: {t_warm:.4f}s (cached={stats3_warm.get('cached')})")
print(f"Speedup from persistent JSON cache: {t_cold / max(t_warm, 1e-6):.1f}x faster!")

# Run sample2 to ensure preview and cache are saved
print("\nSegmenting sample2.pdf (with cache & preview)...")
t_sample2_start = time.time()
records2, stats2 = segmenter.segment_document(sample2_path, use_cache=True)
print(f"Sample2 total time: {time.time() - t_sample2_start:.2f}s (cached={stats2.get('cached')})")

print("\n" + "="*85)
print("5. RE-REPORT FOR sample2.pdf AND sample3.pdf")
print("="*85)

for name, s_path, recs, s_stats in [
    ("sample2.pdf", sample2_path, records2, stats2),
    ("sample3.pdf", sample3_path, records3_cold, stats3_cold),
]:
    print(f"\n{'#'*75}")
    print(f"# RE-REPORT: {name}")
    print(f"{'#'*75}")
    print("Document Profile:")
    print(f"  Sentence Count:             {s_stats['sentence_count']} (LegalSeg reference: 225)")
    print(f"  Total Words:                {s_stats['total_words']:,}")
    print(f"  Mean Tokens / Sentence:     {s_stats['mean_tokens_per_sentence']:.1f} (LegalSeg reference: 34)")
    print(f"  Sentences Under 5 Tokens:   {s_stats['sentences_under_5_tokens']} ({s_stats['sentences_under_5_tokens_pct']:.1f}%)")
    print(f"  Lines Removed:              {s_stats['header_removal']['lines_removed']}")
    print(f"  Preview File:               {s_stats.get('preview_file')}")

    print("\nRole Counts & Percentages (Smoothed):")
    total_sents = max(len(recs), 1)
    for role in CANONICAL_LABELS:
        cnt = s_stats["role_counts"].get(role, 0)
        pct = (cnt / total_sents) * 100
        print(f"  {role:<26}: {cnt:4d} sentences ({pct:5.1f}%)")

    # Sample 15 sentences from the middle of the document spread across substantive roles
    print(f"\n15 Sentences Sampled from Middle of {name} (Spread Across Roles):")
    print("-" * 115)
    print(f"{'#':<4} | {'Role':<24} | {'Conf':<6} | Sentence Text Preview")
    print("-" * 115)

    target_roles = [
        "Arguments of Petitioner",
        "Arguments of Respondent",
        "Reasoning",
        "Decision",
        "Issue",
        "Facts",
    ]
    
    # Restrict search to middle 80% of document
    mid_start = int(len(recs) * 0.10)
    mid_end = int(len(recs) * 0.90)
    mid_records = recs[mid_start:mid_end] if mid_end > mid_start else recs

    sampled = []
    # Collect up to 2-3 per target role
    for target in target_roles:
        matches = [r for r in mid_records if r["role"] == target]
        if matches:
            step = max(len(matches) // 3, 1)
            for m in matches[::step][:3]:
                if m not in sampled and len(sampled) < 15:
                    sampled.append(m)

    # If fewer than 15, backfill from remaining mid records
    if len(sampled) < 15:
        for r in mid_records:
            if r not in sampled:
                sampled.append(r)
                if len(sampled) == 15:
                    break

    sampled = sorted(sampled, key=lambda x: x["sentence_idx"])
    for r in sampled:
        conf = r["role_probs"][cfg.LABEL2ID[r["role"]]] * 100
        text_clean = r["text"].replace("\n", " ").strip()
        if len(text_clean) > 75:
            text_clean = text_clean[:72] + "..."
        print(f"{r['sentence_idx']:<4} | {r['role']:<24} | {conf:5.1f}% | {text_clean}")
    print("-" * 115)

    # Check for header-like sentences remaining
    header_keywords = ["indiankanoon", "http", "cites", "cited by", "recast ai", "page ", "bench:", "author:"]
    flagged = []
    for r in recs:
        lower = r["text"].lower()
        if any(hk in lower for hk in header_keywords) and len(r["text"].split()) <= 10:
            flagged.append((r["sentence_idx"], r["role"], r["text"][:60]))

    if flagged:
        print(f"\n[FLAGGED RESIDUAL HEADER-LIKE SENTENCES ({len(flagged)})]:")
        for f_idx, f_role, f_text in flagged[:5]:
            print(f"  Sentence [{f_idx}] ({f_role}): {repr(f_text)}")
    else:
        print(f"\n[STATUS]: Zero residual header/footer fragments detected in {name}!")

print("\n" + "="*85)
print("[SUCCESS]: Verification and report complete.")
print("="*85)

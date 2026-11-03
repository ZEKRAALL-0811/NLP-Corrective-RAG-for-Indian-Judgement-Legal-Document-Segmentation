"""
Evaluation script for Part A fixes:
1. Cause-title zone rule & extended party details on sample2 and sample3.
2. Reports:
   - Zone length and overridden count per sample
   - The same 10 random sample3 Facts sentences (seed 42)
   - Sample2 role counts before and after
"""

import sys
from pathlib import Path
import random

sys.path.insert(0, ".")
from src.models.segmenter import segment_document

# Clear cache before running
cache_dir = Path("artifacts/cache")
if cache_dir.exists():
    for f in cache_dir.glob("*.json"):
        f.unlink()

print("="*80)
print("PART A: EVALUATING CAUSE-TITLE ZONE AND ADDRESS PATTERNS")
print("="*80)

# 1. Sample 2 Evaluation
rec2, stats2 = segment_document("legalDataset/samples/sample2.pdf")
zone_overrides_s2 = [r for r in rec2 if r.get("override") == "cause_title"]
party_overrides_s2 = [r for r in rec2 if r.get("override") == "party_details"]
total_overrides_s2 = [r for r in rec2 if "override" in r]

print(f"\n--- SAMPLE 2 (Total Sentences: {len(rec2)}) ---")
print(f"Role counts AFTER: {stats2['role_counts']}")
print(f"Cause-title overrides: {len(zone_overrides_s2)}")
print(f"Party-details overrides: {len(party_overrides_s2)}")
print(f"Total overrides: {len(total_overrides_s2)}")

# 2. Sample 3 Evaluation
rec3, stats3 = segment_document("legalDataset/samples/sample3.pdf")
zone_overrides_s3 = [r for r in rec3 if r.get("override") == "cause_title"]
party_overrides_s3 = [r for r in rec3 if r.get("override") == "party_details"]
total_overrides_s3 = [r for r in rec3 if "override" in r]

print(f"\n--- SAMPLE 3 (Total Sentences: {len(rec3)}) ---")
print(f"Role counts AFTER: {stats3['role_counts']}")
print(f"Cause-title overrides: {len(zone_overrides_s3)}")
print(f"Party-details overrides: {len(party_overrides_s3)}")
print(f"Total overrides: {len(total_overrides_s3)}")

# 3. Check the same 10 random sample3 Facts sentences (seed 42)
# In sample3 BEFORE zone fix, Facts had 115 sentences and seed 42 picked:
# [286, 22, 6, 299, 46, 42, 39, 25, 19, 291]
# Let's inspect these exact 10 sentences now:
test_indices = [6, 19, 22, 25, 39, 42, 46, 286, 291, 299]
print("\n--- SAME 10 SAMPLE 3 SENTENCES (originally Facts under seed 42) ---")
for idx in test_indices:
    r = rec3[idx]
    words = len(r["text"].split())
    print(f"[{idx}] Role: {r['role']:<10} | Override: {str(r.get('override')):<12} | Words: {words:<3} | Text: {r['text'][:70]}...")

# Also check new random sample of 10 Facts sentences from current sample3 with seed 42
current_facts_indices = [r["sentence_idx"] for r in rec3 if r["role"] == "Facts"]
random.seed(42)
new_sampled_facts = random.sample(current_facts_indices, min(10, len(current_facts_indices)))
print(f"\n--- NEW 10 RANDOM SAMPLE 3 FACTS SENTENCES (Seed 42, Total Facts: {len(current_facts_indices)}) ---")
for idx in sorted(new_sampled_facts):
    r = rec3[idx]
    print(f"[{idx}] Words: {len(r['text'].split()):<3} | Text: {r['text'][:80]}...")

import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document

# Force re-segmenting sample3
recs3, stats3 = segment_document("legalDataset/samples/sample3.pdf", use_cache=False)
print("SAMPLE 3 RESULTS:")
print(f"Total sentences: {len(recs3)}")
print(f"Role counts: {stats3['role_counts']}")
overrides = [r for r in recs3 if "override" in r]
print(f"Total overrides: {len(overrides)}")
ct_overrides = [r for r in recs3 if r.get("override") == "cause_title"]
pd_overrides = [r for r in recs3 if r.get("override") == "party_details"]
print(f"Cause title overrides: {len(ct_overrides)}")
print(f"Party details overrides: {len(pd_overrides)}")

# Check zone sentences
zone_end = None
for idx, r in enumerate(recs3):
    if r.get("override") != "cause_title" and zone_end is None:
        zone_end = idx

print(f"Zone end detected at index: {zone_end}")
zone_facts = [r for r in recs3[:zone_end] if r["role"] == "Facts"]
print(f"Zone sentences remaining Facts: {len(zone_facts)}")
for r in zone_facts:
    print(f"  [{r['sentence_idx']}] {r['text'][:80]}...")

print("\nChecking sample 2 (role counts must be unchanged):")
recs2, stats2 = segment_document("legalDataset/samples/sample2.pdf", use_cache=False)
print(f"Total sentences: {len(recs2)}")
print(f"Role counts: {stats2['role_counts']}")

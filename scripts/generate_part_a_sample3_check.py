"""
Script to inspect sample3.pdf sentences 0-40, 10 random Facts sentences (seed 42),
and override counts, saving raw output to artifacts/logs/part_a_sample3_check.txt.
"""

import sys
from pathlib import Path
import random

sys.path.insert(0, ".")
from src.models.segmenter import segment_document

# Clear cache before running to ensure clean run
cache_dir = Path("artifacts/cache")
if cache_dir.exists():
    for f in cache_dir.glob("*.json"):
        f.unlink()

out_path = Path("artifacts/logs/part_a_sample3_check.txt")
rec3, stats3 = segment_document("legalDataset/samples/sample3.pdf")

zone_overrides = [r for r in rec3 if r.get("override") == "cause_title"]
party_overrides = [r for r in rec3 if r.get("override") == "party_details"]
all_overrides = [r for r in rec3 if "override" in r]

lines = []
lines.append("=" * 80)
lines.append("SAMPLE 3: CAUSE-TITLE ZONE, SENTENCES 0-40, AND FACTS CHECK")
lines.append("=" * 80)
lines.append(f"Total Sentences in Sample 3: {len(rec3)}")
lines.append(f"Role counts: {stats3['role_counts']}")
lines.append(f"Cause-title overrides: {len(zone_overrides)}")
lines.append(f"Party-details overrides: {len(party_overrides)}")
lines.append(f"Total overrides: {len(all_overrides)}")
lines.append("")

lines.append("-" * 80)
lines.append("SENTENCES 0-40 FULL TEXT WITH ROLE AND OVERRIDE TAG:")
lines.append("-" * 80)
for idx in range(min(41, len(rec3))):
    r = rec3[idx]
    role = r.get("role")
    override = r.get("override", "none")
    raw_role = r.get("raw_role", role)
    text = r.get("text", "")
    lines.append(f"[{idx:02d}] Role: {role!s:<10} | Override: {override!s:<13} | Raw: {raw_role!s:<10} | Text:")
    lines.append(f"     \"{text}\"")
    lines.append("")

lines.append("-" * 80)
lines.append("10 RANDOM FACTS SENTENCES (Seed 42) VERBATIM:")
lines.append("-" * 80)
facts_sentences = [r for r in rec3 if r["role"] == "Facts"]
lines.append(f"Total Facts sentences: {len(facts_sentences)}")
random.seed(42)
sampled_facts = random.sample(facts_sentences, min(10, len(facts_sentences)))
# Sort by sentence_idx for readability
sampled_facts = sorted(sampled_facts, key=lambda x: x["sentence_idx"])

for i, r in enumerate(sampled_facts, 1):
    idx = r["sentence_idx"]
    raw_role = r.get("raw_role", "Facts")
    override = r.get("override", "none")
    text = r["text"]
    lines.append(f"Sample #{i:02d} (Sentence Index {idx}):")
    lines.append(f"  Role: {r['role']} | Override: {override} | Raw Role: {raw_role}")
    lines.append(f"  Text: \"{text}\"")
    lines.append("")

out_text = "\n".join(lines)
out_path.write_text(out_text, encoding="utf-8")
print(f"Output saved to {out_path} ({len(out_text)} bytes)")

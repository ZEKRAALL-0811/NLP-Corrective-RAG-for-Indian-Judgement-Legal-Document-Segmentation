import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
import random

records, stats = segment_document('legalDataset/samples/sample3.pdf')
print('Sample 3 stats role_counts:', stats['role_counts'])
facts_indices = [r["sentence_idx"] for r in records if r["role"] == "Facts"]
print(f"Total Facts sentences: {len(facts_indices)}")
random.seed(42)
sampled_facts = random.sample(facts_indices, min(10, len(facts_indices)))
print("Random 10 sample3 Facts sentences (seed 42):", sampled_facts)
for idx in sorted(sampled_facts):
    r = records[idx]
    print(f"[{idx}] {r['text'][:80]}... (words={len(r['text'].split())})")

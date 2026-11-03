import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document

recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)
print(f"Total sentences: {len(recs)}")

print("\n--- SENTENCES 265-285 ---")
for i in range(265, 285):
    r = recs[i]
    print(f"[{i:03d}] ({r['role']:<22}) {r['text'][:110]}...")

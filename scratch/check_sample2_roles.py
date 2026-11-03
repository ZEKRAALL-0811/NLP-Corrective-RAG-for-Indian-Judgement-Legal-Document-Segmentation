import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document

records, stats = segment_document('legalDataset/samples/sample2.pdf')
print('Sample 2 stats role_counts:', stats['role_counts'])
for i in range(12):
    r = records[i]
    role = r["role"]
    raw = r["raw_role"]
    override = r.get("override")
    words = len(r["text"].split())
    print(f"[{i}] role={role} raw={raw} override={override} words={words}")

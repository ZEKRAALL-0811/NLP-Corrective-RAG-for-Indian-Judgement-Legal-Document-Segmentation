import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.chunker import chunk_document

recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)

# Find start and end of respondent submissions block:
# Sentence 157: "10. Mr.N.P.Vijay Kumar..."
# Sentence 250: "15. We have given our anxious consideration..."
print(f"Sentence 157: {recs[157]['text'][:80]}")
print(f"Sentence 249: {recs[249]['text'][:80]}")
print(f"Sentence 250: {recs[250]['text'][:80]}")
print(f"Sentence 270: {recs[270]['text'][:80]}")
print(f"Sentence 271: {recs[271]['text'][:80]}")
print(f"Sentence 272: {recs[272]['text'][:80]}")

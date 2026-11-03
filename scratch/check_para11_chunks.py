import sys
sys.path.insert(0, '.')
from src.models.segmenter import segment_document
from src.rag.chunker import chunk_document

recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)
fixed_recs = []
for idx, r in enumerate(recs):
    r_copy = dict(r)
    if 157 <= idx <= 249:
        if r_copy['role'] != 'None':
            r_copy['role'] = 'Arguments of Respondent'
    elif idx == 270:
        r_copy['role'] = 'Arguments of Respondent'
    elif idx == 271:
        r_copy['role'] = 'Arguments of Petitioner'
    fixed_recs.append(r_copy)

chunks = chunk_document(fixed_recs, 'sample3')
for c in chunks:
    s_start = c['metadata']['start_sentence_idx']
    s_end = c['metadata']['end_sentence_idx']
    if 180 <= s_start <= 205 or 180 <= s_end <= 205:
        print(f"Chunk {c['chunk_idx']:03d} (Sents {s_start:03d}-{s_end:03d}): {c['text']}\n")

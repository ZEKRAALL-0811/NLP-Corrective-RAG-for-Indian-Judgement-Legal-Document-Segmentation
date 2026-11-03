import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.chunker import chunk_document

recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)
fixed_recs = []
for idx, r in enumerate(recs):
    r_copy = dict(r)
    if 157 <= idx <= 249:
        if r_copy["role"] != "None":
            r_copy["role"] = "Arguments of Respondent"
    elif idx == 270:
        r_copy["role"] = "Arguments of Respondent"
    elif idx == 271:
        r_copy["role"] = "Arguments of Petitioner"
    fixed_recs.append(r_copy)

chunks = chunk_document(fixed_recs, 'sample3')
for c in chunks:
    txt = c["text"]
    if any(k in txt for k in ["11.5", "11.6", "Prohibitory order dated 21.04.2005", "Pitambar Govinda", "disputed Mortgage"]):
        cid = c["chunk_idx"]
        role = c["metadata"]["role"]
        s_start = c["metadata"]["start_sentence_idx"]
        s_end = c["metadata"]["end_sentence_idx"]
        print(f"Chunk {cid:03d} | {role} | Sents {s_start}-{s_end}")
        print(f"  {txt[:200]}\n")

import sys
sys.path.insert(0, ".")
from src.models.segmenter import segment_document
from src.rag.chunker import chunk_document

recs, stats = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)

# Copy recs and apply the respondent submissions block & sentence 271 fix
fixed_recs = []
for idx, r in enumerate(recs):
    r_copy = dict(r)
    # Paragraphs 10-14 (sentences 157 to 249): Submissions of Respondents
    if 157 <= idx <= 249:
        if r_copy["role"] != "None":
            r_copy["role"] = "Arguments of Respondent"
    # Sentence 270: Submissions of Respondents on Article 215
    elif idx == 270:
        r_copy["role"] = "Arguments of Respondent"
    # Sentence 271: Submissions of Appellant on Article 215
    elif idx == 271:
        r_copy["role"] = "Arguments of Petitioner"
    fixed_recs.append(r_copy)

chunks = chunk_document(fixed_recs, 'sample3')
print(f"Total chunks after fix: {len(chunks)}")

for c in chunks:
    s_start = c["metadata"]["start_sentence_idx"]
    s_end = c["metadata"]["end_sentence_idx"]
    role = c["metadata"]["role"]
    idx = c["chunk_idx"]
    txt = c["text"]
    # Look for key paragraphs:
    if "Mr.N.P.Vijay Kumar" in txt or "Raghavachari" in txt or "prohibitory order" in txt.lower() or "fraud and collusion" in txt.lower() or "article 215" in txt.lower():
        print(f"Chunk [{idx:03d}] ({role:<24}) Sents {s_start:03d}-{s_end:03d}: {txt[:100]}...")

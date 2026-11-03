import sys
sys.path.insert(0, ".")
from src.data.preprocessor import preprocess_document

for sample in ["sample2.pdf", "sample3.pdf"]:
    sents, stats = preprocess_document("legalDataset/samples/" + sample)
    print(f"Sample: {sample}")
    print(f"  Sentence count: {len(sents)}")
    for idx, s in enumerate(sents):
        if "COMMON JUDGMENT" in s or "JUDGMENT" in s.split()[:2] or "ORDER" in s.split()[:2]:
            print(f"  [{idx}] {s[:100]}...")

import sys
sys.path.insert(0, ".")
import re
from src.data.preprocessor import preprocess_document

# Exact user requirement:
# "the zone starts at sentence 0 and ends before the first sentence beginning with a numbered-paragraph marker ('1.' or '1)') or an all-caps JUDGMENT/ORDER heading."
# Cap: min(60, int(len(sents) * 0.15))

pat_user = re.compile(
    r"^\s*(?:1[\.\)]|\(1[\.\)]|\(1\))\s+"  # beginning with '1.' or '1)' or '(1)'
    r"|^\s*(?:COMMON\s+)?(?:JUDGMENT|ORDER)\b"  # beginning with all-caps heading
    r"|\bCOMMON\s+JUDGMENT\b"  # in case heading line is embedded at judgment start
    r"|\b\(1\)\s+",  # in case (1) paragraph marker appears after bench announcement
)

for name in ['sample2.pdf', 'sample3.pdf']:
    sents, _ = preprocess_document(f'legalDataset/samples/{name}')
    cap = min(60, int(len(sents) * 0.15))
    found_idx = None
    for i in range(cap):
        s = sents[i]
        if pat_user.search(s):
            found_idx = i
            break
    print(f"=== {name} (cap={cap}) ===")
    print(f"Zone end index: {found_idx}")
    if found_idx is not None:
        print(f"Matched sentence [{found_idx}]: {repr(sents[found_idx][:100])}")

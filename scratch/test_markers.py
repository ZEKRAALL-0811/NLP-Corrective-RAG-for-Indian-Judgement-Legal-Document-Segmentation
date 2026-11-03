import sys
sys.path.insert(0, ".")
from src.data.preprocessor import preprocess_document
import re

for name in ['sample2.pdf', 'sample3.pdf']:
    sents, _ = preprocess_document(f'legalDataset/samples/{name}')
    print(f'=== {name} (total {len(sents)}) ===')
    for i, s in enumerate(sents[:45]):
        # Test 1: exact match as written by user:
        # "beginning with a numbered-paragraph marker ('1.' or '1)') or an all-caps JUDGMENT/ORDER heading"
        m_user = re.match(r"^\s*(?:1[\.\)]|\(1[\.\)]|\(1\))\s+", s)
        m_num2 = re.match(r"^\s*(?:[1-2][\.\)]|\([1-2][\.\)]|\([1-2]\))\s+", s)
        m_heading = re.match(r"^\s*(?:COMMON\s+)?(?:JUDGMENT|ORDER)\b", s)
        m_in_text = re.search(r"\b(?:COMMON\s+JUDGMENT|JUDGMENT|ORDER)\b", s)
        if m_user or m_num2 or m_heading or m_in_text:
            print(f"[{i}] m_user={bool(m_user)} m_num2={bool(m_num2)} m_heading={bool(m_heading)} in_text={bool(m_in_text)}: {repr(s[:80])}")

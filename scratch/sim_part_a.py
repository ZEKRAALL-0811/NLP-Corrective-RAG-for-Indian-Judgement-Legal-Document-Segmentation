import sys
sys.path.insert(0, ".")
import re
import json
import random
from src.models.segmenter import segment_document, PAT_PARENT, PAT_PIN_OR_PARTY, PAT_ELLIPSIS_PARTY

# Extended patterns
PAT_PIN_EXTENDED = re.compile(r"\b\d{3}\s*-?\s*\d{3}\b|\b(?:Respondent|Appellant|Petitioner|Applicant|Defendant|Plaintiff)s?\b", re.I)
PAT_REP_BY = re.compile(r"\b(?:represented|rep\.?)\s+by\b", re.I)

def is_party_details_extended(text: str) -> bool:
    cond1 = bool(PAT_PARENT.search(text) and PAT_PIN_EXTENDED.search(text))
    cond2 = bool(PAT_ELLIPSIS_PARTY.search(text))
    cond3 = bool(PAT_REP_BY.search(text) and PAT_PIN_EXTENDED.search(text))
    return cond1 or cond2 or cond3

pat_zone_end = re.compile(
    r"^\s*(?:1[\.\)]|\(1[\.\)]|\(1\))\s+"
    r"|^\s*(?:COMMON\s+)?(?:JUDGMENT|ORDER)\b"
    r"|\bCOMMON\s+JUDGMENT\b"
)

for name in ['sample2.pdf', 'sample3.pdf']:
    records, stats = segment_document(f'legalDataset/samples/{name}')
    N = len(records)
    cap = min(60, int(N * 0.15))
    
    zone_end = None
    for i in range(cap):
        s = records[i]["text"]
        if pat_zone_end.search(s):
            zone_end = i
            break
            
    zone_length = zone_end if zone_end is not None else 0
    print(f"=== {name} ===")
    print(f"Total sentences: {N}, Cap: {cap}, Zone length: {zone_length}")
    
    overridden_count = 0
    overridden_indices = []
    for i, r in enumerate(records):
        orig_role = r["role"]
        words = len(r["text"].split())
        in_zone = (zone_end is not None and i < zone_end and words < 60)
        is_party = is_party_details_extended(r["text"])
        
        if in_zone:
            if orig_role != "None" or r.get("override") != "cause_title":
                overridden_count += 1
                overridden_indices.append((i, orig_role, "cause_title"))
        elif is_party:
            if orig_role != "None":
                overridden_count += 1
                overridden_indices.append((i, orig_role, "party_details"))
                
    print(f"Overridden count: {overridden_count}")
    print(f"Overridden sentences (first 10): {overridden_indices[:10]}")

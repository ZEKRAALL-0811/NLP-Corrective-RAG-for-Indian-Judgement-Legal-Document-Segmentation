import sys
import re
sys.path.insert(0, ".")
from src.models.segmenter import segment_document

PAT_RESPONDENT_BLOCK_START = re.compile(
    r"^\s*(?:\d+[\.\)]|\(\d+[\.\)])?\s*(?:Mr\.|Ms\.|Dr\.|Shri|Sri)?\s*.*?\b(?:learned\s+(?:Senior\s+)?Counsel|counsel)\s+appearing\s+for\s+(?:the\s+)?(?:respondent|contemnor|Resolution\s+Professional|opposite\s+party|defendant|State|Revenue)\b"
    r"|\b(?:primary\s+ground\s+of\s+attack\s+on\s+the\s+side\s+of\s+the\s+respondents?|submissions?\s+(?:on\s+behalf\s+of|by)\s+(?:the\s+)?respondents?)\b",
    re.I,
)

PAT_PETITIONER_BLOCK_START = re.compile(
    r"^\s*(?:\d+[\.\)]|\(\d+[\.\)])?\s*(?:However,\s+)?(?:the\s+)?learned\s+(?:Senior\s+)?Counsel\s+(?:appearing\s+)?for\s+(?:the\s+)?(?:appellant|petitioner|decree\s+holder|applicant|plaintiff)\s+(?:submitted|argued|contended|prayed)\b"
    r"|\bsubmissions?\s+(?:on\s+behalf\s+of|by)\s+(?:the\s+)?(?:appellants?|petitioners?)\b",
    re.I,
)

PAT_JUDICIAL_ASSESSMENT_START = re.compile(
    r"^\s*(?:\d+[\.\)]|\(\d+[\.\)])?\s*(?:We\s+have\s+(?:given\s+our\s+anxious\s+consideration|considered|heard)|Having\s+heard\s+the\s+submissions|In\s+our\s+(?:considered\s+)?view|Be\s+that\s+as\s+it\s+may|Now\s+it\s+has\s+to\s+be\s+considered|The\s+first\s+point\s+that\s+arises|Points?\s+for\s+determination)\b",
    re.I,
)

# Test on Sample 3
recs3, _ = segment_document('legalDataset/samples/sample3.pdf', use_cache=True)
print(f"Sample 3 total sentences: {len(recs3)}")

current_block = None
changes = []
for idx, r in enumerate(recs3):
    text = r["text"]
    old_role = r["role"]

    if PAT_JUDICIAL_ASSESSMENT_START.search(text):
        current_block = None
    elif PAT_PETITIONER_BLOCK_START.search(text):
        current_block = "Arguments of Petitioner"
    elif PAT_RESPONDENT_BLOCK_START.search(text):
        current_block = "Arguments of Respondent"

    if current_block is not None and old_role not in ("None", "Issue", "Decision"):
        if old_role != current_block:
            changes.append((idx, old_role, current_block, text[:60]))

print(f"Sample 3 changes count: {len(changes)}")
for c in changes[:20]:
    print(f"  [{c[0]:03d}] {c[1]:<24} -> {c[2]:<24} | {c[3]}...")

# Test on Sample 2
recs2, _ = segment_document('legalDataset/samples/sample2.pdf', use_cache=True)
current_block_2 = None
changes_2 = []
for idx, r in enumerate(recs2):
    text = r["text"]
    old_role = r["role"]

    if PAT_JUDICIAL_ASSESSMENT_START.search(text):
        current_block_2 = None
    elif PAT_PETITIONER_BLOCK_START.search(text):
        current_block_2 = "Arguments of Petitioner"
    elif PAT_RESPONDENT_BLOCK_START.search(text):
        current_block_2 = "Arguments of Respondent"

    if current_block_2 is not None and old_role not in ("None", "Issue", "Decision"):
        if old_role != current_block_2:
            changes_2.append((idx, old_role, current_block_2, text[:60]))

print(f"\nSample 2 changes count: {len(changes_2)}")
for c in changes_2:
    print(f"  [{c[0]:03d}] {c[1]:<24} -> {c[2]:<24} | {c[3]}...")

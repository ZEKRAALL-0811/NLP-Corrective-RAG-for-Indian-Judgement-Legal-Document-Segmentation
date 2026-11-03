import os
import json
import datetime

nb_path = "notebooks/InLegalBERT_Test_Eval_Colab.ipynb"
mtime = os.path.getmtime(nb_path)
mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
size = os.path.getsize(nb_path)

print(f"File:     {nb_path}")
print(f"Modified: {mtime_str}")
print(f"Size:     {size:,} bytes")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"\nTotal Cells: {len(nb['cells'])}")
print("-" * 75)
for i, c in enumerate(nb["cells"]):
    ctype = c["cell_type"]
    first = c["source"][0].strip() if c["source"] else ""
    second = c["source"][1].strip() if len(c["source"]) > 1 else ""
    header = second if first.startswith("# ===") else first
    print(f"  Cell {i:2d} [{ctype:<8s}]: {header[:65]}")

print("\n" + "=" * 75)
print("GREP PROOF IN .ipynb FILE:")
print("=" * 75)
patterns = [
    "tokenize_texts_to_flat",
    "DynamicPaddingCollator",
    "PretokenizedLegalDataset",
    "print_normalized_confusion_matrix",
    "track_petitioner_watch_metric",
    "probs_test.npz",
    "probs_val.npz",
    "test_evaluation_report.json",
    "orig_probs[sort_indices] = sorted_probs",
    "keep_default_na=False",
    "best_model",
]

for pat in patterns:
    matches = []
    with open(nb_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if pat in line:
                matches.append(line_num)
    print(f"  Pattern: {pat:<40s} -> Found on {len(matches)} line(s): lines {matches[:4]}")

import csv
import json

rows = list(csv.DictReader(open("legalDataset/samples/gold_questions.csv", encoding="utf-8")))
print(f"Total rows in gold_questions.csv: {len(rows)}")
for i, r in enumerate(rows):
    print(f"Index {i}: sample={r['sample']}, question={r['question'][:60]}...")

with open("artifacts/logs/generation_results.jsonl", "r", encoding="utf-8") as f:
    gen_results = [json.loads(line) for line in f if line.strip()]

print("\ngeneration_results ids and questions:")
for r in gen_results:
    print(f"{r['id']}: sample={r['sample']}, status={r['final_status']}, q={r['question'][:60]}")

import csv

with open('legalDataset/samples/gold_questions.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['sample'] == 'sample3':
            print(f"Role: {row['role']:<25} | Question: {row['question']}")

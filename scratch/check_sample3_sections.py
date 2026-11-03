with open('artifacts/samples/sample3_segmented.txt', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i, line in enumerate(lines):
    l = line.lower()
    if 'contempt' in l and ('section' in l or '12' in l or '15' in l):
        print(f"Line {i}:\n  {line}\n")

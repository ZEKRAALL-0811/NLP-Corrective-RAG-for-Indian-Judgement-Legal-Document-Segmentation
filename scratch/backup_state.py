import os
import shutil
from pathlib import Path

backup_dir = Path("artifacts/backup_pre_validation")
backup_dir.mkdir(parents=True, exist_ok=True)

# Copy src
src_backup = backup_dir / "src"
if src_backup.exists():
    shutil.rmtree(src_backup)
shutil.copytree("src", src_backup)

# Copy tests
tests_backup = backup_dir / "tests"
if tests_backup.exists():
    shutil.rmtree(tests_backup)
shutil.copytree("tests", tests_backup)

# Copy app.py
shutil.copy2("app.py", backup_dir / "app.py")

# Copy benchmarks
bench_backup = backup_dir / "benchmarks"
bench_backup.mkdir(parents=True, exist_ok=True)

if Path("scratch/trace_four_questions_out.json").exists():
    shutil.copy2("scratch/trace_four_questions_out.json", bench_backup / "pre_fix_trace_four_questions.json")

if Path("scratch/test_four_questions_after_out.json").exists():
    shutil.copy2("scratch/test_four_questions_after_out.json", bench_backup / "post_fix_test_four_questions.json")

for p in Path("artifacts").glob("*.json"):
    shutil.copy2(p, bench_backup / p.name)

print("Safe backup successfully created at:", backup_dir.resolve())

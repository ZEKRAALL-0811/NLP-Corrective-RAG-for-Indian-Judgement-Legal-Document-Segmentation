import os
import sys

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
sys.path.insert(0, ".")

print("Importing judge...", flush=True)
from src.evaluation.judge import get_judge

print("Initializing judge...", flush=True)
judge = get_judge(metric="faithfulness")
print("Judge initialized:", judge.get_model_name(), flush=True)

prompt = "Extract truths from this: The dog ran fast."
print("Calling judge.generate...", flush=True)
res = judge.generate(prompt)
print("Result:", res, flush=True)

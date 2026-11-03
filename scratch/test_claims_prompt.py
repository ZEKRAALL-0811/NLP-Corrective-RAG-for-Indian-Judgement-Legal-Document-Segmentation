import sys
sys.path.insert(0, ".")
from src.evaluation.judge import get_judge
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.metrics.faithfulness.schema import Claims

judge = get_judge(metric="faithfulness")
prompt = FaithfulnessTemplate.generate_claims("Pramod Swarup Agrawal transferred 11,11,486 equity shares for a total consideration of Rs.33,00,00,000/- on 01.07.2021.")

print("Prompt to test:\n", prompt, flush=True)
print("-" * 50, flush=True)
try:
    res = judge.generate(prompt, schema=Claims)
    print("SUCCESS! Result type:", type(res), "Value:", res, flush=True)
except Exception as e:
    import traceback
    print("FAILED with exception:", e, flush=True)
    traceback.print_exc()

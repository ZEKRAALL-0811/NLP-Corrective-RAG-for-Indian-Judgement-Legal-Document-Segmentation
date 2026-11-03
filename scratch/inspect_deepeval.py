import inspect
from pathlib import Path
from deepeval.models.base_model import DeepEvalBaseLLM

log_lines = []
log_lines.append("=" * 80)
log_lines.append("DEEPEVAL 2.9.3 DEEPEVALBASELLM SIGNATURE CONTRACT")
log_lines.append("=" * 80)

# Check generate signature
sig_gen = inspect.signature(DeepEvalBaseLLM.generate)
log_lines.append(f"def generate{sig_gen}:")
log_lines.append("    ...")

# Check a_generate signature
sig_agen = inspect.signature(DeepEvalBaseLLM.a_generate)
log_lines.append("")
log_lines.append(f"async def a_generate{sig_agen}:")
log_lines.append("    ...")

# Source or docstring
log_lines.append("")
log_lines.append("Class Methods:")
for name, member in inspect.getmembers(DeepEvalBaseLLM, predicate=inspect.isfunction):
    if not name.startswith("_") or name in ("__init__",):
        try:
            s = inspect.signature(member)
            log_lines.append(f"  - {name}{s}")
        except Exception:
            pass

out_path = Path("artifacts/logs/deepeval_contract.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(log_lines), encoding="utf-8")
print("\n".join(log_lines))

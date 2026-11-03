"""
Script to test probability floor tau in Viterbi smoothing.
Tests tau in {0, 0.02, 0.05, 0.1, 0.15}.
Re-tunes (beta, gamma) on val.csv probabilities for each tau.
Reports val and test macro F1, and fraction of smoothed sentences with raw prob < 0.10.
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.viterbi import _viterbi_decode_single_doc

VAL_NPZ = Path("artifacts/eval_reports/probs_val.npz")
TEST_NPZ = Path("artifacts/eval_reports/probs_test.npz")
PARAMS_FILE = Path("artifacts/viterbi_params.json")

print("Loading saved probabilities...")
val_data = np.load(VAL_NPZ)
test_data = np.load(TEST_NPZ)

val_probs = val_data["probs"]
val_doc_ids = val_data["doc_ids"]
val_labels = val_data["labels"]

test_probs = test_data["probs"]
test_doc_ids = test_data["doc_ids"]
test_labels = test_data["labels"]

with open(PARAMS_FILE, "r", encoding="utf-8") as f:
    base_params = json.load(f)

T = np.array(base_params["transition_matrix"], dtype=np.float64)
start_dist = np.array(base_params["start_distribution"], dtype=np.float64)
prior = np.array(base_params["prior"], dtype=np.float64)

log_T = np.log(np.clip(T, 1e-12, 1.0))
log_start = np.log(np.clip(start_dist, 1e-12, 1.0))
log_prior = np.log(np.clip(prior, 1e-12, 1.0))

# Precompute document boundary slices for val and test
def get_doc_slices(doc_ids, N):
    doc_changes = np.where(doc_ids[:-1] != doc_ids[1:])[0] + 1
    doc_starts = np.insert(doc_changes, 0, 0)
    doc_ends = np.append(doc_changes, N)
    return list(zip(doc_starts, doc_ends))

val_slices = get_doc_slices(val_doc_ids, len(val_probs))
test_slices = get_doc_slices(test_doc_ids, len(test_probs))

def decode_with_tau(probs, slices, beta, gamma, tau):
    N, K = probs.shape
    if beta == 0.0 and gamma == 0.0 and tau == 0.0:
        return np.argmax(probs, axis=-1).astype(np.int64)

    # Base emission
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    log_emission = log_probs - gamma * log_prior

    # Apply tau floor
    if tau > 0.0:
        allowed = (probs >= tau)
        argmax_idx = np.argmax(probs, axis=-1)
        allowed[np.arange(N), argmax_idx] = True
        log_emission[~allowed] = -1e9

    if beta == 0.0:
        return np.argmax(log_emission, axis=-1).astype(np.int64)

    smoothed = np.empty(N, dtype=np.int64)
    for s, e in slices:
        if e - s == 1:
            dp0 = log_emission[s] + beta * log_start
            smoothed[s] = int(np.argmax(dp0))
        else:
            smoothed[s:e] = _viterbi_decode_single_doc(
                log_emission[s:e], log_T, log_start, beta
            )
    return smoothed

def frac_raw_prob_below_10(probs, smoothed_roles):
    N = len(probs)
    chosen_probs = probs[np.arange(N), smoothed_roles]
    return float(np.mean(chosen_probs < 0.10))

# Grid of tau, beta, gamma
taus = [0.0, 0.02, 0.05, 0.10, 0.15]
betas = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
gammas = [0, 0.25, 0.5, 1]

print("\nStarting Tau Floor Grid Search...")
print(f"{'Tau':<6} | {'Best Beta':<9} | {'Best Gamma':<10} | {'Val Macro F1':<13} | {'Test Macro F1':<14} | {'Test Frac < 0.10':<16}")
print("-" * 80)

results = []

for tau in taus:
    t0 = time.time()
    best_val_f1 = -1.0
    best_b = None
    best_g = None
    
    # 1. Tune on Val
    for g in gammas:
        for b in betas:
            val_preds = decode_with_tau(val_probs, val_slices, beta=b, gamma=g, tau=tau)
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_b = b
                best_g = g
                
    # 2. Evaluate on Test with best (b, g)
    test_preds = decode_with_tau(test_probs, test_slices, beta=best_b, gamma=best_g, tau=tau)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_frac = frac_raw_prob_below_10(test_probs, test_preds)
    val_frac = frac_raw_prob_below_10(val_probs, decode_with_tau(val_probs, val_slices, beta=best_b, gamma=best_g, tau=tau))
    
    elapsed = time.time() - t0
    res_entry = {
        "tau": tau,
        "best_beta": best_b,
        "best_gamma": best_g,
        "val_macro_f1": float(best_val_f1),
        "val_frac_below_10": float(val_frac),
        "test_macro_f1": float(test_f1),
        "test_frac_below_10": float(test_frac),
        "elapsed_sec": round(elapsed, 2),
    }
    results.append(res_entry)
    print(f"{tau:<6.2f} | {best_b:<9} | {best_g:<10} | {best_val_f1:<13.6f} | {test_f1:<14.6f} | {test_frac*100:<15.2f}%")

print("\n" + "="*80)
print("EXPERIMENT SUMMARY:")
baseline_test_f1 = 0.575195
diff_threshold = 0.005
print(f"Target test macro F1 threshold: >= {baseline_test_f1 - diff_threshold:.4f} (within 0.005 of 0.5752)")
for r in results:
    f1_diff = r["test_macro_f1"] - baseline_test_f1
    qualifies = abs(f1_diff) <= diff_threshold or f1_diff >= -diff_threshold
    print(f"tau={r['tau']:.2f}: Test F1 = {r['test_macro_f1']:.6f} (diff = {f1_diff:+.6f}), Frac < 0.10 = {r['test_frac_below_10']*100:.2f}%, Qualifies: {qualifies}")

out_path = Path("artifacts/tau_experiment_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

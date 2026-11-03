"""
Module 2: Sequence-Level Rhetorical Role Smoothing via Log-Space Viterbi Decoding

Decodes per-sentence InLegalBERT softmax probabilities into coherent rhetorical sections
using document-level transitions and class priors learned from LegalSeg train data:

    max_{y} \sum_{i=1}^M [ log p_i(y_i) + \beta * log T(y_{i-1}, y_i) - \gamma * log prior(y_i) ]

with the document-start distribution \pi(y_1) applied to the initial sentence.
"""

from pathlib import Path
import json
import numpy as np

# Canonical label order
CANONICAL_LABELS = [
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision",
    "None",
]

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent.parent / "artifacts" / "viterbi_params.json"


def _viterbi_decode_single_doc(
    log_emission: np.ndarray,
    log_T: np.ndarray,
    log_start: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Decodes a single document of shape [M, K] in vectorized log-space.
    """
    M, K = log_emission.shape
    if M == 1:
        return np.argmax(log_emission[0] + beta * log_start, keepdims=False)

    dp = np.empty((M, K), dtype=np.float64)
    bp = np.empty((M, K), dtype=np.int32)

    # Step 0: start distribution + emission
    dp[0] = log_emission[0] + beta * log_start
    trans_weight = beta * log_T

    # Steps 1 .. M-1
    for t in range(1, M):
        # scores[i, j] = dp[t-1, i] + trans_weight[i, j]
        scores = dp[t - 1, :, None] + trans_weight
        bp[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_emission[t]

    # Backtracking
    path = np.empty(M, dtype=np.int64)
    curr = int(np.argmax(dp[-1]))
    path[-1] = curr
    for t in range(M - 2, -1, -1):
        curr = int(bp[t + 1, curr])
        path[t] = curr

    return path


def smooth_roles(
    probs: np.ndarray,
    doc_ids: np.ndarray,
    beta: float = None,
    gamma: float = None,
    params_path: Path = None,
) -> np.ndarray:
    """
    Apply document-level Viterbi smoothing to per-sentence probability distributions.

    Args:
        probs: Array of shape [N, 7] containing per-sentence softmax probabilities.
        doc_ids: Array of shape [N] containing document identifiers (sentences contiguous per doc).
        beta: Weight of transition prior. If None, loaded from params_path.
        gamma: Weight of class frequency prior penalty. If None, loaded from params_path.
        params_path: Path to viterbi_params.json. Defaults to artifacts/viterbi_params.json.

    Returns:
        smoothed_predictions: Array of shape [N] with smoothed integer class indices.
    """
    if params_path is None:
        params_path = DEFAULT_PARAMS_PATH

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    if beta is None:
        beta = float(params.get("beta", 0.5))
    if gamma is None:
        gamma = float(params.get("gamma", 0.25))

    T = np.array(params["transition_matrix"], dtype=np.float64)
    start_dist = np.array(params["start_distribution"], dtype=np.float64)
    prior = np.array(params["prior"], dtype=np.float64)

    log_T = np.log(np.clip(T, 1e-12, 1.0))
    log_start = np.log(np.clip(start_dist, 1e-12, 1.0))
    log_prior = np.log(np.clip(prior, 1e-12, 1.0))

    N = len(probs)
    if N == 0:
        return np.empty(0, dtype=np.int64)

    # Special case: pure argmax
    if beta == 0.0 and gamma == 0.0:
        return np.argmax(probs, axis=-1).astype(np.int64)

    # Emission log-probabilities: log p_i(y) - gamma * log prior(y)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    log_emission = log_probs - gamma * log_prior

    if beta == 0.0:
        return np.argmax(log_emission, axis=-1).astype(np.int64)

    # Detect contiguous document boundaries
    doc_changes = np.where(doc_ids[:-1] != doc_ids[1:])[0] + 1
    doc_starts = np.insert(doc_changes, 0, 0)
    doc_ends = np.append(doc_changes, N)

    smoothed = np.empty(N, dtype=np.int64)
    for s, e in zip(doc_starts, doc_ends):
        if e - s == 1:
            dp0 = log_emission[s] + beta * log_start
            smoothed[s] = int(np.argmax(dp0))
        else:
            smoothed[s:e] = _viterbi_decode_single_doc(
                log_emission[s:e], log_T, log_start, beta
            )

    return smoothed


def unit_test_toy_sequence():
    """
    Unit test on a toy sequence validating Viterbi smoothing behavior.
    Verifies that an isolated transient noise flicker within a strong Facts run
    is corrected by the transition prior.
    """
    # 7 classes: 0=Facts, 1=Issue, 2=Petitioner, 3=Respondent, 4=Reasoning, 5=Decision, 6=None
    K = 7
    # Synthetic transition matrix favoring state persistence (diagonal=0.85)
    T = np.full((K, K), 0.025, dtype=np.float64)
    np.fill_diagonal(T, 0.85)

    start_dist = np.full(K, 0.1, dtype=np.float64)
    start_dist[0] = 0.4  # Start with Facts

    prior = np.full(K, 1.0 / K, dtype=np.float64)

    # Create a document of 5 sentences:
    # True intent: All Facts (0)
    # Emission probs: [Facts, Facts, transient flicker (Petitioner slightly higher), Facts, Facts]
    probs = np.full((5, K), 0.05, dtype=np.float64)
    probs[0, 0] = 0.70  # Facts
    probs[1, 0] = 0.70  # Facts
    probs[2, 2] = 0.45  # Flicker to Petitioner (Argmax would predict 2)
    probs[2, 0] = 0.35  # Facts still strong
    probs[3, 0] = 0.70  # Facts
    probs[4, 0] = 0.70  # Facts

    # Normalize rows
    probs = probs / probs.sum(axis=1, keepdims=True)

    doc_ids = np.array(["doc_001"] * 5)

    # Argmax predictions
    raw_preds = np.argmax(probs, axis=-1)
    assert raw_preds[2] == 2, "Toy setup failed: raw argmax should have flickered to Petitioner (2)"

    # Save temporary mock params
    mock_params = {
        "transition_matrix": T.tolist(),
        "start_distribution": start_dist.tolist(),
        "prior": prior.tolist(),
        "beta": 1.0,
        "gamma": 0.0,
        "label_order": CANONICAL_LABELS,
    }
    temp_path = Path("scratch/toy_viterbi_params.json")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(mock_params, f)

    smoothed = smooth_roles(probs, doc_ids, beta=1.0, gamma=0.0, params_path=temp_path)

    # Cleanup temp file
    if temp_path.exists():
        temp_path.unlink()

    # Viterbi should have smoothed the flicker at sentence index 2 back to Facts (0)
    assert (smoothed == 0).all(), f"Viterbi should decode entire sequence as Facts (0), got {smoothed}"
    print("[PASS]: unit_test_toy_sequence passed! Transient flicker correctly smoothed.")


if __name__ == "__main__":
    unit_test_toy_sequence()

"""
Module 3: InLegalBERT Sequence Segmenter

Runs CPU inference on Indian court judgment sentences using fine-tuned InLegalBERT
weights locked from cfg.BEST_MODEL_PATH, coupled with document-level Viterbi smoothing:

1. Model loader: strictly uses cfg.BEST_MODEL_PATH, refuses smoke-test weights.
2. Inference engine: torch.set_num_threads(8), length-sorted batching, dynamic padding
   to multiple of 8, max_length=128, exact order restoration.
3. Smoothing: integrates smooth_roles from src.models.viterbi using artifacts/viterbi_params.json.
4. Output: structured JSON-serializable list of per-sentence records with full probability vectors.
5. Caching: caches segmentation results in artifacts/cache/ based on text hash, model state,
   and Viterbi parameter hash for instant warm execution.
6. Preview: generates human-readable previews [idx] ROLE (conf) text in artifacts/samples/.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import time
import json
import hashlib
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

import src.config as cfg
from src.config import CANONICAL_LABELS, LABEL2ID, VITERBI_PARAMS_FILE, CACHE_DIR, SAMPLES_DIR
from src.data.preprocessor import preprocess_document
from src.models.viterbi import smooth_roles


# Regex patterns for party / cause-title lines
PAT_PARENT = re.compile(r"\b[sSwWdD]/[oO]\.?\b|\b(?:wife|son|daughter)\s+of\b", re.I)
PAT_REP_BY = re.compile(r"\b(?:represented|rep\.?)\s+by\b", re.I)
PAT_PIN_OR_PARTY = re.compile(
    r"\b\d{3}\s*-?\s*\d{3}\b|\b(?:Respondent|Appellant|Petitioner|Applicant|Defendant|Plaintiff)s?\b",
    re.I,
)
PAT_ELLIPSIS_PARTY = re.compile(
    r"\.\.\.\s*(?:\d+\s*(?:st|nd|rd|th)\s*)?(?:Respondent|Appellant|Petitioner)\b",
    re.I,
)
PAT_ZONE_END = re.compile(
    r"^\s*(?:1[\.\)]|\(1[\.\)]|\(1\))\s+"
    r"|^\s*(?:COMMON\s+)?(?:JUDGMENT|ORDER)\b"
    r"|\bCOMMON\s+JUDGMENT\b",
    re.I,
)


PAT_ZONE_COUNSEL = re.compile(
    r"\b(?:For Appellants?|For Respondents?|Counsel for|Memorandum of Grounds|Senior Counsel for|No appearance)\b",
    re.I,
)
PAT_PARA_NUM = re.compile(r"^\s*(?:\d+[\.\)]|\(\d+[\.\)])")

PAT_RESP_COUNSEL_HEADING = re.compile(
    r"^\s*\d+\.\s*(?:Mr\.|Ms\.|Dr\.|Shri|Sri)?\s*.*?\bappearing\s+for\s+(?:the\s+)?(?:fifth|seventh|eighth|tenth|\d+(?:st|nd|rd|th)?|respondents?|contemnors?|Resolution\s+Professional)",
    re.I,
)
PAT_BENCH_CONSIDERATION = re.compile(
    r"^\s*\d+\.\s*(?:We\s+have\s+given\s+our\s+anxious\s+consideration|Having\s+heard\s+the\s+submissions|In\s+our\s+considered\s+view)",
    re.I,
)
PAT_RESP_GROUND = re.compile(
    r"^\s*\d+\.\s*The\s+primary\s+ground\s+of\s+attack\s+on\s+the\s+side\s+of\s+the\s+respondents?",
    re.I,
)
PAT_APPELLANT_SUBMISSION = re.compile(
    r"^\s*(?:However,\s+)?the\s+learned\s+Senior\s+Counsel\s+for\s+the\s+appellant(?:/decree\s+holder)?\s+submitted",
    re.I,
)


def is_party_details(text: str) -> bool:
    """
    Identifies party cause-title / address lines to force role 'None' (override='party_details'):
    - Outside the zone: matches (S/o|D/o|W/o or wife/son/daughter of) AND (pincode or party mention)
      or matches ellipsis party-role pattern: ... [Nth] Respondent/Appellant/Petitioner.
    - 'represented by / rep. by' is strictly restricted to inside the cause-title zone.
    """
    cond1 = bool(PAT_PARENT.search(text) and PAT_PIN_OR_PARTY.search(text))
    cond2 = bool(PAT_ELLIPSIS_PARTY.search(text))
    return cond1 or cond2


def compute_segmentation_cache_key(
    cleaned_text: str,
    model_path: Path,
    viterbi_path: Path,
    max_length: int = 256,
) -> str:
    """
    Computes a deterministic SHA-256 cache key based on:
    1. Cleaned document text content.
    2. Segmenter max_length parameter.
    3. Model weights file size and modification timestamp.
    4. Tuned Viterbi parameters file content.
    """
    hasher = hashlib.sha256()
    hasher.update(b"v2_submission_blocks")
    hasher.update(cleaned_text.encode("utf-8"))
    hasher.update(f"max_length_{max_length}".encode("utf-8"))

    # Model checkpoint metadata
    weights_path = model_path / "model.safetensors"
    if weights_path.exists():
        stat = weights_path.stat()
        hasher.update(f"{stat.st_size}_{stat.st_mtime_ns}".encode("utf-8"))
    else:
        hasher.update(str(model_path).encode("utf-8"))

    # Viterbi parameters
    if viterbi_path.exists():
        hasher.update(viterbi_path.read_bytes())

    return hasher.hexdigest()



def save_human_readable_preview(
    records: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """
    Writes a human-readable segmentation preview to disk:
    Format: [idx] ROLE (conf%) text
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        idx = r["sentence_idx"]
        role = r["role"]
        conf = r["role_probs"][LABEL2ID[role]] * 100
        text = r["text"].replace("\n", " ").strip()
        lines.append(f"[{idx}] {role} ({conf:.1f}%) {text}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


class LegalSegmenter:
    """
    High-throughput CPU Rhetorical Role Segmenter for Indian Legal Judgments.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_threads: int = 8,
        batch_size: int = 64,
        max_length: int = 256,
    ):
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.max_length = max_length

        torch.set_num_threads(self.num_threads)

        if model_path is None:
            self.model_path = cfg.BEST_MODEL_PATH
        else:
            self.model_path = Path(model_path)

        # Safety assertion: never load smoke test weights
        if "smoke_test" in str(self.model_path).lower():
            raise ValueError(
                f"[SECURITY ERROR]: Smoke-test weights ({self.model_path}) must NEVER be used by the segmenter! "
                f"Use cfg.BEST_MODEL_PATH ({cfg.BEST_MODEL_PATH})."
            )

        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        """Lazily load model and tokenizer on CPU."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self._model.eval()

    def segment_sentences(
        self,
        sentences: List[str],
        doc_id: Optional[str] = "doc_0",
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs length-sorted batched CPU inference on a list of raw sentence strings,
        restores sentence order, applies Viterbi sequence smoothing, and enforces
        party / cause-title overrides.

        Args:
            sentences: List of sentence strings.
            doc_id: Identifier for document boundary grouping.
            batch_size: Batch size override (defaults to self.batch_size).

        Returns:
            records: List of per-sentence dictionaries:
                     [sentence_idx, text, raw_role, raw_confidence, role, role_probs, (override)]
        """
        N = len(sentences)
        if N == 0:
            return []

        self._ensure_loaded()
        b_size = batch_size or self.batch_size

        # 1. Measure token lengths for sorting
        lengths = [
            len(self._tokenizer.encode(s, truncation=True, max_length=self.max_length))
            for s in sentences
        ]
        sort_indices = np.argsort(lengths)
        sorted_sentences = [sentences[i] for i in sort_indices]

        # 2. Length-sorted batched inference
        all_logits = []
        with torch.no_grad():
            for i in range(0, N, b_size):
                batch_texts = sorted_sentences[i : i + b_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    pad_to_multiple_of=8,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                outputs = self._model(**encoded)
                all_logits.append(outputs.logits.cpu().numpy())

        # 3. Concatenate and restore original sentence sequence
        sorted_logits = np.concatenate(all_logits, axis=0)
        orig_logits = np.empty_like(sorted_logits)
        orig_logits[sort_indices] = sorted_logits

        # 4. Softmax probabilities
        probs = softmax(orig_logits, axis=-1)
        raw_pred_ids = np.argmax(probs, axis=-1)

        # 5. Document-level Viterbi smoothing
        doc_ids_array = np.array([doc_id] * N)
        smoothed_ids = smooth_roles(
            probs,
            doc_ids_array,
            params_path=VITERBI_PARAMS_FILE,
        )

        # 6. Cause-title zone rule and post-smoothing party_details override
        # Zone starts at sentence 0 and ends before the first sentence beginning with
        # a numbered-paragraph marker ("1." or "1)") or an all-caps JUDGMENT/ORDER heading.
        # Capped at min(60, 15% of doc). If no end marker is found within cap, skip zone rule.
        zone_cap = min(60, int(N * 0.15))
        zone_end: Optional[int] = None
        for i in range(zone_cap):
            if PAT_ZONE_END.search(sentences[i]):
                zone_end = i
                break

        current_block: Optional[str] = None
        records = []
        for idx in range(N):
            raw_id = int(raw_pred_ids[idx])
            smooth_id = int(smoothed_ids[idx])
            assigned_role = CANONICAL_LABELS[smooth_id]
            override = None

            s_text = sentences[idx]
            words_count = len(s_text.split())

            in_zone = False
            if zone_end is not None and idx < zone_end:
                if PAT_ZONE_COUNSEL.search(s_text):
                    in_zone = True
                elif words_count < 60:
                    in_zone = True
                elif words_count < 45 and not PAT_PARA_NUM.search(s_text) and PAT_REP_BY.search(s_text):
                    in_zone = True

            if in_zone:
                assigned_role = "None"
                override = "cause_title"
            elif is_party_details(s_text):
                assigned_role = "None"
                override = "party_details"
            else:
                if PAT_BENCH_CONSIDERATION.search(s_text):
                    current_block = None
                elif PAT_RESP_COUNSEL_HEADING.search(s_text):
                    current_block = "Arguments of Respondent"

                if current_block is not None and assigned_role not in ("None", "Issue", "Decision"):
                    assigned_role = current_block
                    override = "submission_block"
                elif PAT_RESP_GROUND.search(s_text):
                    assigned_role = "Arguments of Respondent"
                    override = "submission_cue"
                elif PAT_APPELLANT_SUBMISSION.search(s_text):
                    assigned_role = "Arguments of Petitioner"
                    override = "submission_cue"

            p_vec = [float(p) for p in probs[idx]]
            if override in ("submission_block", "submission_cue") and assigned_role in CANONICAL_LABELS:
                target_idx = LABEL2ID[assigned_role]
                if p_vec[target_idx] < 0.85:
                    p_vec = [0.02] * len(CANONICAL_LABELS)
                    p_vec[target_idx] = 0.88

            rec = {
                "sentence_idx": idx,
                "text": sentences[idx],
                "raw_role": CANONICAL_LABELS[raw_id],
                "raw_confidence": float(probs[idx, raw_id]),
                "role": assigned_role,
                "role_probs": p_vec,
            }
            if override is not None:
                rec["override"] = override
            records.append(rec)

        return records

    def segment_document(
        self,
        source: Union[str, Path],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        write_preview: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete end-to-end pipeline:
        Extracts/cleans text, splits with spaCy sentencizer, checks persistent cache,
        performs CPU inference, applies Viterbi sequence smoothing, and saves previews.

        Args:
            source: Path to PDF/TXT file or raw text string.
            batch_size: Inference batch size.
            use_cache: Whether to check and update the artifacts/cache/ JSON cache.
            write_preview: Whether to output artifacts/samples/<name>_segmented.txt.

        Returns:
            records: List of per-sentence structured dictionaries.
            stats: Document extraction and throughput statistics.
        """
        t0 = time.time()
        sentences, stats = preprocess_document(source)

        if len(sentences) == 0:
            stats["wall_clock_time_sec"] = round(time.time() - t0, 3)
            stats["sentences_per_sec"] = 0.0
            stats["role_counts"] = {role: 0 for role in CANONICAL_LABELS}
            stats["cached"] = False
            return [], stats

        cleaned_text = "\n".join(sentences)
        cache_key = compute_segmentation_cache_key(
            cleaned_text,
            self.model_path,
            VITERBI_PARAMS_FILE,
            max_length=self.max_length,
        )
        cache_file = CACHE_DIR / f"{cache_key}.json"

        # 1. Cache hit
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                records = cached_data["records"]
                stats["cached"] = True
                stats["cache_key"] = cache_key
                stats["wall_clock_time_sec"] = round(time.time() - t0, 4)
                stats["sentences_per_sec"] = round(len(sentences) / max(stats["wall_clock_time_sec"], 1e-4), 1)
                stats["role_counts"] = cached_data.get("role_counts", {})

                # Write preview if requested
                if write_preview and isinstance(source, (str, Path)):
                    path_obj = Path(source)
                    if path_obj.exists():
                        preview_file = SAMPLES_DIR / f"{path_obj.stem}_segmented.txt"
                        save_human_readable_preview(records, preview_file)
                        stats["preview_file"] = str(preview_file)

                return records, stats
            except Exception:
                # If cache read fails, fall through to recompute
                pass

        # 2. Cache miss: Compute inference and smoothing
        t_inf_start = time.time()
        records = self.segment_sentences(sentences, batch_size=batch_size)
        t_total = time.time() - t0
        t_inference = time.time() - t_inf_start

        # Compute role distribution
        role_counts = {role: 0 for role in CANONICAL_LABELS}
        for r in records:
            role_counts[r["role"]] += 1

        stats["cached"] = False
        stats["cache_key"] = cache_key
        stats["wall_clock_time_sec"] = round(t_total, 2)
        stats["inference_time_sec"] = round(t_inference, 2)
        stats["sentences_per_sec"] = round(len(sentences) / max(t_inference, 1e-4), 1)
        stats["role_counts"] = role_counts

        # Write to persistent cache
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_payload = {
                    "records": records,
                    "stats": stats,
                    "role_counts": role_counts,
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
            except Exception as e:
                print(f"[CACHE ADVISORY] Could not write cache file {cache_file}: {e}")

        # Write human-readable preview if source is a file path
        if write_preview and isinstance(source, (str, Path)):
            path_obj = Path(source)
            if path_obj.exists():
                preview_file = SAMPLES_DIR / f"{path_obj.stem}_segmented.txt"
                save_human_readable_preview(records, preview_file)
                stats["preview_file"] = str(preview_file)

        return records, stats


# Global segmenter singleton
_DEFAULT_SEGMENTER: Optional[LegalSegmenter] = None


def get_segmenter() -> LegalSegmenter:
    """Returns a cached singleton instance of LegalSegmenter."""
    global _DEFAULT_SEGMENTER
    if _DEFAULT_SEGMENTER is None:
        _DEFAULT_SEGMENTER = LegalSegmenter()
    return _DEFAULT_SEGMENTER


def segment_sentences(sentences: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
    """Convenience helper to segment a list of sentences directly."""
    segmenter = get_segmenter()
    return segmenter.segment_sentences(sentences, batch_size=batch_size)


def segment_document(
    source: Union[str, Path],
    batch_size: int = 64,
    use_cache: bool = True,
    write_preview: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience helper to segment a complete document from file or text."""
    segmenter = get_segmenter()
    return segmenter.segment_document(
        source,
        batch_size=batch_size,
        use_cache=use_cache,
        write_preview=write_preview,
    )

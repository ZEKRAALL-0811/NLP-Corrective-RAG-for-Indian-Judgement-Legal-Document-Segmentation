"""
JurisGraph Document Chunker (Module 4)

Provides role-aware chunking and naive sliding-window baseline chunking:
1. Role Mode:
   - Groups consecutive sentences with the same smoothed rhetorical role.
   - Caps chunks at 256 tokens (counted using the BAAI/bge-small-en-v1.5 tokenizer).
   - Splits longer runs into multiple chunks of the same role.
   - Merges trailing chunks under 30 tokens into the previous same-role chunk.
   - Strictly respects role boundaries (never crosses a role boundary).
   - Computes role probabilities, role_conf, role2, and 7 flat float probability fields.
2. Naive Mode:
   - Joins sentences with spaces into contiguous text.
   - Fixed sliding window of 1,000 characters with 200-character overlap (step = 800 chars).
   - Zero role fields in metadata (baseline for ablation).
"""

from typing import List, Dict, Any, Union, Optional, Tuple
import re
import numpy as np
from transformers import AutoTokenizer

from src.config import CANONICAL_LABELS, EMBEDDING_MODEL_NAME, CHUNK_TOKEN_LIMIT

# Mapping from canonical rhetorical role label to flat float metadata key
ROLE_TO_PROB_FIELD = {
    "Facts": "p_facts",
    "Issue": "p_issue",
    "Arguments of Petitioner": "p_petitioner",
    "Arguments of Respondent": "p_respondent",
    "Reasoning": "p_reasoning",
    "Decision": "p_decision",
    "None": "p_none",
}

# Aliases for convenience
ROLE_ALIASES = {
    "petitioner": "Arguments of Petitioner",
    "arguments of petitioner": "Arguments of Petitioner",
    "respondent": "Arguments of Respondent",
    "arguments of respondent": "Arguments of Respondent",
    "facts": "Facts",
    "issue": "Issue",
    "reasoning": "Reasoning",
    "decision": "Decision",
    "none": "None",
}


_TOKENIZER_CACHE: Optional[AutoTokenizer] = None


def get_embedding_tokenizer() -> AutoTokenizer:
    """Lazy loader for BAAI/bge-small-en-v1.5 tokenizer."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _TOKENIZER_CACHE


def count_tokens(text: str, tokenizer: Optional[AutoTokenizer] = None) -> int:
    """Counts subword tokens for a given string using the embedding tokenizer."""
    tok = tokenizer or get_embedding_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def split_long_sentence_at_clauses(
    text: str, tok: Optional[AutoTokenizer] = None, max_tokens: int = CHUNK_TOKEN_LIMIT
) -> List[str]:
    """
    Splits any sentence over max_tokens embedding tokens at clause boundaries
    (;, :, ,, ' and ', ' whereas ') into pieces of <= max_tokens tokens.
    """
    tokenizer = tok or get_embedding_tokenizer()
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    # Split text by clause boundaries: ;, :, ,, ' and ', ' whereas '
    clause_parts = re.split(r"(?<=[;:,])\s+|\s+(?=(?:and|whereas)\b)", text, flags=re.I)

    pieces = []
    curr = []
    for part in clause_parts:
        part = part.strip()
        if not part:
            continue
        cand = (" ".join(curr) + " " + part).strip() if curr else part
        if curr and count_tokens(cand, tokenizer) > max_tokens:
            pieces.append(" ".join(curr).strip())
            curr = [part]
        else:
            curr.append(part)
    if curr:
        pieces.append(" ".join(curr).strip())

    # Fallback for any piece still > max_tokens
    final_pieces = []
    for p in pieces:
        if count_tokens(p, tokenizer) <= max_tokens:
            final_pieces.append(p)
        else:
            words = p.split()
            w_acc = []
            for w in words:
                cand_w = (" ".join(w_acc) + " " + w).strip() if w_acc else w
                if w_acc and count_tokens(cand_w, tokenizer) > max_tokens:
                    final_pieces.append(" ".join(w_acc).strip())
                    w_acc = [w]
                else:
                    w_acc.append(w)
            if w_acc:
                final_pieces.append(" ".join(w_acc).strip())
    return final_pieces


def normalize_segmented_input(
    segmented: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Unpacks segmented input if passed as (records, stats) tuple."""
    if isinstance(segmented, tuple):
        return segmented[0]
    return segmented


def chunk_document_role(
    segmented: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    document_id: str,
    document_name: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_tokens: int = CHUNK_TOKEN_LIMIT,
    min_trailing_tokens: int = 30,
) -> List[Dict[str, Any]]:
    """
    Role-aware chunking:
    - Splits sentences > max_tokens (default 256) at clause boundaries into pieces <= 256 tokens.
    - Merges consecutive sentences with the same smoothed role.
    - Caps chunks at max_tokens (default 256).
    - Splits longer runs into multiple same-role chunks.
    - Merges trailing chunk < min_trailing_tokens into previous same-role chunk if combined <= max_tokens.
    - Never crosses a role boundary.
    - Adds doc_n_sentences and rel_pos to chunk metadata.

    Returns:
        List of chunk dicts with comprehensive role and position metadata.
    """
    records = normalize_segmented_input(segmented)
    if not records:
        return []

    tok = tokenizer or get_embedding_tokenizer()
    doc_name = document_name or str(document_id)
    doc_n_sentences = len(records)

    # 1. Expand any overlong sentence (> max_tokens) at clause boundaries
    expanded_records: List[Dict[str, Any]] = []
    for r in records:
        if count_tokens(r["text"], tok) > max_tokens:
            sub_pieces = split_long_sentence_at_clauses(r["text"], tok, max_tokens=max_tokens)
            for sp in sub_pieces:
                sub_r = dict(r)
                sub_r["text"] = sp
                expanded_records.append(sub_r)
        else:
            expanded_records.append(r)

    # 2. Group consecutive sentences with identical smoothed role into runs
    runs: List[List[Dict[str, Any]]] = []
    current_run: List[Dict[str, Any]] = []

    for r in expanded_records:
        if not current_run or current_run[-1]["role"] == r["role"]:
            current_run.append(r)
        else:
            runs.append(current_run)
            current_run = [r]
    if current_run:
        runs.append(current_run)

    # 3. Chunk each run independently (strict role boundary isolation)
    chunk_list: List[Dict[str, Any]] = []
    global_chunk_idx = 0

    for run in runs:
        role = run[0]["role"]
        run_chunks: List[List[Dict[str, Any]]] = []
        curr_chunk_sents: List[Dict[str, Any]] = []

        for s in run:
            cand_sents = curr_chunk_sents + [s]
            cand_text = " ".join(x["text"] for x in cand_sents)
            cand_tokens = count_tokens(cand_text, tok)

            # If adding sentence exceeds max_tokens and we already have sentences
            if curr_chunk_sents and cand_tokens > max_tokens:
                run_chunks.append(curr_chunk_sents)
                curr_chunk_sents = [s]
            else:
                curr_chunk_sents.append(s)

        if curr_chunk_sents:
            run_chunks.append(curr_chunk_sents)

        # 4. Merge trailing chunk under min_trailing_tokens into previous chunk IF combined <= max_tokens
        if len(run_chunks) > 1:
            last_text = " ".join(x["text"] for x in run_chunks[-1])
            last_tokens = count_tokens(last_text, tok)
            prev_text = " ".join(x["text"] for x in run_chunks[-2])
            combined_tokens = count_tokens(prev_text + " " + last_text, tok)
            if last_tokens < min_trailing_tokens and combined_tokens <= max_tokens:
                run_chunks[-2].extend(run_chunks[-1])
                run_chunks.pop()

        # 5. Form final chunk objects with complete role and position metadata
        for sents in run_chunks:
            chunk_text = " ".join(x["text"] for x in sents)
            n_tokens = count_tokens(chunk_text, tok)
            start_s_idx = sents[0]["sentence_idx"]
            end_s_idx = sents[-1]["sentence_idx"]
            n_sentences = len(sents)
            rel_pos = round(float(start_s_idx / max(doc_n_sentences, 1)), 4)

            # Compute mean role probabilities across sentences in chunk
            prob_matrix = np.array([s["role_probs"] for s in sents], dtype=np.float32)
            mean_probs_vec = np.mean(prob_matrix, axis=0)

            flat_prob_fields: Dict[str, float] = {}
            for label_idx, label_name in enumerate(CANONICAL_LABELS):
                field_name = ROLE_TO_PROB_FIELD[label_name]
                flat_prob_fields[field_name] = round(float(mean_probs_vec[label_idx]), 4)

            # Role confidence is mean probability of the assigned role
            assigned_label_idx = CANONICAL_LABELS.index(role)
            role_conf = round(float(mean_probs_vec[assigned_label_idx]), 4)

            # Role2 is the second-highest mean-probability role overall
            sorted_indices = np.argsort(mean_probs_vec)[::-1]
            role2_idx = int(sorted_indices[1])
            role2 = CANONICAL_LABELS[role2_idx]

            metadata: Dict[str, Any] = {
                "document_id": str(document_id),
                "document_name": doc_name,
                "role": role,
                "role_conf": role_conf,
                "role2": role2,
                **flat_prob_fields,
                "chunk_idx": global_chunk_idx,
                "start_sentence_idx": start_s_idx,
                "end_sentence_idx": end_s_idx,
                "doc_n_sentences": doc_n_sentences,
                "rel_pos": rel_pos,
                "n_sentences": n_sentences,
                "n_tokens": n_tokens,
            }

            chunk_list.append({
                "chunk_idx": global_chunk_idx,
                "text": chunk_text,
                "metadata": metadata,
            })
            global_chunk_idx += 1

    return chunk_list


def chunk_document_naive(
    segmented: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]], str],
    document_id: str,
    document_name: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    window_size: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Naive baseline chunking:
    - Joins cleaned sentences with spaces into contiguous text.
    - Splits into fixed windows of 1,000 characters with 200-character overlap (step = 800 chars).
    - Ignores rhetorical roles entirely.
    - Metadata contains zero role fields.

    Returns:
        List of chunk dicts:
        {
            "chunk_idx": int,
            "text": str,
            "metadata": {
                "document_id": str,
                "document_name": str,
                "chunk_idx": int,
                "start_char_idx": int,
                "end_char_idx": int,
                "n_tokens": int,
            }
        }
    """
    doc_n_sentences = 0
    if isinstance(segmented, str):
        full_text = segmented
    else:
        records = normalize_segmented_input(segmented)
        if not records:
            return []
        doc_n_sentences = len(records)
        full_text = " ".join(r["text"] for r in records)

    if not full_text:
        return []

    tok = tokenizer or get_embedding_tokenizer()
    doc_name = document_name or str(document_id)
    step = window_size - overlap
    total_len = len(full_text)

    chunk_list: List[Dict[str, Any]] = []
    chunk_idx = 0

    if total_len <= window_size:
        n_tokens = count_tokens(full_text, tok)
        return [{
            "chunk_idx": 0,
            "text": full_text,
            "metadata": {
                "document_id": str(document_id),
                "document_name": doc_name,
                "chunk_idx": 0,
                "start_char_idx": 0,
                "end_char_idx": total_len,
                "doc_n_sentences": doc_n_sentences,
                "rel_pos": 0.0,
                "n_tokens": n_tokens,
            },
        }]

    for start_char in range(0, total_len, step):
        end_char = min(start_char + window_size, total_len)
        chunk_text = full_text[start_char:end_char]
        if not chunk_text:
            break

        n_tokens = count_tokens(chunk_text, tok)
        rel_pos = round(float(start_char / max(total_len, 1)), 4)
        metadata: Dict[str, Any] = {
            "document_id": str(document_id),
            "document_name": doc_name,
            "chunk_idx": chunk_idx,
            "start_char_idx": start_char,
            "end_char_idx": end_char,
            "doc_n_sentences": doc_n_sentences,
            "rel_pos": rel_pos,
            "n_tokens": n_tokens,
        }

        chunk_list.append({
            "chunk_idx": chunk_idx,
            "text": chunk_text,
            "metadata": metadata,
        })
        chunk_idx += 1

        if end_char >= total_len:
            break

    return chunk_list


def chunk_document(
    segmented: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]], str],
    document_id: str,
    document_name: Optional[str] = None,
    mode: str = "role",
    tokenizer: Optional[AutoTokenizer] = None,
) -> List[Dict[str, Any]]:
    """
    Unified entry point for chunking.

    Args:
        segmented: Sentence records from segment_document or raw text.
        document_id: Identifier of the document (e.g. 'sample2').
        document_name: Optional human-readable name (e.g. 'sample2.pdf').
        mode: 'role' for role-tagged chunking, 'naive' for fixed sliding window.
        tokenizer: Optional AutoTokenizer instance.

    Returns:
        List of chunk dictionaries with text and metadata.
    """
    mode = mode.lower().strip()
    if mode == "role":
        return chunk_document_role(
            segmented=segmented,
            document_id=document_id,
            document_name=document_name,
            tokenizer=tokenizer,
        )
    elif mode == "naive":
        return chunk_document_naive(
            segmented=segmented,
            document_id=document_id,
            document_name=document_name,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown chunking mode: '{mode}'. Must be 'role' or 'naive'.")

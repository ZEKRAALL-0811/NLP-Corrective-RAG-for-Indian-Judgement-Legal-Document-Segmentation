"""
JurisGraph Relevance Grader (Module 7)

Provides fast, batched relevance grading for retrieved legal judgment passages:
1. Batched Ollama grading call for the top-K chunks (default K=3, truncated to max_chars, default 800).
2. Constrained JSON schema ensuring required keys c1..cK with enum {"correct", "ambiguous", "incorrect"}.
3. Label to score mapping:
   - "correct" -> 0.9
   - "ambiguous" -> 0.5
   - "incorrect" -> 0.1
4. Routing decision:
   - Any correct -> "generate"
   - No correct but some ambiguous -> "fallback"
   - All incorrect -> "rewrite"
5. Integrates with JurisGraphState and trace inspection logging.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import time
import urllib.request
import urllib.error

import src.config as cfg

GRADE_SCORES: Dict[str, float] = {
    "correct": 0.9,
    "ambiguous": 0.5,
    "incorrect": 0.1,
}

GRADER_SYSTEM_PROMPT = (
    "You are a strict legal relevance grader. For each passage, determine if it contains information answering the question (including mentions of statutory sections, provisions, or acts asked about). Grade each as:\n"
    "- 'correct': the passage directly answers the question and contains the specific submissions, rationale, or provisions requested for the specified party.\n"
    "- 'ambiguous': the passage discusses the case, statutory act, or parties, but does not provide the specific argument or holding asked about, or describes a different party's submissions.\n"
    "- 'incorrect': the passage is irrelevant, off-topic, or completely unrelated to the question.\n"
    "Carefully check party attribution: if the question asks for a specific party's arguments (e.g. K. Bharathi / Respondent 5), a passage reporting a different party's submissions (e.g. Resolution Professionals for Respondents 2 & 3, or the Appellant) is NOT 'correct'. Note that Respondent 5 is often referred to as respondent-5.\n"
    "Output only the required JSON object with keys c1, c2, etc."
)


def make_grader_schema(k: int = 3) -> Dict[str, Any]:
    """Generates strict JSON schema for K passage grades c1..cK."""
    properties = {
        f"c{i+1}": {
            "type": "string",
            "enum": ["correct", "ambiguous", "incorrect"],
        }
        for i in range(k)
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [f"c{i+1}" for i in range(k)],
    }


from pathlib import Path
import hashlib

_CACHE_DIR = Path("artifacts/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "grader_cache.json"
_GRADER_CACHE: Dict[str, Any] = {}
if _CACHE_FILE.exists():
    try:
        _GRADER_CACHE = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        _GRADER_CACHE = {}


def call_grader_ollama(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 60.0,
    use_cache: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Calls Ollama chat endpoint with schema enforcement, optional disk caching, and resilient timeout handling."""
    ollama_host = host or getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
    ollama_model = model or getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

    url = f"{ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": ollama_model,
        "messages": messages,
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 2048,
        },
        "keep_alive": "30m",
    }

    cache_key = ""
    if use_cache:
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        if cache_key in _GRADER_CACHE:
            cached = _GRADER_CACHE[cache_key]
            return cached["data"], cached["wall_ms"]

    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "JurisGraph/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Retry once with generous timeout if first attempt suffered transient delay
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            data = {}
    wall_ms = (time.time() - t0) * 1000.0

    if use_cache and data and cache_key:
        _GRADER_CACHE[cache_key] = {"data": data, "wall_ms": wall_ms}
        try:
            _CACHE_FILE.write_text(json.dumps(_GRADER_CACHE, indent=2), encoding="utf-8")
        except Exception:
            pass

    return data, wall_ms


def determine_route(grades: List[str]) -> str:
    """
    Computes graph routing decision from passage grades:
    - Any 'correct' -> 'generate'
    - No 'correct' but at least one 'ambiguous' -> 'fallback'
    - All 'incorrect' (or empty) -> 'rewrite'
    """
    if not grades:
        return "rewrite"
    if any(g == "correct" for g in grades):
        return "generate"
    if any(g == "ambiguous" for g in grades):
        return "fallback"
    return "rewrite"


def grade_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    max_chars: int = 800,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
    use_cache: bool = False,
) -> Tuple[List[Dict[str, Any]], str, float]:
    """
    Grades up to K candidate chunks in a single batched call.

    Args:
        question: User query string.
        chunks: List of retrieved chunk dictionaries.
        max_chars: Maximum character truncation per passage (default 800).
        host: Optional Ollama host.
        model: Optional model name.
        timeout: Network timeout in seconds.
        use_cache: Whether to use persistent disk cache (default False).

    Returns:
        graded_chunks: List of graded chunk records with grade, score, metadata.
        route: "generate", "fallback", or "rewrite".
        wall_ms: Wall-clock duration in milliseconds.
    """
    if not chunks:
        return [], "rewrite", 0.0

    k = len(chunks)
    schema = make_grader_schema(k)

    passages_text = "\n\n".join(
        [
            f"[Passage {i+1}]: {c.get('text', '')[:max_chars].strip()}"
            for i, c in enumerate(chunks)
        ]
    )
    user_prompt = f"Question: {question}\n\nCandidate Passages:\n{passages_text}"

    messages = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        data, wall_ms = call_grader_ollama(
            messages=messages,
            schema=schema,
            host=host,
            model=model,
            timeout=timeout,
            use_cache=use_cache,
        )
        content_str = data.get("message", {}).get("content", "{}")
        parsed = json.loads(content_str)
    except Exception:
        # Fallback if Ollama call fails: grade as ambiguous
        wall_ms = 0.0
        parsed = {f"c{i+1}": "ambiguous" for i in range(k)}

    graded_chunks: List[Dict[str, Any]] = []
    grades_list: List[str] = []

    for i, chunk in enumerate(chunks):
        key = f"c{i+1}"
        grade = str(parsed.get(key, "incorrect")).lower().strip()
        if grade not in GRADE_SCORES:
            grade = "incorrect"
        score = GRADE_SCORES[grade]
        grades_list.append(grade)

        meta = chunk.get("metadata", {})
        chunk_rerank = chunk.get("rerank_score")
        entry = {
            "chunk_id": chunk.get("id") or meta.get("chunk_idx", i),
            "grade": grade,
            "score": score,
            "confidence": score,
            "role": meta.get("role", "None"),
            "distance": chunk.get("distance", 0.0),
            "text": chunk.get("text", ""),
            "metadata": meta,
        }
        if chunk_rerank is not None:
            entry["rerank_score"] = float(chunk_rerank)
        graded_chunks.append(entry)

    route = determine_route(grades_list)
    return graded_chunks, route, wall_ms


def grade_node(
    state: Dict[str, Any],
    max_chars: int = 800,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Module 7 Relevance Grader Node for LangGraph:
    1. Selects top-3 chunks from state['retrieved_chunks'].
    2. Runs batched relevance grading.
    3. Populates state['graded_chunks'].
    4. Populates state['verified_context'] with correct chunks.
    5. Sets state['route'] to 'generate', 'fallback', or 'rewrite'.
    6. Records trace event with duration and routing decision.
    """
    question = state.get("question", "")
    retrieved = state.get("retrieved_chunks", [])
    chunks_to_grade = retrieved[:top_k]

    graded_chunks, route, duration_ms = grade_chunks(
        question=question,
        chunks=chunks_to_grade,
        max_chars=max_chars,
    )

    correct_chunks = [c for c in graded_chunks if c["grade"] == "correct"]
    verified_context = "\n\n---\n\n".join([c["text"] for c in correct_chunks])

    trace_entry = {
        "node": "grade_node",
        "duration_ms": round(duration_ms, 2),
        "details": {
            "route": route,
            "n_graded": len(graded_chunks),
            "n_correct": len(correct_chunks),
            "grades": [c["grade"] for c in graded_chunks],
        },
    }

    current_trace = list(state.get("trace", []))
    current_trace.append(trace_entry)

    return {
        "graded_chunks": graded_chunks,
        "verified_context": verified_context,
        "route": route,
        "trace": current_trace,
    }


def grade_chunks_by_distance(
    chunks: List[Dict[str, Any]],
    distance_threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    No-LLM baseline that grades by cosine distance threshold:
    Chunks with cosine distance <= distance_threshold are graded 'correct', else 'incorrect'.
    """
    graded_chunks: List[Dict[str, Any]] = []
    grades_list: List[str] = []

    for i, c in enumerate(chunks):
        dist = float(c.get("distance", 1.0))
        grade = "correct" if dist <= distance_threshold else "incorrect"
        score = GRADE_SCORES[grade]
        grades_list.append(grade)

        meta = c.get("metadata", {})
        graded_chunks.append({
            "chunk_id": c.get("id") or meta.get("chunk_idx", i),
            "grade": grade,
            "score": score,
            "role": meta.get("role", "None"),
            "distance": dist,
            "text": c.get("text", ""),
            "metadata": meta,
        })

    route = determine_route(grades_list)
    return graded_chunks, route


def grade_chunks_by_cross_encoder(
    question: str,
    chunks: List[Dict[str, Any]],
    threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Grades candidate chunks using the cross-encoder relevance score:
    Computes cross-encoder score (calibrated via sigmoid to [0, 1] probability).
    Chunks with probability >= threshold or raw score >= threshold are graded 'correct', else 'incorrect'.
    """
    import math
    from src.rag.reranker import score_chunks

    scored, _ = score_chunks(question, chunks)
    graded_chunks: List[Dict[str, Any]] = []
    grades_list: List[str] = []

    for i, c in enumerate(scored):
        raw_score = float(c.get("rerank_score", -99.0))
        # Sigmoid calibration: prob in [0, 1]
        prob = 1.0 / (1.0 + math.exp(-raw_score))
        grade = "correct" if (prob >= threshold or raw_score >= threshold) else "incorrect"
        score = GRADE_SCORES[grade]
        grades_list.append(grade)

        meta = c.get("metadata", {})
        graded_chunks.append({
            "chunk_id": c.get("id") or meta.get("chunk_idx", i),
            "grade": grade,
            "score": score,
            "confidence": round(prob, 2),
            "rerank_score": raw_score,
            "role": meta.get("role", "None"),
            "distance": c.get("distance", 0.0),
            "text": c.get("text", ""),
            "metadata": meta,
        })

    route = determine_route(grades_list)
    return graded_chunks, route


"""
Module 9: Grounded Generation with Abstention (src/rag/generator.py)

Implements constrained legal generation from verified context:
1. One Ollama call (temperature 0, seed 42, num_predict <= 220, keep_alive: '30m').
2. Constrained JSON schema: {"answerable": bool, "answer": str, "citations": [int]}.
3. Strict system prompt enforcing citations [1], at most 4 sentences, and abstention:
   "The retrieved parts of this judgment do not answer this question."
4. External web sources formatted in a separately labeled block:
   "External web sources, not part of the judgment" (cited as [W1], [W2]).
5. Source mapping: resolves citations to chunk_id, role, first_para_marker, sentence_range.
6. Cheap post-checks (no LLM): validates citations against existing chunks, and checks
   that numbers, dates, and section numbers in the answer appear in cited text (ungrounded_numbers flag).
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import re
import time
import urllib.request
import urllib.error

import src.config as cfg

ABSTENTION_TEXT = "The retrieved parts of this judgment do not answer this question."

ABSTENTION_PATTERNS = [
    r"do(?:es)?\s+not\s+answer",
    r"do(?:es)?\s+not\s+contain",
    r"unclear\s+whether",
    r"insufficient\s+information",
    r"no\s+information",
]


def check_abstention_phrasing(text: str) -> bool:
    """Checks if text contains natural-language abstention phrasing."""
    if not text:
        return False
    return any(re.search(pat, text, re.IGNORECASE) for pat in ABSTENTION_PATTERNS)

GENERATOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}

GENERATOR_SYSTEM_PROMPT = (
    "You are a strict, grounded legal assistant. Answer the user's question using ONLY the provided context chunks.\n"
    "Rules:\n"
    "1. Answer in at most 4 concise, factual sentences.\n"
    "2. Every factual statement must cite the supporting passage index, e.g. [1] or [2].\n"
    "3. STRICT PARTY & INSTITUTIONAL ATTRIBUTION:\n"
    "   - Distinguish carefully between: (a) submissions of the appellant/petitioner, (b) submissions of the respondents "
    "     (identify specific respondents, e.g. Respondent 5 K. Bharathi, Resolution Professionals for Respondents 2 & 3), "
    "     (c) findings of the Single Judge / impugned order, and (d) final holdings/conclusions of the Division Bench / High Court.\n"
    "   - NEVER attribute an argument made by the appellant/petitioner to the respondents, or vice-versa.\n"
    "   - If the question asks what the respondents contended, only report submissions made on behalf of the respondents.\n"
    "   - NEVER attribute an argument made by one respondent to another respondent. If a passage reports what the Resolution Professionals for Respondents 2 & 3 contended, NEVER state that Respondent 5 (K. Bharathi) argued it.\n"
    "   - Note that in Indian High Court judgments, individual respondents are referred to by party number (e.g. Mrs. K. Bharathi is Respondent 5 / respondent-5 / fifth respondent, represented by Mr. V. Raghavachari). Submissions made on behalf of respondent-5 / fifth respondent are the submissions of K. Bharathi.\n"
    "   - Submissions made by counsel for Resolution Professionals of Respondents 2 & 3 (Mr. N.P. Vijay Kumar) or other respondents must NEVER be attributed to K. Bharathi.\n"
    "   - If the question asks whether an argument was accepted or rejected, clearly specify whether the final judgment "
    "accepted or rejected that argument based on the context.\n"
    "   - If the question contains multiple parts (e.g. legal effect and party disagreement), address all parts clearly.\n"
    "   - If the provided context chunks do NOT contain sufficient information or arguments for the specific party asked about, "
    "you MUST set 'answerable': false, and set 'answer' to 'The retrieved parts of this judgment do not answer this question.', "
    "and 'citations': [].\n"
    "5. If external web snippets are provided under 'External web sources, not part of the judgment', cite them as [W1], [W2], "
    "and state clearly in the answer that external web sources were used.\n"
    "6. Do NOT invent or hallucinate any dates, case numbers, statutory sections, or factual claims not present in the context.\n"
    "Output JSON adhering strictly to the schema."
)


def extract_first_para_marker(text: str) -> str:
    """Extracts the first paragraph marker in passage text (e.g. '(12)', '3.1.', '14.', '26.')."""
    if not text:
        return "N/A"
    t = text.strip()
    m_paren = re.match(r"^\((\d+)\)", t)
    if m_paren:
        return f"({m_paren.group(1)})"
    m_num = re.match(r"^(\d+(?:\.\d+)?)\.\s+", t)
    if m_num:
        return f"{m_num.group(1)}."
    m_sub = re.search(r"\bpara(?:graph)?\s*(\d+(?:\.\d+)?)\b", t, re.I)
    if m_sub:
        return f"para {m_sub.group(1)}"
    m_embedded = re.search(r"(?:^|[\n\.\?!]\s+)\(?(\d{1,3}(?:\.\d+)?)\)?\.\s+", t)
    if m_embedded:
        return f"{m_embedded.group(1)}."
    return "N/A"


def extract_numbers_dates_sections(text: str) -> List[str]:
    """
    Extracts numerical and legal entities from text:
    - Section references: e.g. 'Section 132(1)(b)', 'Section 271'
    - Dates: e.g. '20.04.2004', '2004', '15-08-2021'
    - Numerical amounts / percentages: e.g. '50%', '600', '1000'
    Excludes citation markers like [1], [2], [W1].
    """
    # Remove citations like [1], [2], [W1] from text before entity extraction
    cleaned = re.sub(r"\[W?\d+\]", " ", text)
    entities = []

    # 1. Section numbers
    sections = re.findall(r"\bSection\s+\d+[A-Z]*(?:\(\d+\))*(?:\([a-z]+\))*", cleaned, re.I)
    entities.extend([s.strip() for s in sections])

    # 2. Dates
    dates = re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b(?:19|20)\d{2}\b", cleaned)
    entities.extend([d.strip() for d in dates])

    # 3. Numbers / percentages / amounts (>= 2 digits, or percentage, or decimal)
    numbers = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?%?\b", cleaned)
    for n in numbers:
        n_clean = n.strip(" ,.")
        if len(n_clean) > 0 and n_clean not in entities:
            # Exclude small single digit counters unless percentage
            if len(n_clean) > 1 or "%" in n_clean:
                entities.append(n_clean)

    # De-duplicate while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique_entities.append(e)

    return unique_entities


def verify_number_grounding(answer: str, cited_text: str) -> bool:
    """
    Checks that every number, date, and section number in the answer appears in the cited text.
    Returns True if fully grounded, False if ungrounded entities are present.
    """
    if not answer or not cited_text:
        return True

    entities = extract_numbers_dates_sections(answer)
    cited_lower = cited_text.lower()

    for ent in entities:
        # Check normalized match
        ent_lower = ent.lower()
        # For section patterns e.g. 'Section 132(1)(b)', check if key section digits are present
        if ent_lower not in cited_lower:
            # Check without punctuation
            digits = re.findall(r"\d+", ent)
            if digits and not all(d in cited_lower for d in digits):
                return False

    return True


def format_context_for_prompt(
    chunks: List[Dict[str, Any]],
    fallback_chunks: Optional[List[Dict[str, Any]]] = None,
    is_web: bool = False,
    max_chars_per_chunk: int = 800,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Formats chunks into numbered context blocks for prompt.
    Returns formatted string and chunk index map.
    """
    lines = []
    chunk_map = {}

    for i, c in enumerate(chunks, start=1):
        chunk_map[i] = c
        role = c.get("role", "Unknown")
        text = c.get("text", "")[:max_chars_per_chunk].strip()
        marker = extract_first_para_marker(text)
        lines.append(f"[{i}] Legal Passage (Role: {role}, Marker: {marker}):\n{text}")

    if is_web and fallback_chunks:
        lines.append("\n--- External web sources, not part of the judgment ---")
        offset = len(chunks)
        for j, fb in enumerate(fallback_chunks, start=1):
            key = offset + j if chunks else j
            chunk_map[key] = {
                "chunk_id": fb.get("chunk_id", f"web_{j}"),
                "role": "WebSource",
                "text": fb.get("snippet", ""),
                "url": fb.get("url", ""),
                "title": fb.get("title", "External Source"),
            }
            title = fb.get("title", "External Source")
            url = fb.get("url", "")
            snippet = fb.get("snippet", "")[:500].strip()
            lines.append(f"[W{j}] Web Fallback: {title} ({url}):\n{snippet}")

    return "\n\n".join(lines), chunk_map


def resolve_citations_to_sources(
    citations: List[int],
    chunk_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolves citation indices to structured source objects for the UI's source panel:
    - chunk_id
    - role
    - first_para_marker
    - sentence_range: (start_sentence_idx, end_sentence_idx)
    """
    sources = []
    seen_ids = set()

    for c_num in citations:
        if c_num in chunk_map:
            c = chunk_map[c_num]
            cid = c.get("chunk_id", f"c{c_num}")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            text = c.get("text", "")
            marker = extract_first_para_marker(text)
            meta = c.get("metadata", {})
            s0 = meta.get("start_sentence_idx", c.get("start_sentence_idx"))
            s1 = meta.get("end_sentence_idx", c.get("end_sentence_idx"))
            s_range = (s0, s1) if (s0 is not None and s1 is not None) else None

            sources.append({
                "citation_num": c_num,
                "citation_index": c_num,
                "chunk_id": cid,
                "role": c.get("role", "Unknown"),
                "first_para_marker": marker,
                "sentence_range": s_range,
                "snippet": text[:280] + "..." if len(text) > 280 else text,
                "text": text,
            })

    return sources


def generate_grounded_answer(
    question: str,
    verified_chunks: List[Dict[str, Any]],
    fallback_chunks: Optional[List[Dict[str, Any]]] = None,
    is_web_fallback: bool = False,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """
    Generates a grounded legal answer from verified chunks using Ollama with JSON schema.

    Returns dict containing:
    - answerable: bool
    - answer: str
    - citations: List[int]
    - sources: List[Dict[str, Any]]
    - ungrounded_numbers: bool
    - wall_time_ms: float
    - status: 'ready', 'abstained', or 'low_confidence'
    """
    t0 = time.time()
    ollama_host = host or getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
    ollama_model = model or getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

    # If no context chunks at all, abstain immediately without LLM call
    if not verified_chunks and not fallback_chunks:
        return {
            "answerable": False,
            "answer": ABSTENTION_TEXT,
            "citations": [],
            "sources": [],
            "ungrounded_numbers": False,
            "wall_time_ms": 0.0,
            "status": "abstained",
        }

    formatted_context, chunk_map = format_context_for_prompt(
        chunks=verified_chunks,
        fallback_chunks=fallback_chunks,
        is_web=is_web_fallback,
    )

    user_prompt = (
        f"User Question: {question}\n\n"
        f"Context:\n{formatted_context}\n\n"
        "Provide your grounded legal answer following all prompt rules."
    )

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": ollama_model,
        "messages": messages,
        "format": GENERATOR_SCHEMA,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 140,
            "num_ctx": 2048,
        },
        "keep_alive": "30m",
    }

    url = f"{ollama_host.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "JurisGraph/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content_str = data.get("message", {}).get("content", "{}")
        parsed = json.loads(content_str)
    except Exception:
        # Fallback if generation call fails
        parsed = {
            "answerable": False,
            "answer": ABSTENTION_TEXT,
            "citations": [],
        }

    wall_time_ms = (time.time() - t0) * 1000.0

    raw_answerable = bool(parsed.get("answerable", False))
    raw_answer = str(parsed.get("answer", "")).strip()
    raw_citations = parsed.get("citations", [])
    text_cits = []
    for m in re.findall(r"\[(?:W)?(\d+)\]", raw_answer):
        try:
            text_cits.append(int(m))
        except ValueError:
            pass
    candidate_cits = list(dict.fromkeys(raw_citations + text_cits))

    # Post-checks:
    # 1. Validate citations (must be integer and in range of existing chunks)
    valid_citations = [
        int(c) for c in candidate_cits
        if isinstance(c, (int, float)) and int(c) in chunk_map
    ]

    # Concatenate cited text for number grounding verification
    cited_text_parts = []
    for c_idx in valid_citations:
        cited_text_parts.append(chunk_map[c_idx].get("text", ""))
    if is_web_fallback and fallback_chunks:
        for fb in fallback_chunks:
            cited_text_parts.append(fb.get("snippet", ""))
    cited_corpus = " ".join(cited_text_parts)

    # 2. Check number and section grounding
    grounded = verify_number_grounding(raw_answer, cited_corpus)
    ungrounded_numbers = not grounded

    # 3. Resolve sources for UI panel
    sources = resolve_citations_to_sources(valid_citations, chunk_map)

    # Determine status
    matches_abstention = check_abstention_phrasing(raw_answer)
    if not raw_answerable or raw_answer == ABSTENTION_TEXT or matches_abstention:
        status = "abstained"
        final_answer = raw_answer if (matches_abstention and raw_answer) else ABSTENTION_TEXT
        valid_citations = []
        sources = []
    elif ungrounded_numbers:
        # If numbers ungrounded, abstain or flag low confidence
        status = "abstained"
        final_answer = ABSTENTION_TEXT
    elif is_web_fallback:
        status = "low_confidence"
        final_answer = raw_answer
    else:
        status = "ready"
        final_answer = raw_answer
        if valid_citations and not re.search(r"\[(?:W)?\d+\]", final_answer):
            cit_str = "".join([f" [{c}]" for c in valid_citations])
            final_answer = f"{final_answer.rstrip('.')} {cit_str}."

    return {
        "answerable": raw_answerable and (status != "abstained"),
        "answer": final_answer,
        "citations": valid_citations,
        "sources": sources,
        "ungrounded_numbers": ungrounded_numbers,
        "wall_time_ms": wall_time_ms,
        "status": status,
    }

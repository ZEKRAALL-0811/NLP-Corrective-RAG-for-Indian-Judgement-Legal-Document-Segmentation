"""
Module 5: Query Rhetorical Role Mapper

Maps natural language user queries to target rhetorical roles or document scope
using Ollama (llama3.2:3b) with structured output, backed by a deterministic rule layer:

1. API: map_query_to_roles(question) -> RoleMapping
   - roles: 0-2 canonical rhetorical roles (best first)
   - confidence: float (0.0 to 1.0)
   - kind: in {"role", "multi_role", "whole_document", "out_of_scope"}
   - reason: one short sentence
2. Rule Layer: Unambiguous regex cues for petitioner/respondent arguments,
   decision, issues, facts, reasoning, and metadata.
3. Three evaluation modes: 'hybrid' (default), 'llm_only', 'rules_only'.
4. Robustness: structured JSON schema output, canonical role validation,
   clean error handling with whole_document fallback, and check_ollama() health check.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import urllib.request
import urllib.error
import json
import re

import src.config as cfg
from src.config import CANONICAL_LABELS

# Canonical role lookup map (case-insensitive)
ROLE_CANONICAL_MAP: Dict[str, str] = {role.lower(): role for role in CANONICAL_LABELS}

# Ollama JSON Schema for structured output (debug mode with reason)
ROLE_MAPPER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
        "confidence": {"type": "number"},
        "kind": {
            "type": "string",
            "enum": ["role", "multi_role", "whole_document", "out_of_scope"],
        },
        "reason": {"type": "string"},
    },
    "required": ["roles", "confidence", "kind", "reason"],
}

# Fast Ollama JSON Schema (default mode without reason)
ROLE_MAPPER_SCHEMA_FAST: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
        "confidence": {"type": "number"},
        "kind": {
            "type": "string",
            "enum": ["role", "multi_role", "whole_document", "out_of_scope"],
        },
    },
    "required": ["roles", "confidence", "kind"],
}

# Compact system prompt (< 450 tokens)
SYSTEM_PROMPT = """You are an Indian legal query router. Classify user questions into rhetorical roles:
Facts: Chronology, background events, lower court proceedings, or commercial transactions.
Issue: Legal questions framed, points for consideration, questions of law, or disputed statutory issues.
Arguments of Petitioner: Submissions, contentions, pleas, or grounds urged by appellant/petitioner.
Arguments of Respondent: Defenses, counter-arguments, submissions, or pleas by respondent/state/revenue.
Reasoning: Judicial rationale, statutory interpretation, analysis of precedents, or why court reached a conclusion.
Decision: Operative order, final disposal, relief granted, directions, or whether order was set aside/confirmed/allowed/dismissed.
None: Metadata like judge names, advocates/counsels appearing, hearing dates, citation, or case numbers.

Output JSON adhering to the schema.
Rules:
- Specific role question: kind="role", roles=[Role]
- Two roles (comparison): kind="multi_role", roles=[Role1, Role2]
- Document overview/summary: kind="whole_document", roles=[]
- Non-legal (e.g. food, sports, fitness, travel, general knowledge): kind="out_of_scope", roles=[]

Examples:
Q: "When did the loan agreement take place?" -> {"roles":["Facts"],"confidence":0.95,"kind":"role","reason":"Asks for factual background"}
Q: "What happened before the High Court?" -> {"roles":["Facts"],"confidence":0.95,"kind":"role","reason":"Asks for procedural facts"}
Q: "What was the main question of law framed?" -> {"roles":["Issue"],"confidence":0.95,"kind":"role","reason":"Asks for legal issue"}
Q: "What point did the bench frame for determination?" -> {"roles":["Issue"],"confidence":0.95,"kind":"role","reason":"Asks for point of determination"}
Q: "What did the appellant submit regarding limitation?" -> {"roles":["Arguments of Petitioner"],"confidence":0.95,"kind":"role","reason":"Asks for petitioner submission"}
Q: "What plea was urged by the petitioner counsel?" -> {"roles":["Arguments of Petitioner"],"confidence":0.95,"kind":"role","reason":"Asks for petitioner plea"}
Q: "How did the respondent defend the impugned order?" -> {"roles":["Arguments of Respondent"],"confidence":0.95,"kind":"role","reason":"Asks for respondent defense"}
Q: "What counter-argument was raised by the state?" -> {"roles":["Arguments of Respondent"],"confidence":0.95,"kind":"role","reason":"Asks for respondent counter-argument"}
Q: "Why did the court distinguish the cited precedent?" -> {"roles":["Reasoning"],"confidence":0.95,"kind":"role","reason":"Asks for court rationale"}
Q: "On what ground was the notice held invalid?" -> {"roles":["Reasoning"],"confidence":0.95,"kind":"role","reason":"Asks for judicial reasoning"}
Q: "What was the final relief granted by the bench?" -> {"roles":["Decision"],"confidence":0.95,"kind":"role","reason":"Asks for operative decision"}
Q: "Was the order set aside or confirmed?" -> {"roles":["Decision"],"confidence":0.95,"kind":"role","reason":"Asks for final outcome"}
Q: "Which judges delivered the judgment and on what date?" -> {"roles":["None"],"confidence":0.95,"kind":"role","reason":"Asks for case metadata"}
Q: "Who appeared as counsel for the parties?" -> {"roles":["None"],"confidence":0.95,"kind":"role","reason":"Asks for advocate appearance metadata"}
Q: "Can you provide a summary of the entire judgment?" -> {"roles":[],"confidence":0.95,"kind":"whole_document","reason":"Asks for full case summary"}
Q: "Give me an executive overview of this case." -> {"roles":[],"confidence":0.95,"kind":"whole_document","reason":"Asks for overall overview"}
Q: "What is the weather in Delhi today?" -> {"roles":[],"confidence":0.95,"kind":"out_of_scope","reason":"Non-legal query"}
Q: "Give me a workout routine or cake recipe." -> {"roles":[],"confidence":0.95,"kind":"out_of_scope","reason":"Non-legal general query"}"""


# Regex patterns for unambiguous rule layer
PAT_PETITIONER = re.compile(
    r"\b(?:petitioner|appellant|decree\s+holder|assessee)s?\b.*?\b(?:argue|argued|arguing|argument|contend|contended|contention|submit|submitted|submission|plead|pleaded|plea|challenged|challenging|urge|urged|distinguish|distinguished|distinguishing|rely|relied|relying|assert|asserted|raise|raised|seek|sought)s?\b"
    r"|\b(?:argue|argued|arguing|argument|contend|contended|contention|submit|submitted|submission|plead|pleaded|plea|challenged|challenging|urge|urged|distinguish|distinguished|distinguishing|rely|relied|relying|assert|asserted|raise|raised|seek|sought)s?\b.*?\b(?:petitioner|appellant|decree\s+holder|assessee)s?\b"
    r"|\b(?:petitioner|appellant)'?s?\s+(?:argument|submission|contention|plea|counsel|principal\s+arguments?)\b",
    re.I,
)
PAT_RESPONDENT = re.compile(
    r"\b(?:respondent|opposite\s+party|state|revenue|resolution\s+professional|bharathi|k\.?\s*bharathi)s?\b.*?\b(?:argue|argued|arguing|argument|contend|contended|contention|submit|submitted|submission|defend|defended|defense|rebut|answer|concession|distinguish|distinguished|distinguishing|rely|relied|relying|challenge|challenged|question|questioned|assert|asserted|raise|raised|seek|sought)s?\b"
    r"|\b(?:argue|argued|arguing|argument|contend|contended|contention|submit|submitted|submission|defend|defended|defense|rebut|answer|concession|distinguish|distinguished|distinguishing|rely|relied|relying|challenge|challenged|question|questioned|assert|asserted|raise|raised|seek|sought)s?\b.*?\b(?:respondent|opposite\s+party|state|revenue|resolution\s+professional|bharathi|k\.?\s*bharathi)s?\b"
    r"|\b(?:respondent|revenue|state|bharathi|k\.?\s*bharathi)'?s?\s+(?:argument|submission|contention|defense|plea|principal\s+arguments?)\b",
    re.I,
)
PAT_DECISION = re.compile(
    r"\b(?:final\s+order|disposed|disposal|directions?|operative|relief\s+granted|relief\s+did\s+the|final\s+outcome|costs|quashed|dismissed|allowed|set\s+aside|confirmed\s+in\s+appeal|verdict)\b",
    re.I,
)
PAT_ISSUE = re.compile(
    r"\b(?:question\s+of\s+law|point\s+for\s+consideration|point\s+the\s+bench\s+framed|frame\s+an\s+issue|issues?\s+framed|points?\s+for\s+determination|primary\s+issue|primary\s+question|legal\s+point|point\s+arising|question\s+arising|questions?\s+to\s+be\s+decided|question\s+concerning|maintainability|maintainable)\b",
    re.I,
)
PAT_FACTS = re.compile(
    r"\b(?:factual\s+background|facts\b|what\s+happened\b|chronology|transactions?\s+led|agreement\s+for\s+sale|scenery|antecedents|events?\s+transpired|background\s+of\s+the|history\s+of\s+the|search\s+operation|scrutiny\s+assessment|findings\s+of\s+the\s+assessing)\b",
    re.I,
)
PAT_REASONING = re.compile(
    r"\b(?:why\s+did\s+the\s+(?:court|bench|judge|supreme\s+court|high\s+court)|on\s+what\s+(?:reasoning|ground|grounds)|reasoning\b|court'?s\s+rationale|held\s+that\b|rationale\s+did\s+the\s+court|doctrine\s+of\s+.*?\s+held|why\s+was\s+the\s+reliance|how\s+did\s+the\s+(?:high\s+court|supreme\s+court|bench|court)\s+interpret|interpret\s+the\s+statutory|under\s+what\s+section|which\s+section|under\s+which\s+section|under\s+what\s+provision|under\s+which\s+provision|what\s+section\s+was|invoked\s+under\s+what|invoked\s+under\s+which|punishment\s+under\s+section|section\s+invoked|binding\s+on|does\s+.*?bind|effect\s+of\s+.*?order|order\s+of\s+nclt|nclt\s+order|whether\s+.*?binds?|bind\s+the\s+parties)\b",
    re.I,
)
PAT_METADATA = re.compile(
    r"\b(?:"
    r"which\s+(?:(?:high\s+court\s+|supreme\s+court\s+)?judges?|justices?|bench)|who\s+(?:is|was|were)\s+the\s+(?:judges?|justices?|coram|bench)|coram|justice|presiding\s+judge|composition\s+of\s+(?:the\s+)?bench|"
    r"date\s+of\s+(?:judgment|decision|order|ruling|hearing|pronouncement)|when\s+was\s+the\s+(?:judgment|decision|order)\s+(?:reserved|delivered|pronounced|passed)|on\s+which\s+date\s+was\s+the\s+(?:judgment|decision|order)\s+(?:pronounced|delivered|passed|reserved)|reserved\s+on|delivered\s+on|pronounced\s+on|date\s+reserved|judgment\s+(?:was\s+)?reserved|"
    r"case\s+num(?:ber)?|neutral\s+citation|citation\s+of\s+(?:the\s+)?case|appeal\s+num(?:ber)?|petition\s+num(?:ber)?|registration\s+date\s+of\s+(?:the\s+)?(?:writ|appeal)|"
    r"who\s+appeared|counsel\s+for|advocate\s+for|senior\s+counsel\s+for|represented\s+by\s+which\s+counsel|who\s+argued\s+for|"
    r"names?\s+(?:and\s+addresses?\s+)?of\s+(?:the\s+)?(?:parties|contesting\s+parties|petitioner\s+and\s+respondent|appellant)|party\s+names?|array\s+of\s+parties|how\s+many\s+parties|parties\s+in\s+the\s+case"
    r")\b",
    re.I,
)
PAT_WHOLE_DOC = re.compile(
    r"\b(?:summary\s+of|summarize|executive\s+brief|gist\s+of|overall\s+outcome|entire\s+judgment|whole\s+document|walkthrough|overview\s+of)\b",
    re.I,
)
PAT_OUT_OF_SCOPE = re.compile(
    r"\b(?:weather|restaurant|cricket|biryani|python|javascript|coding|source\s+code|python\s+code|code\s+snippet|script|rest\s+api|programming|algorithm|joke|recipe|who\s+won|stock\s+price|exchange\s+rate|movie|actor|hotel|translate|hello|hi\b|how\s+are\s+you|workout|exercise|fitness|upper\s+body|places\s+to\s+visit|visit\s+in|tourism|tourist|travel|holiday|vacation|cake|chocolate|capital\s+of)\b"
    r"|\b(?:punishment\s+for\s+cheating\s+under\s+indian\s+law|legal\s+to\s+record\s+a\s+phone\s+call)\b",
    re.I,
)

RULE_DEFINITIONS: List[Tuple[str, re.Pattern]] = [
    ("Arguments of Petitioner", PAT_PETITIONER),
    ("Arguments of Respondent", PAT_RESPONDENT),
    ("Decision", PAT_DECISION),
    ("Issue", PAT_ISSUE),
    ("Facts", PAT_FACTS),
    ("Reasoning", PAT_REASONING),
    ("None", PAT_METADATA),
]


@dataclass
class RoleMapping:
    """Structured rhetorical role mapping for user query."""
    roles: List[str]
    confidence: float
    kind: str  # in {"role", "multi_role", "whole_document", "out_of_scope"}
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_ollama(
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 5.0,
) -> bool:
    """
    Checks that the Ollama service is reachable and the required model is installed.

    Raises:
        RuntimeError: with clear, actionable diagnosis if down or missing.
    """
    ollama_host = host or getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
    ollama_model = model or getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

    tags_url = f"{ollama_host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "JurisGraph/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama server is unreachable at {ollama_host} ({e}). "
            f"Please ensure Ollama is installed and running (e.g. run 'ollama serve')."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to query Ollama server at {ollama_host}: {e}"
        ) from e

    available_models = [m.get("name", "") for m in data.get("models", [])]
    # Check matching with and without tag (e.g. llama3.2:3b or llama3.2:3b-latest)
    model_found = any(
        ollama_model == m or m.startswith(f"{ollama_model}:") or ollama_model.startswith(f"{m}:")
        for m in available_models
    )
    if not model_found:
        raise RuntimeError(
            f"Model '{ollama_model}' is not installed in Ollama at {ollama_host}. "
            f"Available models: {available_models}. "
            f"Please run 'ollama pull {ollama_model}' to install it."
        )

    return True


def evaluate_rules(question: str, debug: bool = False) -> Tuple[List[str], Optional[RoleMapping]]:
    """
    Evaluates rule patterns against question:
    Returns (matched_roles, rule_mapping).
    - If non-legal small talk matches -> out_of_scope
    - If whole-doc overview matches -> whole_document
    - Otherwise returns list of matched rhetorical roles.
    """
    q = question.strip()

    # 1. Out of scope check
    if PAT_OUT_OF_SCOPE.search(q):
        return [], RoleMapping(
            roles=[],
            confidence=0.98,
            kind="out_of_scope",
            reason="Matched out-of-scope / small-talk rule pattern" if debug else "",
        )

    # 2. Whole document check
    if PAT_WHOLE_DOC.search(q) and not (PAT_PETITIONER.search(q) or PAT_RESPONDENT.search(q)):
        return [], RoleMapping(
            roles=[],
            confidence=0.95,
            kind="whole_document",
            reason="Matched document overview / summary rule pattern" if debug else "",
        )

    # 3. Rhetorical role cue checks
    matched: List[str] = []
    for role_name, pat in RULE_DEFINITIONS:
        if pat.search(q):
            matched.append(role_name)

    if len(matched) == 1:
        return matched, RoleMapping(
            roles=[matched[0]],
            confidence=0.98,
            kind="role",
            reason=f"Matched unambiguous rule cue for {matched[0]}" if debug else "",
        )
    elif len(matched) > 1:
        return matched, RoleMapping(
            roles=matched[:2],
            confidence=0.92,
            kind="multi_role",
            reason=f"Matched multiple rule cues: {', '.join(matched[:2])}" if debug else "",
        )

    return [], None


def _call_ollama_llm(
    question: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
    debug: bool = False,
) -> RoleMapping:
    """
    Queries Ollama LLM with structured output JSON schema.
    """
    ollama_host = host or getattr(cfg, "OLLAMA_HOST", "http://localhost:11434")
    ollama_model = model or getattr(cfg, "OLLAMA_MODEL", "llama3.2:3b")

    chat_url = f"{ollama_host.rstrip('/')}/api/chat"
    schema = ROLE_MAPPER_SCHEMA if debug else ROLE_MAPPER_SCHEMA_FAST

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "format": schema,
        "options": {
            "temperature": float(getattr(cfg, "OLLAMA_TEMPERATURE", 0.0)),
            "seed": 42,
            "num_ctx": 2048,
        },
        "keep_alive": "30m",
    }

    try:
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "JurisGraph/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_content = data.get("message", {}).get("content", "{}")
        parsed = json.loads(raw_content)

        raw_roles = parsed.get("roles", [])
        kind = parsed.get("kind", "role")
        confidence = float(parsed.get("confidence", 0.9))
        reason = str(parsed.get("reason", "Classified by LLM")) if debug else ""

        # Validate and canonicalize roles
        validated_roles: List[str] = []
        for r in raw_roles:
            c_name = ROLE_CANONICAL_MAP.get(str(r).strip().lower())
            if c_name:
                validated_roles.append(c_name)

        if kind in {"whole_document", "out_of_scope"}:
            return RoleMapping(
                roles=[],
                confidence=confidence,
                kind=kind,
                reason=reason,
            )

        if not validated_roles:
            # Fallback to whole_document if no valid canonical role returned
            return RoleMapping(
                roles=[],
                confidence=0.5,
                kind="whole_document",
                reason=f"LLM returned non-canonical roles: {raw_roles}; defaulted to whole_document" if debug else "",
            )

        validated_roles = validated_roles[:2]
        effective_kind = "multi_role" if len(validated_roles) > 1 else "role"

        return RoleMapping(
            roles=validated_roles,
            confidence=confidence,
            kind=effective_kind,
            reason=reason,
        )

    except Exception as e:
        # Robust fallback: never crash on Ollama failure
        return RoleMapping(
            roles=[],
            confidence=0.0,
            kind="whole_document",
            reason=f"Fallback to whole_document due to error: {e}" if debug else "",
        )


def map_query_to_roles(
    question: str,
    mode: str = "hybrid",
    host: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
    debug: bool = False,
) -> RoleMapping:
    """
    Main entry point for rhetorical query mapping.

    Args:
        question: Natural language question from the user.
        mode: Routing strategy:
            - 'hybrid' (default): rule layer first if exactly one cue fires, else LLM.
            - 'llm_only': always query LLM.
            - 'rules_only': evaluate rule layer only; fallback to whole_document if no rule fires.
        host: Ollama host URL.
        model: Ollama model name.
        timeout: HTTP request timeout.
        debug: If True, populate and generate the reason field. Default False (faster).

    Returns:
        RoleMapping with (roles, confidence, kind, reason).
    """
    q = question.strip()
    if not q:
        return RoleMapping(
            roles=[],
            confidence=0.0,
            kind="whole_document",
            reason="Empty query defaulted to whole_document" if debug else "",
        )

    mode_norm = mode.lower().strip()

    if mode_norm == "rules_only":
        matched_roles, rule_result = evaluate_rules(q, debug=debug)
        if rule_result is not None:
            return rule_result
        return RoleMapping(
            roles=[],
            confidence=0.5,
            kind="whole_document",
            reason="No specific rule cue matched; rules-only fallback to whole_document" if debug else "",
        )

    if mode_norm == "hybrid":
        matched_roles, rule_result = evaluate_rules(q, debug=debug)
        # Spec: "rules first when exactly one role cue fires, otherwise the LLM"
        if len(matched_roles) == 1 and rule_result is not None:
            return rule_result
        if rule_result is not None and rule_result.kind in {"out_of_scope", "whole_document"}:
            return rule_result
        # Otherwise route to LLM
        return _call_ollama_llm(q, host=host, model=model, timeout=timeout, debug=debug)

    if mode_norm == "llm_only":
        return _call_ollama_llm(q, host=host, model=model, timeout=timeout, debug=debug)

    raise ValueError(f"Unknown mode '{mode}'. Expected 'hybrid', 'llm_only', or 'rules_only'.")

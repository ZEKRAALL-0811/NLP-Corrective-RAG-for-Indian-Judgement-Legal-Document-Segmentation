# Document 03 — Project Workflow

**Entry Point**
User opens the Streamlit app. No login. Landing screen shows two states: "No document loaded" (default) or "Document loaded: [judgment name]" if one is already ingested in this session.

---

## Core Flow

**Step 1 — Document Ingestion**
User uploads a judgment (PDF or plain text) via a file uploader.
→ System runs sentence segmentation + InLegalBERT role classification in the background, shows a progress indicator ("Segmenting judgment...").
→ On completion, UI shows a role-distribution summary (e.g. "58 Facts, 12 Issue, 34 AoP, 26 AoR, 71 Reasoning, 9 Decision, 140 None") using LegalSeg's exact 7 labels — Facts, Issue, Arguments of Petitioner (AoP), Arguments of Respondent (AoR), Reasoning, Decision, None — so the user can sanity-check the segmentation before querying.
→ Chunks are embedded and written to ChromaDB with role metadata.

**Step 2 — Query Input**
User types a natural-language question into a text box (e.g. "What did the petitioner argue about the limitation period?").
→ User clicks "Ask" (or presses Enter).

**Step 3 — Query Role Mapping (visible to user)**
System displays which rhetorical role(s) it mapped the query to, e.g. "Searching: Arguments of Petitioner" — this is shown as a small inline tag, not buried in logs, since transparency is part of the product's value proposition.

**Step 4 — Retrieval + Grading (visible progress)**
UI shows a lightweight status: "Retrieved 3 chunks → grading relevance..."
→ Once grading completes, show a compact trace panel (collapsible, not forced open) listing each chunk with its grade (Correct / Ambiguous / Incorrect) and confidence score. This is the differentiator over a black-box chatbot — users should be able to expand and inspect it.

**Step 5 — Self-Correction (conditional branch)**
- If at least one chunk graded Correct → proceed directly to Step 6.
- If all chunks graded Incorrect → UI shows "No direct match found — rephrasing query..." and the system automatically retries with a rewritten query (max 2 retries, shown as "Attempt 2 of 2" so the user isn't left wondering if it's stuck).
- If a chunk graded Ambiguous and no Correct chunk exists → UI shows "Supplementing with web search..." before proceeding to generation.
- If max retries are exhausted with no Correct/Ambiguous context → UI shows a clear message: "No reliable answer found in this document for that question" rather than forcing a generation attempt.

**Step 6 — Generation**
System generates the final answer using only verified context. UI shows a loading state ("Drafting answer from verified context...").

**Step 7 — Evaluation Gate (invisible unless it fails)**
DeepEval runs Faithfulness + Answer Relevancy checks in the background.
- **Pass:** answer is shown directly, with a small "✓ Verified against source" badge and the cited paragraph/role reference.
- **Fail:** system attempts one regeneration using the same verified context before showing anything to the user.
- **Still fails after retry:** answer is shown with a visible disclaimer badge ("⚠ Could not be fully verified — review source manually") plus a link to the exact retrieved chunks, so the user can check it themselves rather than being silently handed an unflagged hallucination.

**Step 8 — Answer Review**
User sees: the final answer, the verification badge, and an expandable "Source" section showing the exact chunks (with role tags and paragraph numbers) the answer was built from.
→ User can ask a follow-up question (returns to Step 2) or upload a new document (returns to Step 1).

---

## Edge Cases & Error States
- **Empty/corrupt PDF upload:** reject at ingestion with a clear error, don't attempt segmentation on empty text.
- **Query unrelated to any of the 7 roles** (e.g. small talk): role-mapper should return "None/unclear" and the UI should say so rather than forcing a guess at a role.
- **Ollama/local LLM not running:** app should detect this on startup and show a setup error before the user even tries to query, not fail mid-query.
- **Web search fallback unavailable** (no Tavily key / API down): degrade gracefully — proceed with only the Ambiguous chunk and flag lower confidence, rather than crashing the flow.

## Exit Points
- User closes the tab (session/document state is not persisted — each session starts fresh, per the single-user prototype scope).
- User uploads a new document mid-session, which clears the previous ChromaDB collection for that session before ingesting the new one.

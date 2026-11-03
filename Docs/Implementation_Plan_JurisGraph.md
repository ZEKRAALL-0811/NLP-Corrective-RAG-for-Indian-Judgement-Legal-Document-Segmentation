# Document 05 — Implementation Plan

Build order matters here specifically because later modules (grading, self-correction) are meaningless to test without earlier ones (segmentation, retrieval) already producing correct output. Each module below has a hard completion criterion — don't let the agent move to the next module until the current one passes it.

---

## Module 1 — Data Preparation
**Goal:** Clean, verified, label-mapped LegalSeg data ready for training.
**Build:**
- Load LegalSeg train/val/test CSVs
- Validate: confirm all 7 labels present and spelled exactly (`Facts`, `Issue`, `Arguments of Petitioner`, `Arguments of Respondent`, `Reasoning`, `Decision`, `None`) with no stray whitespace/typos
- Drop blank Text/Label rows
- Map labels to integer IDs (label2id/id2label dict)
- Produce a class-count report (value_counts per split)
**Completion criteria:** A script that loads the data and prints row counts + label distribution matching the paper's stats (~1.12M train sentences, 7 distinct labels). No blanks, no unmapped labels.
**Do not proceed to Module 2 until this passes** — a broken label map silently corrupts every downstream module.

---

## Module 2 — InLegalBERT Fine-Tuning
**Goal:** A trained sentence-classification model that predicts rhetorical role from raw sentence text.
**Build:**
- `AutoModelForSequenceClassification` on `law-ai/InLegalBERT`, 7-class head
- Training loop (the script from earlier in this project) — start on a stratified subsample (a few thousand sentences/class) before committing to a full run
- Track macro F1 + confusion matrix, not just accuracy
**Completion criteria:** Model checkpoint saved locally, macro F1 reported on val set, confusion matrix reviewed (expect Reasoning↔Decision and AoP↔AoR confusion — acceptable; near-zero recall on any class is not).
**Depends on:** Module 1.

---

## Module 3 — Segmentation Pipeline (Inference Wrapper)
**Goal:** A function that takes a raw judgment (PDF/text) and returns role-tagged sentences.
**Build:**
- Sentence splitter (raw PDF/text → list of sentences)
- Batch inference through the Module 2 model
- Output format: `[{"text": ..., "role": ..., "sentence_idx": ...}]`
**Completion criteria:** Feed in one full unseen judgment, get back a role-tagged sentence list; spot-check 10 sentences by eye against their assigned role.
**Depends on:** Module 2.

---

## Module 4 — Vector Store (ChromaDB)
**Goal:** Role-tagged chunks embedded and stored, filterable by role metadata.
**Build:**
- Group consecutive same-role sentences into chunks (not sentence-by-sentence storage — that's noisy)
- Embed with `sentence-transformers`
- Write to ChromaDB with `role` as metadata field
- Test a `where={"role": "Facts"}` filtered query returns only Facts chunks
**Completion criteria:** Ingest one document, run a metadata-filtered query, confirm zero cross-role leakage in results.
**Depends on:** Module 3.

---

## Module 5 — Query Role-Mapper
**Goal:** Given a user question, determine which rhetorical role(s) it needs.
**Build:**
- LLM prompt (few-shot, using the 7 exact role names) that maps a question → role
- Handle the "unclear/None" case explicitly rather than forcing a guess
**Completion criteria:** Test against 15–20 hand-written sample questions covering all 7 roles + 1 unrelated question; role-mapper gets the obvious ones right and correctly flags the unrelated one as unclear.
**Depends on:** Nothing upstream except knowing the label set — can be built in parallel with Modules 2–4, but integration-tested only after Module 4 exists.

---

## Module 6 — Retrieval Node (LangGraph)
**Goal:** LangGraph node that combines Module 5's role + the user query into a filtered ChromaDB search.
**Build:**
- Define LangGraph `TypedDict` state (question, mapped_role, retrieved_chunks, grades, answer, retry_count)
- Retrieval node: role-filtered top-k query against Module 4's store
**Completion criteria:** Given a test query, node returns only role-correct chunks in state.
**Depends on:** Modules 4, 5.

---

## Module 7 — Relevance Grader Node
**Goal:** Grade each retrieved chunk Correct/Ambiguous/Incorrect via local LLM.
**Build:**
- Ollama call (Llama-3.2-3B-Instruct, temperature 0), structured JSON output
- Truncate chunks to 800 chars before grading
- Parse grade + confidence into state
**Completion criteria:** Feed in a known-good chunk and a known-irrelevant chunk for the same query; confirm grades land in the expected buckets consistently across repeated runs (temperature-0 determinism check — same input, same grade, every time).
**Depends on:** Module 6.

---

## Module 8 — Self-Correction Routing
**Goal:** LangGraph conditional edges implementing the Correct/Ambiguous/Incorrect branching.
**Build:**
- Correct → straight to Module 9
- Incorrect (all chunks) → Query Rewriter node (LLM rewrite call) → loop back to Module 6, hard-capped at 2 retries
- Ambiguous (no Correct present) → Tavily fallback search → merge into verified context → Module 9
- Retries exhausted, still nothing usable → explicit "no reliable answer" state, not a forced generation
**Completion criteria:** Force each branch manually (mock a grader response) and confirm the graph routes correctly, and confirm the retry cap actually halts execution at 2 — this is the most important failure mode to test, since an uncapped loop will hang your demo.
**Depends on:** Module 7.

---

## Module 9 — Generation Node
**Goal:** Produce the final answer strictly from verified context.
**Build:**
- Prompt template that only includes Correct (+ any Ambiguous/web-fallback) context, never raw retrieved-but-ungraded chunks
**Completion criteria:** Generated answer references only facts present in the passed context (manual check on 3–5 test queries).
**Depends on:** Module 8.

---

## Module 10 — DeepEval Evaluation Gate
**Goal:** Post-generation Faithfulness + Answer Relevancy check, with re-generation on failure.
**Build:**
- DeepEval `FaithfulnessMetric` + `AnswerRelevancyMetric` wired to run against every generation
- On fail: one re-generation attempt using the same context; on second fail, return answer with disclaimer badge state
**Completion criteria:** Deliberately feed a bad/hallucinated answer through the check and confirm it's flagged (don't just test the happy path — you need to see it catch something).
**Depends on:** Module 9.

---

## Module 11 — Streamlit UI
**Goal:** Wire Modules 1–10 into the interface described in the Workflow and UI/UX Design Brief docs.
**Build:**
- Upload → ingestion progress → role-distribution card (Module 3+4)
- Query box → role tag display (Module 5) → collapsible grading trace (Module 7) → answer + verification badge (Module 9+10)
- Apply the color/typography tokens from the Design Brief
**Completion criteria:** Full click-through of the workflow doc's Core Flow, steps 1–8, on one real judgment, without touching code mid-session.
**Depends on:** All prior modules functional individually.

---

## Module 12 — End-to-End Integration Test
**Goal:** Confirm the whole graph behaves correctly across edge cases, not just the happy path.
**Build:** Run the edge cases listed in the Workflow doc — corrupt upload, unrelated query, Ollama down, Tavily key missing — and confirm each degrades the way the workflow doc specifies rather than crashing.
**Completion criteria:** Every edge case produces the documented UI state, not a stack trace.
**Depends on:** Module 11.

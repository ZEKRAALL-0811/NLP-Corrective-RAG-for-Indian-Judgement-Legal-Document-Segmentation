# Document 01 — PRD

**Project Name**
JurisGraph — Agentic Corrective RAG for Indian Legal Documents

**Tagline**
A self-correcting RAG system that segments Indian court judgments by rhetorical role and grades its own retrievals before answering, so legal Q&A stays grounded instead of hallucinated.

**Problem**
Standard RAG chunks legal documents by arbitrary character limits, mixing Facts, Arguments, and Rulings into the same fragment. When queried, this returns noisy, structurally mismatched context, and the LLM confidently generates legally incorrect or hallucinated answers. Legal researchers can't trust generic RAG tools for case work because there's no guarantee the retrieved text is even the right *kind* of legal content, let alone factually grounded.

**Target User**
Legal researchers, paralegals, and law students working with lengthy Indian Supreme Court / High Court judgments. They need to pull one specific structural element of a case — e.g. "what did the petitioner argue" or "what was the final ruling" — without reading the full document or risking an AI-fabricated answer they can't cite.

**Core Features (Must Have)**
- Rhetorical-role segmentation: InLegalBERT fine-tuned to classify every sentence of a judgment into one of 7 roles (Facts, Issues, Arguments of Petitioner, Arguments of Respondent, Precedent, Statute, Ratio Decidendi)
- Role-tagged vector store: ChromaDB holding each chunk with its predicted role as metadata, queryable with a strict `where` filter
- Query role-mapping: a step that maps the user's question to the rhetorical role(s) it needs, before any vector search runs
- Relevance Grader node (LangGraph + Ollama, temp=0): grades every retrieved chunk as Correct (≥0.7) / Ambiguous (0.3–0.7) / Incorrect (<0.3) against the query
- Self-correction loop: Incorrect chunks trigger automatic query rewriting and a retry (capped retry count); Ambiguous chunks trigger a fallback web/legal search
- Generation node: answers using only chunks that passed grading
- DeepEval quality gate: post-generation check for Faithfulness, Answer Relevancy, Contextual Precision/Recall; failed generations trigger a re-generation attempt before falling back to a disclaimed best-effort answer

**Out of Scope**
- Predicting case outcomes or giving legal advice — this is a research/retrieval aid, not a substitute for counsel
- Non-English judgments or translation
- Scanned/non-OCR PDFs, handwritten notes, or any non-text evidence
- Large (70B+) models — the system is scoped to run on resource-efficient local/small models (InLegalBERT ~110M params, Llama-3.2-3B grader)
- Multi-document / cross-case reasoning in this version — each query operates on one ingested judgment at a time

**User Stories**
- As a legal researcher, I want to restrict my query to "Arguments of the Petitioner" so I don't get the respondent's counter-arguments or the judge's opinion mixed in.
- As a paralegal, I want the system to discard irrelevant results and rewrite my query automatically so I don't have to manually re-prompt when a search comes back empty.
- As a law student, I want every generated answer to pass a faithfulness check so I can cite the underlying paragraph without worrying it's fabricated.
- As a developer building this, I want each LangGraph node (retrieval, grading, rewriting, generation, evaluation) built and testable as an isolated module so I can build it incrementally in Antigravity rather than as one monolith.

**Success Metrics**
- InLegalBERT segmentation: baseline ≥ 0.50 macro F1 on the LegalSeg test set (matching the paper’s sentence-level InLegalBERT, 0.49), stretch ≥ 0.60 with sequence smoothing
- Context Precision: >85% of chunks reaching the generation node are graded "Correct" by the relevance grader
- Faithfulness: DeepEval faithfulness score >95% across test queries
- Self-correction recovery rate: >75% of initially-failed retrievals (all chunks Incorrect) succeed after query rewriting

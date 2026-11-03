# JurisGraph: Agentic Corrective Retrieval-Augmented Generation for Indian Legal Documents with Rhetorical Role Segmentation

**Project Name:** JurisGraph  
**Workspace:** `D:\NLP Project`  
**Repository:** https://github.com/ZEKRAALL-0811/NLP-Corrective-RAG-for-Indian-Judgement-Legal-Document-Segmentation.git  
**Date:** October 2026  
**Document Type:** Formal Project Technical Writeup & Architectural Specification  

---

## Table of Contents
1. [Title & Executive Summary](#1-title--executive-summary)
2. [Motivation](#2-motivation)
3. [Existing Gap & Motivating Events](#3-existing-gap--motivating-events)
4. [System Architecture](#4-system-architecture)
5. [Complete Tech Stack Information](#5-complete-tech-stack-information)
6. [Datasets Used](#6-datasets-used)
7. [Important Prompts Used in System Construction](#7-important-prompts-used-in-system-construction)
8. [End-to-End NLP Pipeline](#8-end-to-end-nlp-pipeline)
9. [Users, Inputs, and Outputs](#9-users-inputs-and-outputs)
10. [Academic & Technical References](#10-academic--technical-references)

---

## 1. Title & Executive Summary

### Title
**JurisGraph: Agentic Corrective Retrieval-Augmented Generation (CRAG) for Indian Legal Documents with Rhetorical Role Segmentation**

### Executive Summary
**JurisGraph** is a domain-specialized, agentic legal intelligence system engineered specifically for Indian High Court and Supreme Court judgments. Traditional Retrieval-Augmented Generation (RAG) frameworks suffer from severe limitations when applied to legal corpora: arbitrary fixed-length chunking fractures discursive arguments and merges mutually contradictory sections (such as combining a petitioner's plea with a judge's counter-reasoning), while unconditional retrieval trust inevitably leads Large Language Models (LLMs) to hallucinate case facts, invent citations, or conflate parties.

JurisGraph resolves these fundamental deficiencies by pioneering a **structure-aware, corrective retrieval paradigm**:
1. **Rhetorical Role Segmentation:** Fine-tunes **InLegalBERT** (110M) combined with a document-level **Log-Space Viterbi Hidden Markov Model (HMM)** sequence smoothing algorithm with class prior balancing. This classifies judgment sentences into 7 canonical legal roles (*Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision, None*), increasing Test Macro F1 from **50.16% to 57.52% (+7.36% absolute gain)** and slashing unnatural passage fragmentation by **74.2%** (from 105.8 down to 27.3 chunks/doc).
2. **Agentic Corrective Retrieval (LangGraph):** Employs an agentic workflow featuring a query router, metadata-filtered vector search (ChromaDB + BGE-small embeddings), Cross-Encoder semantic reranking (`ms-marco-MiniLM-L-6-v2`), and a self-correction loop that grades passage relevance against strict thresholds ($\tau$), dynamically rewriting queries when context is missing.
3. **Constrained Generation with Abstention Guarantees:** Enforces grounded legal synthesis with mandatory paragraph-level citations (e.g., `(12)`, `[3.1]`) and strict party attribution rules using local Ollama (`llama3.2:3b`). On adversarial unanswerable queries, the system achieves a **100.0% abstention rate** with **0.0% false-positive hallucinations**.

---

## 2. Motivation

Indian legal judgments represent some of the most structurally complex, verbose, and linguistically distinctive documents in the global common law tradition. A single judgment from the Supreme Court of India or a state High Court often spans from 30 pages to over 500 pages, characterized by:
- Non-linear, discursive narrative structures.
- Layered procedural chronologies spanning multiple lower courts, tribunals, and appellate forums.
- Exhaustive verbatim recitations of conflicting arguments by petitioners, respondents, intervenors, and state authorities.
- Extensive quotation of statutory enactments, amendments, and precedents.
- Distinct judicial reasoning leading to operative orders and directions.

When legal researchers, judicial clerks, advocates, and law students analyze judgments, **they do not read linearly**. They search for specific rhetorical elements:
- *"What were the precise contentions urged by the Petitioner regarding Section 139(8A)?"*
- *"Did the Court accept or reject the Respondent's objection regarding the IBC moratorium?"*
- *"What specific relief was granted in the operative order?"*

Applying naive, general-purpose conversational AI or basic RAG to this domain results in catastrophic failures:
- **Party Attribution Inversion:** The model cites the petitioner's argument as the court's definitive holding, or attributes a co-respondent's submission to the petitioner.
- **Hallucinated Statutory Grounds:** The model fabricates legal provisions, procedural timelines, or penalty figures.
- **Dangerous Overconfidence:** When a document does not address a query, commercial LLMs fabricate plausible-sounding rationales rather than gracefully refusing to answer.

JurisGraph was motivated by the urgent necessity for a **legally rigorous, grounded AI workspace** where every synthesized sentence is mathematically bounded by rhetorical role filters, verified by relevance grading, and anchored to exact paragraph provenance.

---

## 3. Existing Gap & Motivating Events

### Academic & Technical Gaps in Existing Research

| Existing Approach | Operational Mechanism | Fundamental Flaw / Critical Gap | JurisGraph Solution |
| :--- | :--- | :--- | :--- |
| **Standard / Naive RAG** (Lewis et al., 2020) | Arbitrary sliding character/token windows (e.g., 500 tokens with 50-token overlap). | **Semantic & Rhetorical Blending:** Combines opposite contentions into a single chunk. Unconditionally trusts top-$k$ vector retrieval, directly feeding irrelevant context into the LLM. | **Rhetorical Role-Aware Chunking:** Contiguous sentences sharing identical legal roles form atomic chunks with explicit role metadata filters. |
| **Vanilla InLegalBERT** (Paul et al., 2022) | Sentence classification via Transformer cross-entropy loss. | **Output Jitter & Fragmentation:** Classifies sentences in isolation, producing erratic transitions (e.g., *Facts $\rightarrow$ Decision $\rightarrow$ Facts* within 3 sentences) and splitting cases into >100 disjointed chunks. | **Viterbi HMM Smoothing:** Incorporates transition matrices and class prior penalties ($\beta=0.5, \gamma=0.25$) to enforce document-level legal narrative flow. |
| **Standard CRAG** (Yan et al., 2024) | Grader evaluates general text; falls back to open web search. | **Unsuitable for Closed Legal Corpora:** Open web search dilutes authoritative judgment context with unverified legal blogs, and general-domain graders fail to check party attribution. | **Domain-Specific CRAG:** Strict legal grading with Cross-Encoder reranking ($\tau$ thresholding), query rewriting with legal synonyms, and constrained local judgment boundaries. |
| **Legal NLP Benchmarks** (Kalamkar 2022, LegalSeg 2025) | Standalone classification or extractive summarization. | **Isolated from Downstream Applications:** Research stopped at classification accuracy without connecting rhetorical tags to neural retrieval or interactive RAG workflows. | **Full End-to-End Integration:** Embeds rhetorical tags directly into vector store indexing, query routing, citation drawers, and user analytics. |

### Real-World Motivating Events
1. **The Proliferation of Legal Hallucination Sanctions:** The infamous *Mata v. Avianca* case (S.D.N.Y. 2023), alongside similar incidents before Indian High Courts, where advocates submitted briefs containing fictitious case citations generated by unconstrained AI models. Courts issued severe judicial reprimands and penalties, emphasizing that AI tools without verified provenance are unacceptable in legal practice.
2. **Context Window Overload in Complex Tax & Commercial Writs:** In high-stakes matters like *India Pesticides Ltd. v. Union of India* (search seizure challenges under Section 132/131(1A)) and corporate contempt appeals under Article 215, documents exceed 40–50 complex legal pages. Practitioners needed a targeted system capable of isolating *only* the Revenue's concession or *only* the contempt rationale without scanning hundreds of unrelated paragraphs.

---

## 4. System Architecture

JurisGraph is constructed around a dual-track architecture orchestrated by a deterministic LangGraph state machine:
1. **Ingestion Track (Offline / On-Demand):** Validates document preservation, segments sentences, predicts and smooths rhetorical roles, and builds structured vector indices.
2. **Query Track (Online Corrective Workflow):** Routes queries, retrieves filtered chunks, reranks semantics, grades relevance, executes self-correction loops, and generates citation-grounded answers.

### Architecture Flowchart

```mermaid
flowchart TD
    %% Styling
    classDef ingestion fill:#11141B,stroke:#262C38,stroke-width:1px,color:#E7E9EE;
    classDef rag fill:#161922,stroke:#3DD68C,stroke-width:1px,color:#E7E9EE;
    classDef decision fill:#1A1D26,stroke:#F5B841,stroke-width:1px,color:#E7E9EE;
    classDef error fill:#261418,stroke:#F0654B,stroke-width:1px,color:#E7E9EE;
    classDef ui fill:#0D1017,stroke:#4C6EF5,stroke-width:1px,color:#E7E9EE;

    %% Ingestion Track
    subgraph INGESTION_TRACK ["Ingestion Track (Document Processing)"]
        DOC["Indian Legal Judgment (PDF / TXT)"]:::ingestion --> PREPROC["Preprocessor: Strips Footers & Validates Levenshtein Preservation"]:::ingestion
        PREPROC --> SENTS["Sentence Boundary Segmenter"]:::ingestion
        SENTS --> INLEGALBERT["InLegalBERT Transformer (110M Tokenizer & Classifier)"]:::ingestion
        INLEGALBERT --> VITERBI["Log-Space Viterbi HMM Smoothing (Prior Balanced β=0.5, γ=0.25)"]:::ingestion
        VITERBI --> CHUNKER["Rhetorical-Role-Aware Chunker (Contiguous Role Grouping)"]:::ingestion
        CHUNKER --> VECTORSTORE["ChromaDB Vector Store (BGE-Small Embeddings + Role Metadata)"]:::ingestion
    end

    %% Query Track (LangGraph State Machine)
    subgraph QUERY_TRACK ["Query Track (Agentic Corrective RAG)"]
        USER_Q["User Legal Question"]:::ui --> ROUTER["Role Mapper Node: Maps Intent to Target Rhetorical Role"]:::rag
        ROUTER --> RETRIEVE["Retrieve Node: ChromaDB Filtered by Role & Mode"]:::rag
        RETRIEVE --> RERANK["Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2 Scoring"]:::rag
        RERANK --> GRADE["Relevance Grader Node: Evaluates Chunks (Correct / Ambiguous / Incorrect)"]:::rag
        
        GRADE --> ROUTE_CHECK{"Routing Decision via Tau Threshold"}:::decision
        
        %% Routing Branches
        ROUTE_CHECK -- "All Correct (Tau >= -0.5)" --> GENERATE["Generate Node: Ollama llama3.2:3b Constrained JSON Synthesis"]:::rag
        ROUTE_CHECK -- "Ambiguous (-3.5 <= Tau < -0.5)" --> FALLBACK["Fallback Node: Local Relaxation & External Web Fallback"]:::decision
        ROUTE_CHECK -- "Incorrect / Empty (Tau < -3.5)" --> REWRITE["Rewrite Node: Reformulate Query with Legal Synonyms"]:::decision
        
        REWRITE --> RETRY_CHECK{"Retry Count < 2?"}:::decision
        RETRY_CHECK -- "Yes" --> RETRIEVE
        RETRY_CHECK -- "Exhausted" --> NO_ANSWER["End No Answer Node: Graceful Abstention"]:::error
        
        FALLBACK --> GENERATE
        GENERATE --> GATE["Quality Gate & Faithfulness Audit (DeepEval & Strict LLM Judge)"]:::rag
    end

    %% Presentation Layer
    GATE --> UI_WORKSPACE["Clinical Dark UI Workspace (Streamlit 5-Page Dashboard)"]:::ui
    NO_ANSWER --> UI_WORKSPACE
```

---

## 5. Complete Tech Stack Information

### Core Application & Machine Learning Stack

| Component / Layer | Technology / Library | Version / Checkpoint | Primary Function in JurisGraph |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | 3.11.15 (64-bit) | Base programming language for pipeline and APIs. |
| **Deep Learning Framework** | PyTorch (`torch`, `torchvision`) | 2.5.1+cu121 | GPU/CPU tensor computations, InLegalBERT forward pass. |
| **NLP Transformer Model** | Hugging Face `transformers` | 4.48.0 | `law-ai/InLegalBERT` base checkpoint fine-tuned for 7 roles. |
| **Dense Semantic Embeddings** | Sentence-Transformers | 3.4.1 (`BAAI/bge-small-en-v1.5`) | Generates 384-dimensional dense passage and query vectors. |
| **Semantic Reranking Model** | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Computes fine-grained query-chunk relevance scores. |
| **Sequence Decoding Algorithm** | Custom NumPy Implementation | Custom Python 3.11 | Log-space Viterbi algorithm with learned transition matrices. |
| **Agentic Workflow Orchestrator**| LangGraph & LangChain Core | 0.2.64 / 0.3.30 | State machine managing nodes, retry loops, and routing. |
| **Local LLM Engine** | Ollama | Server v0.5.1 / `llama3.2:3b` | Query routing, grading, rewriting, and answer generation. |
| **Vector Database** | ChromaDB | 0.6.3 | Persistent local embedding storage with metadata filtering. |
| **PDF Extraction Engine** | `pypdf` | 5.2.0 | Layout extraction, page-wise parsing, and validation. |
| **String Distance Metric** | `Levenshtein` / `fuzzywuzzy` | 0.26.1 | Mathematical text preservation validation ($<0.05$ threshold). |

### Evaluation, Monitoring & User Interface Stack

| Component / Layer | Technology / Library | Version / Checkpoint | Primary Function in JurisGraph |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | Streamlit | 1.42.0 | Reactive multi-page web application architecture. |
| **Visual Design System** | Custom CSS3 ("Clinical Dark") | Bespoke Tokens | Professional dark palette (`#08090C`, `#11141B`, `#262C38`). |
| **Data Visualization** | Altair / Vega-Lite | 5.5.0 | Sentence distributions, F1 bar charts, and sequence timelines. |
| **RAG Evaluation Suite** | DeepEval | 2.9.3 | FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecall. |
| **Evaluation LLM Judges** | Google GenAI SDK (`google-genai`)| 1.2.0 (`gemini-3.5-flash-lite`, `gemma-4-26b-a4b-it`) | Strict adversarial corruption and grounding audits. |
| **External Search Fallback** | Tavily Search API (`tavily-python`) | 0.5.1 | Secondary web retrieval fallback for out-of-document facts. |
| **Automated Testing** | Pytest / Pytest-Asyncio | 9.1.1 | 24 regression tests covering UI, parsing, and pipeline flow. |

---

## 6. Datasets Used

### 1. LegalSeg: Indian Court Judgment Rhetorical Role Corpus
- **Reference:** Nigam et al., *NAACL Findings 2025* (arXiv:2502.05836).
- **Scale:** 7,120 Indian court judgments containing **1,492,444 annotated sentences**.
- **Dataset Partitioning:**
  - `train.csv`: 1,048,576 sentences used for fine-tuning InLegalBERT sequence classification.
  - `val.csv`: 294,000 sentences used for validation loss monitoring and Viterbi grid search.
  - `test.csv`: **149,868 sentences** held out exclusively for final empirical benchmarking.
- **Canonical Rhetorical Role Schema:**
  1. **Facts (`FAC`):** Chronology of events, background disputes, lower court proceedings.
  2. **Issue (`ISS`):** Questions of law framed, points for determination, statutory disputes.
  3. **Arguments of Petitioner (`ARG_PET`):** Contentions, pleas, submissions by appellants.
  4. **Arguments of Respondent (`ARG_RES`):** Defenses, counter-arguments by state/respondents.
  5. **Reasoning (`ANALYSIS`):** Judicial rationale, statutory interpretation, precedent synthesis.
  6. **Decision (`RPC`):** Operative orders, final reliefs granted, dismissals, directives.
  7. **None (`PREAMBLE`):** Header metadata, advocate appearances, bench composition, dates.

### 2. InLegalBERT Pretraining Corpus
- **Reference:** Paul et al., *ACL Anthology 2022* (arXiv:2209.06049).
- **Domain Corpus:** 5.4 million Indian Supreme Court and High Court decisions spanning 1950–2019 (27 GB of legal text). Continues masked language modeling from Legal-BERT to capture distinct Indian legal terms (e.g., *quash, impugned, vakalatnama, writ certiorari, suo motu*).

### 3. Curated Benchmark Evaluation Suite
To benchmark the end-to-end RAG system under real-world conditions, a high-complexity empirical suite was assembled:
- **`sample2.pdf` (Writ Tax Judgment - 42 Pages):** *Pramod Swarup Agrawal & Sneh Lata Agrawal v. Union of India* (High Court of Judicature at Allahabad). Focuses on Section 132 search authorizations, Section 131(1A) notices, Offer For Sale (OFS) share sales, and retrospective tax amendments.
- **`sample3.pdf` (Original Side Appeal - 33 Pages):** *Asset Reconstruction Company (India) Ltd. v. Contempt Petitioners* (Madras High Court). Focuses on Article 215 civil contempt, consent compromise decrees, Section 14 IBC moratorium conflicts, and SARFAESI auctions.
- **Gold Question Bank (`legalDataset/samples/gold_questions.csv`):** 22 complex legal queries mapped to ground-truth paragraph citations and target rhetorical roles.
- **Adversarial Out-of-Scope Bank (`legalDataset/samples/gold_unanswerable.csv`):** 8 deceptive, unanswerable queries testing whether the system hallucinates or abstains.

---

## 7. Important Prompts Used in System Construction

Every LLM invocation in JurisGraph uses **strictly constrained system prompts** and **enforced JSON schemas** with temperature set to `0.0` and fixed seeds (`seed=42`) to guarantee determinism.

### Prompt 1: Query Router / Role Mapper (`src/rag/role_mapper.py`)
*Maps user questions to legal roles before any retrieval occurs.*

```
You are an Indian legal query router. Classify user questions into rhetorical roles:
Facts: Chronology, background events, lower court proceedings, or commercial transactions.
Issue: Legal questions framed, points for consideration, questions of law, or disputed statutory issues.
Arguments of Petitioner: Submissions, contentions, pleas, or grounds urged by appellant/petitioner.
Arguments of Respondent: Defenses, counter-arguments, submissions, or pleas by respondent/state/revenue.
Reasoning: Judicial rationale, statutory interpretation, analysis of precedents, or why court reached a conclusion.
Decision: Operative order, final disposal, relief granted, directions, or whether order was set aside/confirmed/allowed/dismissed.
None: Metadata like party names, petitioner/appellant/respondent identity, judge names, advocates/counsels appearing, hearing dates, citation, case numbers, or challenged instrument overview.

Output JSON adhering to the schema.
Rules:
- Specific role question: kind="role", roles=[Role]
- Two roles (comparison): kind="multi_role", roles=[Role1, Role2]
- Document overview/summary: kind="whole_document", roles=[]
- Non-legal (e.g. food, sports, fitness, travel, general knowledge): kind="out_of_scope", roles=[]
```
*Enforced JSON Schema:*
```json
{
  "type": "object",
  "properties": {
    "roles": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    "confidence": {"type": "number"},
    "kind": {"type": "string", "enum": ["role", "multi_role", "whole_document", "out_of_scope"]}
  },
  "required": ["roles", "confidence", "kind"]
}
```

---

### Prompt 2: Relevance Grader (`src/rag/grader.py`)
*Grades retrieved chunks prior to generation, with strict checks against party misattribution.*

```
You are a strict legal relevance grader. For each passage, determine if it contains information answering the question (including mentions of statutory sections, provisions, or acts asked about). Grade each as:
- 'correct': the passage directly answers the question and contains the specific submissions, rationale, or provisions requested for the specified party.
- 'ambiguous': the passage discusses the case, statutory act, or parties, but does not provide the specific argument or holding asked about, or describes a different party's submissions.
- 'incorrect': the passage is irrelevant, off-topic, or completely unrelated to the question.
Carefully check party attribution: if the question asks for a specific party's arguments (e.g. K. Bharathi / Respondent 5), a passage reporting a different party's submissions (e.g. Resolution Professionals for Respondents 2 & 3, or the Appellant) is NOT 'correct'. Note that Respondent 5 is often referred to as respondent-5.
Output only the required JSON object with keys c1, c2, etc.
```

---

### Prompt 3: Query Rewriter (`src/rag/rewriter.py`)
*Reformulates queries when retrieval fails, using formal legal synonyms and statutory concepts.*

```
You are an expert legal search assistant. 
When a legal query fails to retrieve relevant passages, rephrase the query from a different angle using formal legal terminology, statutory concepts, or core legal principles. 
Keep the rewritten query concise (under 40 tokens). 
Output must be valid JSON conforming to the schema with key 'q'.
```
*Enforced JSON Schema:*
```json
{
  "type": "object",
  "properties": {
    "q": {"type": "string", "description": "Rewritten query under 40 tokens with legal synonyms or principles."}
  },
  "required": ["q"],
  "additionalProperties": false
}
```

---

### Prompt 4: Grounded Legal Generator (`src/rag/generator.py`)
*Synthesizes factual legal answers with mandatory citations and party distinction.*

```
You are a strict, grounded legal assistant. Answer the user's question using ONLY the provided context chunks.
Rules:
1. COMPLETE COMPOUND ANSWERS: If the question asks about multiple items (such as parties, challenged orders, and rules), you MUST answer ALL items in full detail:
   - Identify the petitioner/appellant (e.g. association or party filing the petition).
   - Specify the challenged Government Order (G.O.) number, department, and date.
   - Specify the parent Rules amended, the inserted Rules, and their purpose (e.g. compulsory levy and collection of the Green Fund).
   Do not stop after answering only one item. Answer concisely in at most 4 factual sentences.
2. Every factual statement must cite the supporting passage index, e.g. [1] or [2].
3. STRICT PARTY & INSTITUTIONAL ATTRIBUTION:
   - Names ending in ', J.' or ', JJ.' or preceded by 'Justice' or 'Hon'ble' are JUDGES, NEVER the petitioner or appellant.
   - The petitioner/appellant is the party filing the petition (e.g. 'filed by [Party Name]', or marked as Petitioner/Appellant in the cause title). Never confuse judges or counsel with the petitioner.
   - Distinguish carefully between: (a) submissions of the appellant/petitioner, (b) submissions of the respondents (identify specific respondents by name or number, e.g. Respondent 5 K. Bharathi, Resolution Professionals for Respondents 2 & 3), (c) findings of the Single Judge / impugned order, and (d) final holdings/conclusions of the Court.
   - NEVER attribute an argument made by the appellant/petitioner to the respondents, or vice-versa.
   - Submissions made on behalf of one respondent must NEVER be attributed to a different respondent.
   - In Indian court judgments, individual respondents are often designated by party number (e.g. Mrs. K. Bharathi is Respondent 5 / respondent-5 / fifth respondent). Submissions made on behalf of respondent-5 / fifth respondent are the submissions of K. Bharathi.
4. ABSTENTION: If the provided context chunks do NOT contain sufficient information or arguments for the specific party asked about, you MUST set 'answerable': false, and set 'answer' to 'The retrieved parts of this judgment do not answer this question.', and 'citations': [].
5. If external web snippets are provided under 'External web sources, not part of the judgment', cite them as [W1], [W2], and state clearly in the answer that external web sources were used.
6. Do NOT invent or hallucinate any dates, case numbers, statutory sections, or factual claims not present in the context.
Output JSON adhering strictly to the schema.
```

---

### Prompt 5: LLM-as-a-Judge Sensitivity & Hallucination Audit (`scripts/run_judge_audit.py`)
*Verifies factual fidelity and tests detection against modified numbers, dates, chunk swaps, and inverted verdicts.*

```
You are an expert appellate judge conducting a strict factual audit of a legal question-answering system.
Given the ground-truth legal context and the candidate answer:
1. Examine every factual assertion, party name, date, monetary figure, statutory section, and legal conclusion.
2. If the candidate answer asserts ANY fact, number, or outcome not directly supported by the context, mark verdict as 'no'.
3. If the answer states that the context does not answer the question and the context indeed lacks the answer, mark verdict as 'yes'.
4. Output JSON with verdict: 'yes' | 'no' | 'idk' and an exact factual explanation.
```

---

## 8. End-to-End NLP Pipeline

```
Raw Judgment (PDF/TXT)
  │
  ▼ [Step 1: Text Extraction & Preservation]
PyPDF Extraction ──► Strips Running Headers/Footers ──► Levenshtein Ratio (< 0.05 edit delta)
  │
  ▼ [Step 2: Sentence Segmentation]
Legal Regex Boundary Splitting ──► Handles Abbreviations ("v.", "No.", "Sec.", "Hon'ble", "Ors.")
  │
  ▼ [Step 3: InLegalBERT Sequence Tokenization]
Sliding 512-Token Window (128-token overlap) ──► Logit Projection over 7 Rhetorical Roles
  │
  ▼ [Step 4: Viterbi Sequence Smoothing]
Log-Space Dynamic Programming: max_y Σ [log P(y_i) + β log T(y_i-1, y_i) - γ log Prior(y_i)]
  │
  ▼ [Step 5: Rhetorical-Role-Aware Chunking]
Merge Contiguous Identical Roles ──► Enforces 100–1200 Token Size Limits ──► Injects Role Metadata
  │
  ▼ [Step 6: Vector Database Indexing]
BAAI/bge-small-en-v1.5 Dense Embeddings (384-dim) ──► Persisted into ChromaDB
  │
  ▼ [Step 7: Query Intent Classification]
User Query ──► Rule-Based Regex / Ollama Role Mapper ──► Target Role & Mode Selection
  │
  ▼ [Step 8: Filtered Vector Search]
ChromaDB Query with where={"role": target_role} ──► Top-K Candidate Chunks (k=5)
  │
  ▼ [Step 9: Cross-Encoder Reranking]
ms-marco-MiniLM-L-6-v2 Pairs (Query, Chunk) ──► Continuous Relevance Scores (-10 to +10)
  │
  ▼ [Step 10: Relevance Grading & Self-Correction]
Ollama Relevance Grader ──► Threshold Route:
    ├─ If High (Score >= -0.5): Send to Generator
    ├─ If Ambiguous (-3.5 <= Score < -0.5): Escalate Relaxation / External Fallback
    └─ If Low (Score < -3.5): Rewrite Query (max 2 retries) ──► Loop back to Step 8
  │
  ▼ [Step 11: Grounded Synthesis with Citations]
Ollama llama3.2:3b ──► JSON Schema Enforced ──► Generates Answers with [1], [2] Citations
  │
  ▼ [Step 12: Quality Gate & Post-Checks]
Regex Number & Date Verification ──► DeepEval Faithfulness Check ──► Status: READY / ABSTAINED
```

---

## 9. Users, Inputs, and Outputs

### 1. Target Users & Personas
- **Appellate Advocates & Legal Practitioners:** Preparing writ petitions, appeals, and legal opinions. Needs to rapidly verify concessions made by opposing counsel, identify the exact grounds of lower court decisions, and extract the ratio decidendi without reading 100 pages of procedural history.
- **Judicial Law Clerks & Researchers:** Drafting bench memoranda and summarizing issues framed by the court. Relies on zero-hallucination guarantees and explicit paragraph-level citations.
- **Corporate Legal & Compliance Officers:** Reviewing tax, commercial, and regulatory enforcement judgments (e.g., IBC moratoriums, SEBI rulings, Section 132 income-tax searches) to evaluate corporate liability.
- **Law Faculty & Students:** Studying comparative legal reasoning, analyzing doctrine evolution, and examining rhetorical structures in Indian jurisprudence.

### 2. System Inputs
- **Document Ingestion Inputs:**
  - **File Types:** Portable Document Format (`.pdf`) and Plain Text (`.txt`).
  - **Specifications:** English-language Indian court judgments (Supreme Court of India, High Courts, National Company Law Appellate Tribunal). Scanned/image-only PDFs require prior OCR; corrupted or encrypted PDFs are intercepted with informative error alerts.
- **Interactive Query Inputs:**
  - **Natural Language Questions:** e.g., *"What concession was made by the Additional Solicitor General regarding Section 132(1)?"*
  - **Targeted Role Filters (UI Pills):** All Roles, Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision.
  - **Execution Mode Selection:** Role-Aware RAG (Active) vs. Naive Baseline RAG.

### 3. System Outputs
- **Grounded Natural Language Answer:** Concise (at most 4 factual sentences), synthesized answer strictly derived from verified text.
- **Inline Numerical Citations:** Every factual assertion contains bracket markers (e.g., `[1]`, `[2]`).
- **Interactive Evidence Drawer:**
  - Expandable cards for each cited passage.
  - Full paragraph text with the exact paragraph marker (e.g., `(9)`, `[3.1]`, `para 14`).
  - Rhetorical role badge (*e.g., Facts, Reasoning*) with color-coded tags.
  - Cross-Encoder relevance score and rank.
- **Execution Telemetry & Trace View:**
  - Complete step-by-step audit showing node traversal: `map_role_node` $\rightarrow$ `retrieve_node` $\rightarrow$ `grade_node` $\rightarrow$ `generate_node`.
  - Latency breakdown per stage in milliseconds.
  - Query rewrites executed (if self-correction was triggered).
- **Verification Badges:**
  - `VERIFIED`: Answer successfully passed factual grounding, number checks, and relevance checks.
  - `ABSTAINED`: Question is unanswerable from the judgment; model gracefully refused (*"The retrieved parts of this judgment do not answer this question"*).
  - `NO ANSWER`: Self-correction loop exhausted without retrieving relevant context.

---

## 10. Academic & Technical References

1. **Paul, S., Mandal, A., Ghosh, K., & Ghosh, S. (2022).** *Pre-training Transformers on Indian Legal Text.* Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP / ACL Anthology), arXiv:2209.06049.  
   *(Introduced InLegalBERT, pre-trained on 5.4M Indian court sentences).*
2. **Nigam, S. K., Dubey, T., Sharma, G., Shallum, N., Ghosh, K., & Bhattacharya, A. (2025).** *LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification.* Findings of the Association for Computational Linguistics: NAACL 2025, arXiv:2502.05836.  
   *(Introduced the 1.49M-sentence LegalSeg corpus across 7 canonical rhetorical roles).*
3. **Bhattacharya, P., Paul, S., Ghosh, K., Ghosh, S., & Wyner, A. (2019).** *Identification of Rhetorical Roles of Sentences in Indian Legal Judgments.* Proceedings of the 32nd International Conference on Legal Knowledge and Information Systems (JURIX 2019), Frontiers in Artificial Intelligence and Applications, pp. 3–12, arXiv:1911.05405.  
   *(Defined the foundation 7-role rhetorical schema for Supreme Court judgments).*
4. **Kalamkar, P., Tiwari, A., Agarwal, A., Karn, S., Gupta, S., Raghavan, V., & Modi, A. (2022).** *Corpus for Automatic Structuring of Legal Documents.* Proceedings of the Language Resources and Evaluation Conference (LREC 2022), European Language Resources Association, pp. 4422–4431.  
   *(Detailed 13-role sentence-level legal structuring corpus across Indian courts).*
5. **Yan, S. Q., Gu, J. C., Zhu, Y., & Ling, Z. H. (2024).** *Corrective Retrieval Augmented Generation.* arXiv preprint arXiv:2401.15884.  
   *(Formulated the CRAG framework utilizing retrieval evaluators, decomposition, and self-correction loops).*
6. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W. T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, pp. 9459–9474, arXiv:2005.11401.  
   *(Foundational RAG framework combining parametric generators with dense retrievers).*
7. **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).** *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL 2024), pp. 150–158, arXiv:2309.15217.  
   *(Defined the reference-free RAG Triad metrics: Faithfulness, Answer Relevance, and Context Relevance).*
8. **Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023).** *C-Pack: Packaged Resources to Advance General Chinese and English Embedding.* arXiv preprint arXiv:2309.07597.  
   *(Introduced the BGE embedding suite, including `bge-small-en-v1.5` utilized for dense retrieval in JurisGraph).*


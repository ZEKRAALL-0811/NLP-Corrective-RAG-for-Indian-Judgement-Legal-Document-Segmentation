# JurisGraph — Project Structure & File Directory

Comprehensive index and technical description of every file and folder in the **JurisGraph** repository (Agentic Corrective RAG for Indian Legal Documents with Rhetorical Role Segmentation).

---

## 📁 Repository Overview

```
D:\NLP Project\
├── .agents/                    # Custom agent skills and workflow definitions
├── .claude/                    # Claude agent settings and memory
├── .deepeval/                  # DeepEval benchmark cache and test configs
├── .env                        # Local environment variables & API keys
├── .gitignore                  # Git untracked pattern definitions
├── .pytest_cache/              # Pytest test execution cache
├── .venv/                      # Python 3.11 virtual environment
├── CRAG-NLP Review1.pptx       # Presentation deck for Review 1
├── Docs/                       # Engineering specifications and architecture diagrams
├── IMPLEMENTATION_PLAN.md      # Master project implementation roadmap
├── NLP Project details.docx    # Comprehensive university project specification
├── Things Learnt during NLP Project.docx  # Engineering logbook and technical learnings
├── app.py                      # Main Streamlit web application entry point
├── literature_survey.md        # Academic literature survey & references
├── requirements.txt            # Python dependencies
├── artifacts/                  # Checkpoints, Viterbi matrices, and evaluation reports
├── data/                       # Local persistent vector store data
├── legalDataset/               # Raw court judgments, dataset splits, and gold QA
├── notebooks/                  # Colab training & evaluation Jupyter notebooks
├── scratch/                    # Diagnostic experiments, scratch scripts, & validation tools
├── scripts/                    # Production benchmark runners and report generators
├── src/                        # Core application source code
└── tests/                      # Automated unit and integration test suite
```

---

## 1. 📄 Root Directory Files

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| `app.py` | Python Script | Main Streamlit application entry point. Houses page configuration, global session state initialization, custom CSS injection, persistent sidebar with document loading status, and multi-page routing. |
| `IMPLEMENTATION_PLAN.md` | Markdown | Master architectural blueprint detailing the 10 core modules, mathematical formulations, target metrics, task checklists, and verification gates. |
| `PROJECT_STRUCTURE.md` | Markdown | Comprehensive index and technical breakdown of all files and folders in the JurisGraph repository. |
| `literature_survey.md` | Markdown | Academic survey analyzing legal NLP literature, Rhetorical Role Labeling (RRL), InLegalBERT, and Corrective RAG (CRAG) paradigms. |
| `requirements.txt` | Config | Manifest of Python package dependencies including PyTorch, Transformers, LangGraph, ChromaDB, Streamlit, Altair, and evaluation libraries. |
| `.env` | Config | Environment variables storing local configurations, Ollama API URLs, Hugging Face credentials, and optional search API keys. |
| `.gitignore` | Config | Excludes virtual environments, vector caches, model checkpoints, temporary media, and OS-specific files from version control. |
| `CRAG-NLP Review1.pptx` | PowerPoint | Slide presentation detailing project motivation, pipeline architecture, rhetorical role taxonomy, and preliminary evaluation for academic review. |
| `NLP Project details.docx` | Document | Detailed course specification and formal guidelines outlining project scope, grading rubrics, and deliverables. |
| `Things Learnt during NLP Project.docx` | Document | Developer log capturing technical insights, debugging lessons, Viterbi smoothing nuances, and RAG optimization strategies. |

---

## 2. 📁 `Docs/` — Engineering Specifications & Diagrams

Contains product requirement documents, technical architectures, and workflow diagrams.

| File | Type | Description |
| :--- | :--- | :--- |
| `Implementation_Plan_JurisGraph.md` | Markdown | Step-by-step implementation guide covering data pipelines, InLegalBERT fine-tuning, and LangGraph state machine assembly. |
| `PRD_JurisGraph.md` | Markdown | Product Requirements Document outlining functional objectives, target personas (legal researchers, advocates), and system capabilities. |
| `TRD_JurisGraph.md` | Markdown | Technical Requirements Document specifying latency budgets, hardware requirements, model constraints, and component interfaces. |
| `UIUX_Brief_JurisGraph.md` | Markdown | Design brief outlining the "Clinical Dark" design system, color palette, typography hierarchy, and rhetorical role color tags. |
| `Workflow_JurisGraph.md` | Markdown | Detailed operational workflows tracing legal document ingestion, validation, query routing, and iterative answer correction. |
| `NLP Architecture.png` | Image | High-resolution diagram illustrating the modular components, data flow, and model interactions across the JurisGraph system. |
| `NLP Pipleline.png` | Image | Schematic diagram visualizing the sequential stages from PDF parsing to InLegalBERT tagging, vector indexing, and generation. |

---

## 3. 📁 `src/` — Core Application Codebase

The production engine of JurisGraph, structured into modular subpackages for data processing, machine learning models, retrieval-augmented generation, evaluation, and user interface.

### `src/` (Root)
| File | Type | Description |
| :--- | :--- | :--- |
| `__init__.py` | Python | Package initializer exposing the core `src` namespace. |
| `config.py` | Python | Central configuration registry containing directory paths, model identifiers, rhetorical role labels, vectorstore settings, and confidence thresholds. |

---

### `src/data/` — Document Ingestion & Dataset Handling
Handles document parsing, legal text preservation validation, sentence boundary detection, and dataset preparation.

| File | Type | Description |
| :--- | :--- | :--- |
| `__init__.py` | Python | Exports data processing and dataset classes. |
| `dataset.py` | Python | PyTorch `Dataset` implementation for batching and tokenizing sentence sequences for InLegalBERT classification. |
| `preprocessor.py` | Python | Comprehensive text extraction engine. Parses PDF/TXT files, strips headers/footers, executes sentence segmentation with legal abbreviation handling, and validates text preservation (Levenshtein distance). |

---

### `src/models/` — Rhetorical Role Classification & Sequence Modeling
Manages InLegalBERT model architecture, fine-tuning loops, sequence smoothing, and inference.

| File | Type | Description |
| :--- | :--- | :--- |
| `__init__.py` | Python | Exports model prediction and segmentation interfaces. |
| `classifier.py` | Python | Wrapper class around InLegalBERT transformer model for sequence classification across the 7 canonical rhetorical roles. |
| `segmenter.py` | Python | Full document segmentation pipeline. Executes sliding-window tokenization, InLegalBERT inference, probability calibration, and Viterbi decoding. |
| `viterbi.py` | Python | Implementation of the Viterbi algorithm using learned state transition probabilities and initial state priors to smooth sentence predictions. |
| `train.py` | Python | Training routine for fine-tuning InLegalBERT on Indian Supreme Court / High Court judgment datasets with cross-entropy loss and metric logging. |

---

### `src/rag/` — Corrective Retrieval-Augmented Generation Engine
Implements role-aware chunking, vector indexing, query routing, cross-encoder reranking, relevance grading, and agentic query correction.

| File | Type | Description |
| :--- | :--- | :--- |
| `__init__.py` | Python | Exports RAG components and pipeline orchestrator. |
| `chunker.py` | Python | Rhetorical-role-aware chunker that groups contiguous sentences with identical roles into coherent passages within strict token boundaries. |
| `vectorstore.py` | Python | ChromaDB vector store manager. Indexes chunks with BGE-small embeddings and attaches role, document ID, and positional metadata. |
| `role_mapper.py` | Python | Hybrid query routing engine using rule-based heuristics and intent classification to map queries to target legal roles. |
| `reranker.py` | Python | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) reranker scoring semantic relevance between query and retrieved legal chunks. |
| `grader.py` | Python | Relevance grading module assessing whether retrieved passages contain sufficient evidence to answer the question. |
| `rewriter.py` | Python | Query rewriting engine that reformulates ambiguous or under-retrieved queries during the self-correction loop. |
| `generator.py` | Python | LLM generation interface (Ollama `llama3.2:3b`) assembling grounded legal prompts and enforcing strict paragraph/statute citations. |
| `graph.py` | Python | LangGraph state graph compiling the complete Corrective RAG pipeline: Router -> Retriever -> Reranker -> Grader -> Correction Loop -> Generator. |
| `web_search.py` | Python | External search fallback tool (Tavily API integration with mock mode) invoked when legal queries exceed the indexed judgment scope. |

---

### `src/evaluation/` — Automated Auditing & Quality Gating
Implements LLM-as-a-Judge and quality threshold enforcement.

| File | Type | Description |
| :--- | :--- | :--- |
| `judge.py` | Python | LLM-as-a-Judge evaluation harness assessing generated answers for factual faithfulness, answer relevance, legal grounding, and hallucinations. |
| `quality_gate.py` | Python | Production quality gate verifying confidence scores, citation validity, and hallucination absence before releasing answers. |

---

### `src/ui/` — Streamlit Multi-Page Interface
Implements the "Clinical Dark" enterprise workspace with dedicated views for analysis, query, architecture, and benchmarking.

| File | Type | Description |
| :--- | :--- | :--- |
| `__init__.py` | Python | UI module exports. |
| `styles.py` | Python | CSS design system tokens, typography definitions, color constants, and custom HTML wrapper styles for Streamlit. |
| `components.py` | Python | Reusable UI rendering utilities: rhetorical role badges, KPI metric cards, citation drawers, and status indicators. |
| `data_loader.py` | Python | Secure data loader retrieving benchmark evaluation runs, confusion matrices, and latency metrics for the dashboard. |

#### `src/ui/pages/`
| File | Type | Description |
| :--- | :--- | :--- |
| `home.py` | Python | Page 1: Workspace landing view featuring system overview, value proposition cards, quick-start workflow guide, and 1-click sample judgment loaders. |
| `upload_query.py` | Python | Page 2: Primary legal workbench with document ingestion panel, text preservation audit, query bar, role filter pills, and citation drawer. |
| `pipeline.py` | Python | Page 3: Visual architecture diagram, dual-branch data flow (Ingestion vs. Query Track), and live execution trace of the current query. |
| `document_analysis.py` | Python | Page 4: Judgment inspection tool showing sentence-level rhetorical annotations, role distribution charts, and text preservation metrics. |
| `dataset_details.py` | Python | Page 5: Benchmark portfolio analysis displaying judgment specifications, question taxonomy distribution, and canonical role reference tables. |

---

## 4. 📁 `scripts/` — Benchmark Runners & Evaluation Utilities

Scripts for running experiments, calculating metrics, and compiling evaluation reports.

| File | Type | Description |
| :--- | :--- | :--- |
| `run_day2_benchmark.py` | Python | Comprehensive evaluation harness executing end-to-end benchmarks across all gold questions and calculating retrieval/generation metrics. |
| `run_judge_audit.py` | Python | Executes LLM-as-a-Judge audit across benchmark answers, measuring faithfulness, hallucination rate, and citation precision. |
| `run_deepeval_benchmark.py` | Python | Runs DeepEval framework benchmarks testing hallucination and faithfulness metrics. |
| `run_deepeval.py` | Python | Runner script for executing DeepEval test suites against RAG outputs. |
| `run_part_a_evaluation.py` | Python | Evaluates Part A: InLegalBERT sequence classification accuracy, confusion matrix, and macro F1 scores. |
| `run_part_a_signal_eval.py` | Python | Evaluates raw InLegalBERT signal confidence before and after sequence smoothing. |
| `run_viterbi_experiment.py` | Python | Evaluates Viterbi transition matrix optimization across validation and test sets. |
| `run_tau_floor_experiment.py` | Python | Sweeps relevance threshold $\tau$ to determine the optimal cutoff for reranker filtering and abstention. |
| `run_latency_benchmark.py` | Python | Profiles latency and execution time across individual pipeline stages (tokenization, retrieval, reranking, generation). |
| `run_pipelines.py` | Python | Batch execution script running end-to-end queries across multiple test documents. |
| `run_role_mapper_final_eval.py`| Python | Validates query router accuracy against gold query intent labels. |
| `run_module7_eval.py` | Python | Evaluates Module 7 retrieval performance (Recall@k, MRR) with and without role-aware filtering. |
| `run_module10_corrections.py` | Python | Benchmarks query rewriting effectiveness and Corrective RAG self-correction loops. |
| `eval_generation.py` | Python | Evaluates grounded generation quality, lexical overlap (ROUGE/BLEU), and citation precision. |
| `eval_retrieval.py` | Python | Evaluates retriever recall and precision on benchmark question-document pairs. |
| `eval_loop.py` | Python | Runs continuous evaluation loops over test suites for regression testing. |
| `eval_dev_role_mapper.py` | Python | Evaluates rule-based role mapper on development queries. |
| `eval_role_mapper_suite.py` | Python | Comprehensive test suite assessing intent detection across diverse legal phrasings. |
| `evaluate_part_a.py` | Python | Generates performance metrics for InLegalBERT sentence classification. |
| `generate_all_reports.py` | Python | Compiles raw benchmark outputs into unified markdown and JSON evaluation reports. |
| `generate_generation_report.py`| Python | Generates formatted reports detailing generation fidelity and refusal accuracy. |
| `generate_part_a_sample3_check.py` | Python | Validates segmentation and role labeling on sample3.pdf. |
| `generate_part_a_sample3_check_v2.py`| Python | Enhanced version verifying sample3.pdf segmentation edge cases. |
| `create_benchmark.py` | Python | Prepares structured benchmark files and gold Q&A pairings from raw judgment data. |
| `create_gold_paragraphs.py` | Python | Extracts ground-truth paragraph citations from legal judgments for benchmark validation. |
| `create_colab_notebook.py` | Python | Automates generation of the InLegalBERT fine-tuning Google Colab notebook (`.ipynb`). |
| `create_test_eval_colab_notebook.py` | Python | Generates the evaluation notebook for running InLegalBERT tests on Google Colab. |
| `reingest_samples.py` | Python | Re-ingests and indexes benchmark PDFs into ChromaDB vector store. |
| `measure_query_ram.py` | Python | Profiles memory utilization and RAM footprint during query execution and model inference. |
| `show_row.py` | Python | Quick CLI utility inspecting specific rows in benchmark CSV datasets. |
| `verify_generation_results.py`| Python | Validates consistency and accuracy of stored generation benchmark outputs. |
| `verify_provenance.py` | Python | Verifies data provenance and integrity across gold standard annotations. |

---

## 5. 📁 `tests/` — Automated Test Suite

Comprehensive automated test suite executed via `pytest`.

| File | Type | Description |
| :--- | :--- | :--- |
| `test_ui.py` | Python | Unit and regression tests verifying all Streamlit pages, UI components, session state integrity, and data loaders (24 passing tests). |
| `test_pdf_ingestion_preservation.py` | Python | Tests PDF extraction fidelity, whitespace normalization, and Levenshtein preservation thresholds. |
| `test_segmenter.py` | Python | Validates InLegalBERT sentence classification, sliding-window chunking, and Viterbi decoding. |
| `test_vectorstore.py` | Python | Tests ChromaDB chunk insertion, role-metadata filtering, and similarity search queries. |
| `test_role_mapper.py` | Python | Verifies query routing logic across factual, issue-related, and petitioner/respondent argument queries. |
| `test_grader.py` | Python | Tests retrieval relevance grader logic against threshold $\tau$. |
| `test_generator.py` | Python | Tests prompt assembly, Ollama generation invocation, and strict citation formatting. |
| `test_graph.py` | Python | Tests LangGraph state machine node transitions, state updates, and end-to-end execution. |
| `test_loop.py` | Python | Tests the corrective rewrite loop and fallback conditions under low-relevance retrieval. |
| `test_rag_abstention_fix.py` | Python | Verifies system abstention and graceful refusal on unanswerable/out-of-scope legal queries. |
| `test_quality_gate.py` | Python | Tests quality gate pass/fail criteria based on confidence thresholds and hallucination checks. |
| `test_judge.py` | Python | Tests LLM-as-a-Judge grading prompts, score extraction, and audit report generation. |
| `test_case_identification.py`| Python | Tests case title, citation, and bench metadata extraction from judgment headers. |
| `test_audit_regression.py` | Python | Regression test suite preventing regressions in LLM-as-a-Judge scores across revisions. |

---

## 6. 📁 `legalDataset/` — Legal Corpus & Gold Annotations

Contains Indian court judgments and gold-standard evaluation benchmarks.

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| `train.csv` | CSV | Training split containing labeled sentences and rhetorical roles for InLegalBERT fine-tuning. |
| `val.csv` | CSV | Validation split used during model training and hyperparameter tuning. |
| `test.csv` | CSV | Independent test split (149,868 sentences) used for final empirical evaluation. |
| `samples/` | Directory | Subdirectory housing curated High Court benchmark judgment PDFs and ground-truth annotations: |
| `samples/sample1.pdf` | PDF | Indian High Court judgment 1 (Large reference corpus, ~30MB). |
| `samples/sample2.pdf` | PDF | Indian High Court Writ Tax Judgment (42 pages, Section 132/131(1A) search and notice challenge). |
| `samples/sample3.pdf` | PDF | Indian High Court Original Side Appeal (33 pages, Article 215 contempt application and IBC moratorium). |
| `samples/gold_questions.csv` | CSV | 32 curated gold-standard questions mapped to target rhetorical roles and source paragraphs. |
| `samples/gold_unanswerable.csv` | CSV | 8 out-of-scope / unanswerable adversarial questions for testing model abstention. |
| `samples/gold_keyfacts.csv` | CSV | Ground-truth key facts extracted from benchmark judgments for factual recall audits. |
| `samples/gold_paragraphs.json` | JSON | Ground-truth paragraph boundaries and content for citation verification. |

---

## 7. 📁 `notebooks/` — Google Colab Workbooks

Jupyter notebooks configured for cloud-accelerated training and benchmarking.

| File | Type | Description |
| :--- | :--- | :--- |
| `InLegalBERT_FineTuning_Colab.ipynb` | Jupyter Notebook | Complete GPU training notebook for fine-tuning InLegalBERT on the Indian legal dataset using Hugging Face Trainer. |
| `InLegalBERT_Test_Eval_Colab.ipynb` | Jupyter Notebook | Standalone notebook for evaluating the fine-tuned checkpoint on the test set and computing confusion matrices. |
| `README.md` | Markdown | Usage instructions, GPU configuration guidelines, and dataset download instructions for Google Colab. |

---

## 8. 📁 `artifacts/` — Models, Parameters & Evaluation Reports

Stores trained weights, sequence parameters, and experimental findings.

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| `viterbi_params.json` | JSON | State transition matrix, prior probabilities, and emission mapping for Viterbi HMM sequence smoothing. |
| `viterbi_params_v1.json` | JSON | Baseline Viterbi transition parameters saved during initial tuning. |
| `label_mapping.json` | JSON | Mapping between InLegalBERT integer class IDs and the 7 canonical rhetorical role names. |
| `tau_experiment_results.json` | JSON | Empirical results from relevance threshold $\tau$ sweep experiments. |
| `role_mapper_benchmark.json` | JSON | Evaluation metrics measuring query router performance across question categories. |
| `role_mapper_evaluation_report.json` | JSON | Detailed classification report and confusion breakdown for the role mapper. |
| `model_checkpoint/` | Directory | Local storage directory for fine-tuned InLegalBERT model weights, tokenizer, and config files. |
| `chroma_db/` | Directory | Persistent ChromaDB vector database storing document chunks and embeddings. |
| `samples/` | Directory | Stores segmented text representations (`sample2_segmented.txt`, `sample3_segmented.txt`, etc.) with attached rhetorical roles. |
| `reports/` | Directory | Houses generated technical reports: `generation_report.md`, `failure_analysis.md`, `judge_audit.md`, `module10_report.md`. |
| `eval_reports/` | Directory | Detailed empirical metrics: classification reports (`test_evaluation_report.json`), training loss logs, and `viterbi_report.md`. |

---

## 9. 📁 `scratch/` — Diagnostic Scripts & Experimental Tools

Working scratch directory containing isolated tests, diagnostic probes, and experimental prototypes developed during iterative engineering.

- **Sample Inspection**: `inspect_sample3_contempt.py`, `inspect_party_details.py`, `check_sample2_roles.py`, `check_sample3_facts.py` (probes specific judgment sections).
- **Segmentation & Paragraphs**: `find_gold_sentence_spans.py`, `test_paragraph_spans_detection.py`, `test_heading_split.py` (sentence and paragraph boundary diagnostics).
- **Retrieval & Reranker Diagnostics**: `check_rank_117.py`, `test_rerank_top20.py`, `score_q1_chunks.py`, `test_retrieval_with_fix.py` (analyzes chunk ranking and score distributions).
- **Generation & LLM Prototyping**: `test_gen_3chunks.py`, `test_prompt_compound.py`, `test_new_refusals.py`, `build_generation_results.py` (tests prompt engineering and generation formats).
- **Pipeline Tracing**: `trace_four_questions.py`, `diagnose_query.py`, `test_all_queries.py` (traces execution paths and node outputs across sample queries).
- **Gate & Verification Scripts**: `test_phase3_live_gate.py`, `test_phase4_safe_failure.py`, `test_phase5_ui_integration.py`, `test_phase6_generalisation.py` (verifies architectural milestone gates).

---

## 10. 📁 Environment & Hidden Tooling Folders

| Folder | Description |
| :--- | :--- |
| `.venv/` | Python 3.11 virtual environment containing all installed libraries (PyTorch, Transformers, ChromaDB, Streamlit, etc.). |
| `.agents/` | Custom Antigravity agent skills and project guidance configurations. |
| `.claude/` | Claude Code CLI memory files and interaction settings. |
| `.deepeval/` | Cache directory created by the DeepEval framework for evaluation runs. |
| `.pytest_cache/` | Cache directory used by pytest to optimize test runs and track failures. |
| `data/chroma_db/` | Persistent vector store cache storing chunk embeddings and inverted indices. |

# Document 02 — TRD

**Frontend**
Streamlit (Python). No JS framework needed — this is a research tool, not a consumer product, and Streamlit lets you build the query UI, role-filter display, and grading trace visualization directly in Python without context-switching stacks. Swap for Next.js only if you later want a polished public demo.

**Backend**
Python, orchestrated as a LangGraph application (FastAPI wrapping the graph if you need a separate API layer for Streamlit to call; otherwise Streamlit can invoke the graph directly in-process for the prototype).

**Database**
- ChromaDB (local, persistent client) — vector store for role-tagged judgment chunks; this is your primary "database," not a relational one.
- SQLite — lightweight store for query logs, grading traces, and DeepEval run history, so you have evaluation history without standing up Postgres.

**Auth**
None for the prototype — single-user local/academic tool. Skip this line item entirely rather than bolting on Supabase Auth for a system nobody but you and your reviewer will touch.

**Hosting**
- Local machine or Google Colab (GPU) for InLegalBERT fine-tuning and inference.
- Local machine for the Streamlit app + LangGraph pipeline during development/review.
- If you need a shareable demo link: Hugging Face Spaces (free GPU-optional tier, handles Streamlit natively) is the feasible option — avoid Vercel/cloud GPU hosting, which adds cost and complexity you don't need for a course project.

**Third-party APIs**
| Name | Purpose | Tier |
|---|---|---|
| Ollama (local, not an API in the network sense) | Hosts Llama-3.2-3B-Instruct for the Relevance Grader and Query Rewriter nodes | Free, local |
| Tavily Search API | Fallback web search for Ambiguous-graded chunks | Free tier (1,000 calls/month) |
| Hugging Face Hub | Download InLegalBERT base weights, host your fine-tuned model | Free |
| DeepEval + an LLM-as-judge endpoint (can reuse Ollama, or OpenAI/Gemini if you want a stronger judge) | Faithfulness/Context Precision/Recall scoring | Free if judge is local Ollama; paid if using GPT-4/Gemini as judge |

**Key Libraries**
- `transformers`, `torch` — InLegalBERT fine-tuning and inference
- `langgraph`, `langchain-core` — state machine orchestration
- `chromadb` — vector storage and metadata filtering
- `sentence-transformers` — embedding model for chunk vectors
- `deepeval` — evaluation suite (Faithfulness, Answer Relevancy, Contextual Precision/Recall)
- `ollama` (Python client) — local LLM calls for grading/rewriting
- `tavily-python` — web search fallback
- `streamlit` — UI
- `pandas`, `scikit-learn` — training data handling, F1/confusion-matrix reporting for InLegalBERT eval

**Environment Variables**
- `TAVILY_API_KEY`
- `HUGGINGFACE_TOKEN`
- `OLLAMA_HOST` (defaults to `localhost:11434` if unset)
- `CHROMA_PERSIST_DIR`
- `DEEPEVAL_JUDGE_MODEL` (and its API key, only if you use a paid judge instead of local Ollama)

**Constraints**
- Must run end-to-end on a single consumer GPU (or free-tier Colab GPU) — no dependency on 70B+ models or paid inference at scale.
- Must work fully offline for the core pipeline (segmentation, retrieval, grading, generation); only the Tavily fallback and optional cloud judge require internet.
- Grader and rewriter LLM calls must run at temperature 0 for deterministic, reproducible grading — non-negotiable for your evaluation metrics to mean anything.
- Retry loops (query rewriting) must be hard-capped in code, not just documented, to prevent infinite LangGraph cycles.
- No production-grade auth/scaling requirements — this is a single-user academic prototype, not a deployed multi-tenant product.

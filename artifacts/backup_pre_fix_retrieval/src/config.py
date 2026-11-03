"""
JurisGraph System Configuration
Defines paths, rhetorical role labels, model parameters, and environment settings.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "legalDataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "model_checkpoint"
REPORTS_DIR = ARTIFACTS_DIR / "eval_reports"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"

# Embedding Model (BAAI/bge-small-en-v1.5)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ROLE_COLLECTION_NAME = "jurisgraph_role_chunks"
NAIVE_COLLECTION_NAME = "jurisgraph_naive_chunks"

# Fine-Tuned InLegalBERT Model (Single Source of Truth for Segmenter)
# artifacts/model_checkpoint/best_model contains the genuine fine-tuned model checkpoint (Full-Val Macro F1: ~0.5572).
# Handles both direct and nested directory extractions seamlessly.
# NOTE: artifacts/model_checkpoint/best_model_smoke_test contains untrained/dummy smoke-test weights
# and must NEVER be used by the segmenter.
BEST_MODEL_DIR = CHECKPOINT_DIR / "best_model"
BEST_MODEL_PATH = (
    BEST_MODEL_DIR / "best_model"
    if (BEST_MODEL_DIR / "best_model" / "model.safetensors").exists()
    else BEST_MODEL_DIR
)

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Splits
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# Canonical 7 LegalSeg Rhetorical Roles (Strict Casing & Order)
CANONICAL_LABELS = [
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision",
    "None",
]

# Canonical Label Mappings
LABEL2ID = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(CANONICAL_LABELS)}
LABEL_MAPPING_FILE = ARTIFACTS_DIR / "label_mapping.json"
VITERBI_PARAMS_FILE = ARTIFACTS_DIR / "viterbi_params.json"

# InLegalBERT Base Model
BASE_MODEL_NAME = "law-ai/InLegalBERT"
MAX_SEQ_LENGTH = 128
NUM_LABELS = len(CANONICAL_LABELS)

# Ollama, Cross-Encoder & LangGraph Settings
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.0
MAX_CRAG_RETRIES = 2
CHUNK_TOKEN_LIMIT = 256
CHUNK_MAX_CHAR_GRADING = 800
ENABLE_WEB_FALLBACK = True
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module 10: DeepEval & Judge Settings
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini")  # "gemini" or "groq"
JUDGE_MODEL_PRIMARY = "gemini-3.5-flash-lite"
JUDGE_MODEL_SECONDARY = "gemma-4-26b-a4b-it"
JUDGE_CROSSCHECK_MODEL = "gemma-4-31b-it"

# Account Limits
# Gemma 4: 30 RPM, 16K TPM, 14.4K RPD
GEMMA_RPM_LIMIT = 30
GEMMA_TPM_LIMIT = 16000
GEMMA_RPD_LIMIT = 14400

# Flash-Lite: 15 RPM, 250K TPM, 500 RPD
FLASH_LITE_RPM_LIMIT = 15
FLASH_LITE_TPM_LIMIT = 250000
FLASH_LITE_RPD_LIMIT = 500

# Rate Limiter Operating Thresholds (sliding 60-second window)
RATE_LIMIT_MAX_TPM = 14000
RATE_LIMIT_MAX_RPM = 14
JUDGE_DAILY_REQUEST_BUDGET = int(os.getenv("JUDGE_DAILY_REQUEST_BUDGET", "450"))

# Legacy/Groq fallback configurations (only used if JUDGE_PROVIDER == "groq")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
GROQ_JUDGE_MODEL_CHEAP = os.getenv("GROQ_JUDGE_MODEL_CHEAP", "qwen/qwen3.8-27b")
JUDGE_DAILY_TOKEN_BUDGET = int(os.getenv("JUDGE_DAILY_TOKEN_BUDGET", "100000"))
ENABLE_ONLINE_GATE = os.getenv("ENABLE_ONLINE_GATE", "false").lower() in ("true", "1", "yes")


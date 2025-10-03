"""Centralised configuration for the local RAG stack."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_MD_PATH = DATA_DIR / "processed" / "sex_harassment_prevention_act.md"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"

# Qdrant storage settings. When STORAGE_PATH is None the client runs fully in memory.
QDRANT_COLLECTION = "legal_chunks"
QDRANT_STORAGE_PATH = DATA_DIR / "vector_store"
QDRANT_REFRESH = False  # Set True to force re-ingest regardless of cached state.
RERANKER_POOL_SIZE = 100

# Model identifiers (assumed to be available locally via the Hugging Face cache).
GENERATION_MODEL = "models/qwen"
EMBEDDING_MODEL = "models/embedding"
RERANKER_MODEL = "models/reranker"

# Retrieval hyperparameters.
TOP_K = 5
MAX_CONTEXT_CHARS = 1024
ENABLE_RERANKER = True
CONVERSATION_WINDOW = 3

# Default generation parameters.
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9

# Logging configuration.
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PREFIX = "rag_queries"

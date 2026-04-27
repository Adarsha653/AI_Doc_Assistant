"""Runtime configuration from environment (no secrets here)."""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# Document chunking (used by DocumentLoader)
DOC_CHUNK_SIZE = _int("DOC_CHUNK_SIZE", 1000)
DOC_CHUNK_OVERLAP = _int("DOC_CHUNK_OVERLAP", 200)

# Retrieval
RETRIEVER_K = _int("RETRIEVER_K", 5)
# Broader questions (e.g. "tell me about this resume") need more chunks than a pinpoint fact.
RETRIEVER_K_OVERVIEW = _int("RETRIEVER_K_OVERVIEW", 10)
# "Who is this" — only header/contact-focused chunks; large k makes the model summarize the whole resume.
RETRIEVER_K_IDENTITY = _int("RETRIEVER_K_IDENTITY", 5)
# Compare / relate multiple uploads — need enough chunks to span files, then interleaved per file.
RETRIEVER_K_MULTI = _int("RETRIEVER_K_MULTI", 15)
# When comparing uploads: max chunks **per file** after balancing (total ≈ this × number of files).
MULTI_FILE_CHUNKS_PER_FILE = _int("MULTI_FILE_CHUNKS_PER_FILE", 5)
# Initial similarity pool size before per-file carve (large docs need a bigger pool).
MULTI_FILE_POOL_K = _int("MULTI_FILE_POOL_K", 180)

# Embeddings model id (SentenceTransformers / Hugging Face hub id)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
# Room for structured answers; raise via GROQ_MAX_TOKENS if replies still cut off mid-sentence.
GROQ_MAX_TOKENS = _int("GROQ_MAX_TOKENS", 3072)
# Tight cap for "who is this" / name questions — short answer only (avoids mid-sentence cuts from long ramble).
GROQ_MAX_TOKENS_IDENTITY = _int("GROQ_MAX_TOKENS_IDENTITY", 256)
# Lower default keeps answers closer to the documents (yes/no, names, facts).
GROQ_TEMPERATURE = _float("GROQ_TEMPERATURE", 0.35)
# Nucleus sampling; slightly below 1.0 can reduce rambling.
GROQ_TOP_P = _float("GROQ_TOP_P", 0.92)
# Reduces repeated phrases in long answers (0 = off; typical 0.1–0.3).
GROQ_FREQUENCY_PENALTY = _float("GROQ_FREQUENCY_PENALTY", 0.12)

# Local HF fallback (transformers pipeline)
HF_LOCAL_MODEL = os.getenv("HF_LOCAL_MODEL", "google/flan-t5-small")

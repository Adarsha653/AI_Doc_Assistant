# AI_Doc_Assistant

Small document question-answering assistant with a Chainlit UI. Upload PDFs/TXT (extendable to DOCX/PPTX/images), the app builds a FAISS vectorstore and answers user questions using Groq (primary) and Hugging Face (hosted or local transformers) as fallbacks.

## Features
- Upload PDF / TXT and extract text
- Build and persist FAISS vectorstore for retrieval
- Retrieve relevant passages and answer with:
  - Groq (configurable via `GROQ_MODEL`)
  - Hugging Face hosted inference or local `transformers` fallback
- Defensive error handling and debug logs
- Local fallback downloads models if hosted inference is unavailable

## Project structure
- app.py — Chainlit entrypoint and UI handlers
- src/
  - qa.py — QASystem: retrieval + LLM orchestration (Groq, HF, local fallback)
  - document_loader.py — load/parse uploaded documents (PDF/TXT; extendable)
  - other modules (helpers, embeddings) as needed
- .chainlit/ — Chainlit UI assets and translations
- vectorstore/ — persisted FAISS data (ignored in git)
- .venv* / venv / .venv_groq — virtualenvs (ignored)
- README.md, requirements.txt, .gitignore

## Requirements
- macOS (instructions use bash/zsh)
- Python 3.10+ (3.11 recommended)
- Git

## Environment variables (do NOT commit real keys)
- HUGGINGFACE_API_KEY — Hugging Face token (optional; required for HF hosted)
- GROQ_API_KEY — Groq API key (required if using Groq)
- GROQ_MODEL — Groq model id (e.g. `openai/gpt-oss-120b`)
- USE_GROQ — optional toggle (`true`/`false`)
Store these in a local `.env` (never commit).

## Quick start (local)
1. Create and activate main venv
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
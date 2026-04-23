from __future__ import annotations

import difflib
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config

# Paths sometimes leak into PDF text (links, viewers); remove before display / LLM.
_LEAKED_PATH_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.-]{0,72}@?(?:/Users/|/home/)[^\s]+", re.IGNORECASE),
    re.compile(r"(?:file:/?)+[^\s]+", re.IGNORECASE),
    re.compile(
        r"\b[A-Za-z]:\\(?:[^\\\s]+\\)+[^\s\\\n]+\.(?:pdf|docx?|txt|md)\b",
        re.IGNORECASE,
    ),
    re.compile(r"/[^\s]*?/workspaceStorage/[^\s]+", re.IGNORECASE),
    re.compile(r"/[^\s]*?/pdfs/[^\s]+\.pdf\b", re.IGNORECASE),
)


def strip_leaked_paths(text: str) -> str:
    """Remove absolute paths that sometimes appear inside extracted PDF text."""
    t = text
    for rx in _LEAKED_PATH_RES:
        t = rx.sub(" ", t)
    return t


def is_overview_style_query(question: str) -> bool:
    q = (question or "").lower()
    if not q.strip():
        return False
    phrases = (
        "tell me about",
        "tell me the topic",
        "topic of this",
        "topic of the",
        "main topic",
        "what is the topic",
        "what's the topic",
        "describe this",
        "describe the",
        "summary of",
        "summarize",
        "summarise",
        "overview",
        "walk me through",
        "high-level",
        "high level",
        "what is this",
        "what's this",
        "what does this",
        "who is this",
        "what can you tell me",
        "give me an overview",
        "quick overview",
        "brief overview",
    )
    if any(p in q for p in phrases):
        return True
    doc_words = ("resume", "cv", "document", "file", "profile")
    ask_words = ("about", "overview", "summarize", "summarise", "describe", "explain")
    return any(d in q for d in doc_words) and any(a in q for a in ask_words)


def wants_broader_retrieval(question: str) -> bool:
    """Use more chunks for open summaries and for role-fit / opinion questions."""
    if is_overview_style_query(question):
        return True
    q = (question or "").lower()
    needles = (
        "good for",
        "good as",
        "strong for",
        "weak for",
        "right for",
        "suitable for",
        "fit for",
        "prepared for",
        "qualified for",
        "match this role",
        "for this role",
        "for a data scientist",
        "for data scientist",
        "data scientist role",
        "data science role",
        "ml engineer role",
        "software engineer role",
        "should i apply",
        "would i get",
        "hire them",
        "stand out",
        "strengths",
        "weaknesses",
        "what's missing",
        "whats missing",
        "improve this resume",
        "improve my resume",
        "do you think",
        "would you say",
        "is this resume",
        "is this cv",
        "how good",
        "competitive",
        "good resume",
        "bad resume",
    )
    return any(n in q for n in needles)


def resume_chunk_priority(doc: Document) -> int:
    """Higher = show first for resume-wide questions (header, summary, education, skills)."""
    raw = doc.page_content or ""
    head = raw[:800].lower()
    full = raw.lower()
    score = 0
    if "@" in head and any(
        x in head for x in ("gmail", "outlook", "edu", "linkedin", "github", "http")
    ):
        score += 8
    if "summary" in head or "professional summary" in full:
        score += 6
    if "master of" in full and "data" in full:
        score += 4
    if "education" in full and "university" in full:
        score += 4
    if "experience" in head or "work experience" in full:
        score += 3
    if any(x in full for x in ("skills", "technical skills", "technologies")):
        score += 2
    if "project" in full and any(
        x in full for x in ("pyspark", "machine learning", "databricks", "mlflow", "pandas")
    ):
        score += 1
    return score


def humanize_excerpt_preview_line(preview: str) -> str:
    """Avoid leading mid-sentence fragments like '(Pandas, NumPy) to analyze…'."""
    t = (preview or "").strip()
    if len(t) < 55:
        return t
    if t[0] in "([":
        for needle in (
            "● ",
            "●",
            "Built ",
            "Summary ",
            "Education ",
            "Experience ",
            "Developed ",
            "Designed ",
            "Side Projects",
            "Technical Projects",
        ):
            k = t.find(needle)
            if 0 < k < 360:
                out = t[k:].lstrip("● ").strip()
                if len(out) > 45:
                    return out
        m = re.search(r"\.\s+[A-Z][a-z]", t[:420])
        if m:
            return t[m.start() + 1 :].strip()
    low = t[:24].lower()
    if low.startswith("and ") or low.startswith("or ") or low.startswith("but "):
        dot = t.find(". ", 0, 160)
        if dot != -1:
            return t[dot + 2 :].strip()
    return t


def wants_multi_file_retrieval(question: str) -> bool:
    """Questions that need passages from more than one upload (compare, relate, etc.)."""
    q = (question or "").lower()
    if not q.strip():
        return False
    multi = (
        "these files",
        "those files",
        "multiple files",
        "three files",
        "3 files",
        "two files",
        "2 files",
        "four files",
        "4 files",
        "all the files",
        "all files",
        "each file",
        "every file",
        "across files",
        "between the files",
        "between files",
        "the files i",
        "files i uploaded",
        "documents i uploaded",
    )
    compare = (
        "compare",
        "comparison",
        "contrast",
        "difference between",
        "differences between",
        "how are",
        "how do these",
        "relationship",
        "related",
        "relate",
        "same or different",
        "overlap",
    )
    if any(m in q for m in multi):
        return True
    if re.search(r"\b\d+\s+files?\b", q):
        return True
    if re.search(r"these\s+\d+\s+files?", q):
        return True
    if any(c in q for c in compare) and any(
        w in q for w in ("file", "document", "pdf", "doc", "upload", "resume", "report")
    ):
        return True
    return False


def _doc_file_label(doc: Document) -> str:
    m = doc.metadata or {}
    return (m.get("file_name") or Path(str(m.get("source") or "")).name or "unknown").strip()


def _doc_dedupe_key(doc: Document) -> tuple[str, str]:
    return (_doc_file_label(doc), (doc.page_content or "")[:400])


def diversify_chunks_by_file(
    scored: List[Tuple[Document, float]], k: int
) -> List[Tuple[Document, float]]:
    """Interleave best chunks per file so one document does not dominate retrieval."""
    if k <= 0 or not scored:
        return scored
    buckets: dict[str, List[Tuple[Document, float]]] = defaultdict(list)
    for doc, score in scored:
        buckets[_doc_file_label(doc)].append((doc, score))
    for fn in buckets:
        buckets[fn].sort(key=lambda x: float(x[1]))
    keys = [fn for fn, items in buckets.items() if items]
    if len(keys) <= 1:
        return scored[:k]

    out: List[Tuple[Document, float]] = []
    while len(out) < k and any(buckets.values()):
        for fn in list(keys):
            if len(out) >= k:
                break
            b = buckets.get(fn, [])
            if b:
                out.append(b.pop(0))
        keys = [fn for fn in keys if buckets.get(fn)]
    return out


# Groq / HF: steer toward direct, grounded answers (plain language; avoid jargon like “excerpt”.)
_QA_SYSTEM_PROMPT = """You answer using **only** the document text below. Do not invent facts.

Rules:
- If the question is yes/no and the document text clearly supports it, start with **Yes** or **No**, then one short sentence pointing to what the document says (e.g. a name or heading).
- If the answer is not in the document, say so in one sentence.
- Use readable Markdown (short paragraphs, bullets when helpful). Avoid huge tables unless the user asks.
- When the document states something plainly (e.g. a resume header with a person's name), answer directly—do not hedge with unnecessary disclaimers.
- Prefer everyday words (**document**, **file**, **upload**, **section**, **what it says**). Do **not** use words like **excerpt**, **passage**, or **corpus** in your reply.
- For **broad** requests (summarize this resume/CV/document, tell me about it, overview, walk me through it): respond with **scannable sections** using `##` headings such as Contact, Professional summary, Experience, Education, Skills & tools, Projects, Languages, Other. Open with 2–4 sentences on who they are and their focus, then the sections—**synthesize**; do not dump or paraphrase every bullet from the document verbatim.
- For **role-fit or opinion** questions (e.g. “Is this resume good for a data scientist role?”): answer from what the document shows; give a concise view (strengths, gaps, who it fits) in plain language. If you only have part of the file below, say briefly what you can and cannot judge from that—do not invent credentials not present in the document.
- The text below is split into blocks headed by **Source file:** with the filename. When several **different** filenames appear, they are **different uploads**—do not claim they are “the same document” unless every block shares one identical filename. Explain how they relate using what each file shows.
- **No repetition:** if the same idea appears in more than one block below, state it **once**. Do not copy the same paragraph or sentence multiple times, and do not loop a closing summary. Finish when the question is answered."""


class QASystem:
    def __init__(self, embeddings, vector_store_path: str = "vectorstore"):
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.vector_store: FAISS | None = None

    def _index_exists(self) -> bool:
        return os.path.exists(os.path.join(self.vector_store_path, "index.faiss"))

    def has_index(self) -> bool:
        return self.vector_store is not None or self._index_exists()

    def build_vectorstore(self, documents: List[Document]) -> None:
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved to {self.vector_store_path}")

    def load_vectorstore(self) -> None:
        if self.vector_store is None:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def _ensure_vector_store(self) -> None:
        if self.vector_store is None:
            if self._index_exists():
                self.load_vectorstore()
            else:
                raise RuntimeError("No vector store in memory or on disk.")

    def ingest_documents(self, documents: List[Document]) -> None:
        """Create index on first ingest; append chunks when an index already exists."""
        if not documents:
            return
        if self.vector_store is None and self._index_exists():
            self.load_vectorstore()
        if self.vector_store is None:
            self.build_vectorstore(documents)
        else:
            self.vector_store.add_documents(documents)
            self.vector_store.save_local(self.vector_store_path)

    def list_indexed_file_names(self) -> List[str]:
        """Distinct `file_name` labels in the index (sorted), excluding unknown."""
        self._ensure_vector_store()
        assert self.vector_store is not None
        vs = self.vector_store
        labels: set[str] = set()
        mapping = getattr(vs, "index_to_docstore_id", None) or {}
        for _, doc_id in mapping.items():
            try:
                d = vs.docstore.search(doc_id)
            except Exception:
                continue
            if isinstance(d, Document):
                lab = _doc_file_label(d)
                if lab and lab != "unknown":
                    labels.add(lab)
        return sorted(labels)

    def _chunks_grouped_by_file_from_index(self) -> dict[str, List[Document]]:
        """All indexed chunks grouped by display filename (for backfilling small uploads)."""
        self._ensure_vector_store()
        assert self.vector_store is not None
        vs = self.vector_store
        out: dict[str, List[Document]] = defaultdict(list)
        mapping = getattr(vs, "index_to_docstore_id", None) or {}
        for idx, doc_id in sorted(mapping.items(), key=lambda kv: kv[0]):
            try:
                d = vs.docstore.search(doc_id)
            except Exception:
                continue
            if isinstance(d, Document):
                out[_doc_file_label(d)].append(d)
        return out

    def _retrieve_balanced_across_files(
        self, question: str, file_names: List[str]
    ) -> List[Tuple[Document, float]]:
        """Guarantee chunks from each uploaded file; similarity-only search loses small PDFs."""
        self._ensure_vector_store()
        assert self.vector_store is not None
        vs = self.vector_store
        per = max(2, config.MULTI_FILE_CHUNKS_PER_FILE)
        cap = min(48, per * len(file_names) + 8)
        pool_k = min(500, max(config.MULTI_FILE_POOL_K, per * len(file_names) * 12))
        pool = vs.similarity_search_with_score(question, k=pool_k)

        by_pool: dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        for doc, sc in pool:
            by_pool[_doc_file_label(doc)].append((doc, float(sc)))
        for fn in by_pool:
            by_pool[fn].sort(key=lambda x: x[1])

        seen: set[tuple[str, str]] = set()
        out: List[Tuple[Document, float]] = []

        def take_from_pool(fn: str) -> bool:
            b = by_pool.get(fn, [])
            while b:
                doc, sc = b.pop(0)
                k = _doc_dedupe_key(doc)
                if k in seen:
                    continue
                seen.add(k)
                out.append((doc, sc))
                return True
            return False

        for _ in range(per):
            for fn in file_names:
                if len(out) >= cap:
                    break
                take_from_pool(fn)
            if len(out) >= cap:
                break

        full = self._chunks_grouped_by_file_from_index()
        for fn in file_names:
            need = per - sum(1 for d, _ in out if _doc_file_label(d) == fn)
            if need <= 0:
                continue
            for doc in full.get(fn, []):
                if need <= 0:
                    break
                k = _doc_dedupe_key(doc)
                if k in seen:
                    continue
                seen.add(k)
                out.append((doc, 1e6))
                need -= 1

        return out[:cap]

    def retrieve_with_scores(
        self, question: str, session_file_names: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        self._ensure_vector_store()
        assert self.vector_store is not None
        names = [x.strip() for x in (session_file_names or []) if x.strip()]
        if wants_multi_file_retrieval(question) and len(names) >= 2:
            return self._retrieve_balanced_across_files(question, names)

        if wants_multi_file_retrieval(question):
            k = config.RETRIEVER_K_MULTI
        elif wants_broader_retrieval(question):
            k = config.RETRIEVER_K_OVERVIEW
        else:
            k = config.RETRIEVER_K
        scored = self.vector_store.similarity_search_with_score(question, k=k)
        if wants_multi_file_retrieval(question) and len(scored) > 1:
            scored = diversify_chunks_by_file(scored, k)
        return scored

    @staticmethod
    def normalize_chunk_text(text: str) -> str:
        """Collapse PDF-style runs of spaces/tabs and stray zero-width chars; keep line breaks."""
        if not text:
            return ""
        t = strip_leaked_paths(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\u200b", "").replace("\ufeff", "")
        lines_out: List[str] = []
        for line in t.split("\n"):
            line = re.sub(r"[ \t\u00a0\f\v]+", " ", line).strip()
            if line:
                lines_out.append(line)
        return "\n".join(lines_out).strip()

    @staticmethod
    def dedupe_retrieved_documents(
        docs: List[Document], *, similarity_threshold: float = 0.86, compare_len: int = 1400
    ) -> List[Document]:
        """Drop near-duplicate chunks (overlap + similar headings) so the model is not fed the same paragraph many times."""
        kept: List[Document] = []
        norms: List[str] = []
        for d in docs:
            raw = d.page_content or ""
            n = QASystem.normalize_chunk_text(raw)
            if len(n) < 72:
                kept.append(d)
                norms.append(n)
                continue
            head = n[:compare_len]
            redundant = False
            for prev in norms:
                if len(prev) < 72:
                    continue
                pr = prev[:compare_len]
                if difflib.SequenceMatcher(None, head, pr).ratio() >= similarity_threshold:
                    redundant = True
                    break
            if not redundant:
                kept.append(d)
                norms.append(n)
        return kept

    @staticmethod
    def context_from_docs(docs: List[Document]) -> str:
        blocks: List[str] = []
        for d in docs:
            body = QASystem.normalize_chunk_text(d.page_content or "")
            if not body:
                continue
            label = _doc_file_label(d)
            blocks.append(f"### Source file: {label}\n{body}")
        return "\n\n".join(blocks).strip()

    def _extract_basic_info(self, text: str) -> dict:
        info = {}
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if m:
            info["email"] = m.group(0)
        m = re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text)
        if m:
            info["phone"] = m.group(0)
        m = re.search(r"^(?:Name[:\s]*)([A-Z][A-Za-z\s]{1,80})", text, re.IGNORECASE | re.MULTILINE)
        if m:
            info["name"] = m.group(1).strip()
        else:
            first_lines = text.strip().splitlines()[:8]
            for line in first_lines:
                cand = line.strip()
                if cand and re.match(r"^[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3}$", cand):
                    info.setdefault("name", cand)
                    break
        return info

    def _answer_with_groq(self, context: str, question: str) -> Optional[str]:
        try:
            from groq import Groq

            model = config.GROQ_MODEL
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": _QA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Document text:\n{context}\n\nQuestion: {question}",
                    },
                ],
                model=model,
                max_tokens=config.GROQ_MAX_TOKENS,
                temperature=config.GROQ_TEMPERATURE,
            )
            try:
                return resp.choices[0].message.content
            except Exception:
                if hasattr(resp, "content"):
                    return str(resp.content)
                try:
                    return json.dumps(resp, default=str)
                except Exception:
                    return str(resp)
        except Exception:
            traceback.print_exc()
            return None

    def _stream_groq(self, context: str, question: str) -> Generator[str, None, None]:
        from groq import Groq

        model = config.GROQ_MODEL
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Document text:\n{context}\n\nQuestion: {question}",
                },
            ],
            model=model,
            max_tokens=config.GROQ_MAX_TOKENS,
            temperature=config.GROQ_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta

    def _normalize_hf(self, resp) -> Optional[str]:
        if resp is None:
            return None
        if isinstance(resp, dict):
            if "generated_text" in resp:
                return resp["generated_text"]
            if "data" in resp and isinstance(resp["data"], list) and resp["data"]:
                first = resp["data"][0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
            if "error" in resp:
                return None
        if isinstance(resp, list) and resp:
            first = resp[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for v in first.values():
                    if isinstance(v, str) and v.strip():
                        return v
        if hasattr(resp, "generated_text"):
            return getattr(resp, "generated_text")
        if hasattr(resp, "text"):
            return getattr(resp, "text")
        if hasattr(resp, "content"):
            return str(getattr(resp, "content"))
        if hasattr(resp, "generations"):
            try:
                gens = getattr(resp, "generations")
                first = gens[0]
                if isinstance(first, list):
                    return first[0].text
                return first.text
            except Exception:
                pass
        try:
            s = str(resp)
            return s if s.strip() else None
        except Exception:
            return None

    def _answer_with_hf(self, context: str, question: str) -> Optional[str]:
        token = os.getenv("HUGGINGFACE_API_KEY")
        prompt = (
            f"{_QA_SYSTEM_PROMPT}\n\nDocument text:\n{context}\n\nQuestion: {question}"
        )

        try:
            from huggingface_hub import InferenceApi as _InferenceApi
        except ImportError:
            _InferenceApi = None  # type: ignore[misc, assignment]

        model_candidates = [
            "google/flan-t5-large",
            "google/flan-t5-small",
            "facebook/bart-large-cnn",
            "gpt2",
        ]

        for model in model_candidates:
            print(f"DEBUG: Trying model: {model}")

            if token:
                try:
                    from huggingface_hub import InferenceClient

                    client = InferenceClient(token=token)
                    call_attempts = [
                        lambda m=model: client.text_generation(model=m, inputs=prompt, max_new_tokens=256),
                        lambda m=model: client.text_generation(model=m, prompt=prompt, max_new_tokens=256),
                        lambda m=model: client.text_generation(m, prompt, max_new_tokens=256),
                        lambda m=model: client.text_generation(prompt, model=m, max_new_tokens=256),
                    ]
                    resp = None
                    for call in call_attempts:
                        try:
                            resp = call()
                            print("DEBUG: InferenceClient call succeeded for model", model)
                            break
                        except TypeError:
                            continue
                        except StopIteration:
                            print("DEBUG: InferenceClient provider mapping StopIteration for model", model)
                            resp = None
                            break
                        except Exception as e:
                            print("DEBUG: InferenceClient call exception:", repr(e))
                            resp = None
                            break
                    out = self._normalize_hf(resp)
                    if out:
                        print("DEBUG: Normalized response from InferenceClient")
                        return out
                except Exception as e:
                    print("DEBUG: InferenceClient construction/usage failed:", repr(e))

            if token and _InferenceApi is not None:
                try:
                    task = "text2text-generation" if "flan" in model or "t5" in model else "text-generation"
                    api = _InferenceApi(repo_id=model, token=token, task=task)
                    api_calls = [
                        lambda: api(prompt),
                        lambda: api(inputs=prompt),
                        lambda: api(prompt, {"max_new_tokens": 256}),
                        lambda: api(inputs=prompt, params={"max_new_tokens": 256}),
                    ]
                    resp = None
                    for call in api_calls:
                        try:
                            resp = call()
                            print("DEBUG: InferenceApi call succeeded for model", model)
                            break
                        except TypeError:
                            continue
                        except Exception as e:
                            print("DEBUG: InferenceApi call exception:", repr(e))
                            resp = None
                            break
                    out = self._normalize_hf(resp)
                    if out:
                        print("DEBUG: Normalized response from InferenceApi")
                        return out
                except Exception as e:
                    print("DEBUG: InferenceApi unavailable/failed:", repr(e))

            if token:
                try:
                    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
                    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}
                    urls = [
                        f"https://router.huggingface.co/models/{model}",
                        f"https://api-inference.huggingface.co/models/{model}",
                    ]
                    for url in urls:
                        try:
                            print("DEBUG: POST to", url)
                            r = requests.post(url, headers=headers, json=payload, timeout=30)
                            ct = r.headers.get("Content-Type", "")
                            print("DEBUG: status", r.status_code, "content-type", ct)
                            if r.status_code == 200 and "html" not in ct.lower():
                                try:
                                    resp = r.json()
                                except Exception:
                                    resp = r.text
                                out = self._normalize_hf(resp)
                                if out:
                                    print("DEBUG: Normalized response from HTTP POST to", url)
                                    return out
                        except Exception as e:
                            print("DEBUG: HTTP call to", url, "failed:", repr(e))
                except Exception as e:
                    print("DEBUG: Direct HTTP section failed:", repr(e))

        try:
            print("DEBUG: Falling back to local transformers pipeline")
            local_model = config.HF_LOCAL_MODEL
            from transformers import pipeline

            pipe = pipeline("text2text-generation", model=local_model, device=-1)
            out_list = pipe(prompt, max_new_tokens=256)
            if isinstance(out_list, list) and out_list:
                first = out_list[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                return str(first)
            return None
        except Exception as e:
            print("DEBUG: Local transformers fallback failed:", repr(e))

        print("DEBUG: No usable HF response from any model/provider")
        return None

    def iter_answer_tokens(self, question: str, context: str) -> Generator[str, None, None]:
        """Stream Groq tokens when possible; otherwise yield one chunk with the full answer."""
        if os.getenv("GROQ_API_KEY"):
            try:
                yielded = False
                for t in self._stream_groq(context, question):
                    yielded = True
                    yield t
                if yielded:
                    return
            except Exception:
                traceback.print_exc()

        groq_ans = self._answer_with_groq(context, question)
        if groq_ans:
            yield groq_ans
            return

        hf_ans = self._answer_with_hf(context, question)
        if hf_ans:
            yield hf_ans
            return

        info = self._extract_basic_info(context)
        parts = [
            "Note: both LLM providers (HuggingFace and Groq) are unavailable or returned no usable response."
        ]
        if info:
            parts.append("Extracted info:")
            for k, v in info.items():
                parts.append(f"- {k.capitalize()}: {v}")
        else:
            parts.append("No clear name/email/phone found.")
        doc_preview = (context[:1200] + "…") if context and len(context) > 1200 else context
        if doc_preview:
            parts.append("\nPart of your document (for reference):\n" + doc_preview)
        else:
            parts.append("\nNo document text available.")
        yield "\n".join(parts)

    def answer_question(self, question: str) -> str:
        scored = self.retrieve_with_scores(question)
        docs = self.dedupe_retrieved_documents([d for d, _ in scored])
        context = self.context_from_docs(docs)

        groq_ans = self._answer_with_groq(context, question)
        if groq_ans:
            return groq_ans

        hf_ans = self._answer_with_hf(context, question)
        if hf_ans:
            return hf_ans

        info = self._extract_basic_info(context or self.context_from_docs(docs))
        parts = [
            "Note: both LLM providers (HuggingFace and Groq) are unavailable or returned no usable response."
        ]
        if info:
            parts.append("Extracted info:")
            for k, v in info.items():
                parts.append(f"- {k.capitalize()}: {v}")
        else:
            parts.append("No clear name/email/phone found.")
        doc_preview = (context[:1200] + "…") if context and len(context) > 1200 else context
        if doc_preview:
            parts.append("\nPart of your document (for reference):\n" + doc_preview)
        else:
            parts.append("\nNo document text available.")
        return "\n".join(parts)

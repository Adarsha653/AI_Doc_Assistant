from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
import os
import re
import json
import traceback
import requests
from typing import Optional, List


class QASystem:
    def __init__(self, embeddings, vector_store_path: str = "vectorstore"):
        self.embeddings = embeddings
        self.vector_store_path = vector_store_path
        self.vector_store = None

    def build_vectorstore(self, documents: List):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Vector store saved to {self.vector_store_path}")

    def load_vectorstore(self):
        if not self.vector_store:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)

    def _extract_basic_info(self, text: str) -> dict:
        info = {}
        # email
        m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if m:
            info['email'] = m.group(0)
        # phone (simple)
        m = re.search(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
        if m:
            info['phone'] = m.group(0)
        # candidate name heuristic: "Name:" prefix or first few capitalized lines
        m = re.search(r'^(?:Name[:\s]*)([A-Z][A-Za-z\s]{1,80})', text, re.IGNORECASE | re.MULTILINE)
        if m:
            info['name'] = m.group(1).strip()
        else:
            first_lines = text.strip().splitlines()[:8]
            for line in first_lines:
                cand = line.strip()
                if cand and re.match(r'^[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,3}$', cand):
                    info.setdefault('name', cand)
                    break
        return info

    def _answer_with_groq(self, context: str, question: str) -> Optional[str]:
        try:
            from groq import Groq
            # read model from env so you can change without editing code
            model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
                model=model,
                max_tokens=1024,
                temperature=0.7,
            )
            # defensive extraction for varying response shapes
            try:
                return resp.choices[0].message.content
            except Exception:
                if hasattr(resp, "content"):
                    return str(resp.content)
                try:
                    return json.dumps(resp, default=str)
                except Exception:
                    return str(resp)
        except Exception as e:
            # print full traceback for debugging, then a short hint if model decommissioned
            traceback.print_exc()
            msg = str(e).lower()
            if "decommission" in msg or "decommissioned" in msg:
                print("DEBUG: Groq model appears decommissioned. Set GROQ_MODEL to a supported model (see https://console.groq.com/docs/deprecations).")
            return None

    def _answer_with_hf(self, context: str, question: str) -> Optional[str]:
        token = os.getenv("HUGGINGFACE_API_KEY")
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        def _normalize(resp) -> Optional[str]:
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

        model_candidates = [
            "google/flan-t5-large",
            "google/flan-t5-small",
            "facebook/bart-large-cnn",
            "gpt2"
        ]

        # Try hosted APIs first, then local transformers as final fallback
        for model in model_candidates:
            print(f"DEBUG: Trying model: {model}")

            # 1) InferenceClient (defensive)
            if token:
                try:
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=token)
                    call_attempts = [
                        lambda: client.text_generation(model=model, inputs=prompt, max_new_tokens=256),
                        lambda: client.text_generation(model=model, prompt=prompt, max_new_tokens=256),
                        lambda: client.text_generation(model, prompt, max_new_tokens=256),
                        lambda: client.text_generation(prompt, model=model, max_new_tokens=256),
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
                    out = _normalize(resp)
                    if out:
                        print("DEBUG: Normalized response from InferenceClient")
                        return out
                except Exception as e:
                    print("DEBUG: InferenceClient construction/usage failed:", repr(e))

            # 2) Legacy InferenceApi (some hf versions)
            if token:
                try:
                    from huggingface_hub import InferenceApi
                    task = "text2text-generation" if "flan" in model or "t5" in model else "text-generation"
                    api = InferenceApi(repo_id=model, token=token, task=task)
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
                    out = _normalize(resp)
                    if out:
                        print("DEBUG: Normalized response from InferenceApi")
                        return out
                except Exception as e:
                    print("DEBUG: InferenceApi unavailable/failed:", repr(e))

            # 3) Direct HTTP POST to HF router / api-inference
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
                                out = _normalize(resp)
                                if out:
                                    print("DEBUG: Normalized response from HTTP POST to", url)
                                    return out
                                else:
                                    print("DEBUG: HTTP returned JSON but normalization failed; resp type:", type(resp))
                        except Exception as e:
                            print("DEBUG: HTTP call to", url, "failed:", repr(e))
                except Exception as e:
                    print("DEBUG: Direct HTTP section failed:", repr(e))

        # 4) Local transformers fallback
        try:
            print("DEBUG: Falling back to local transformers pipeline")
            # prefer a small text2text model; change to t5-small or another small model if needed
            local_model = "google/flan-t5-small"
            from transformers import pipeline
            pipe = pipeline("text2text-generation", model=local_model, device=-1)  # device=-1 uses CPU
            out_list = pipe(prompt, max_new_tokens=256)
            # pipeline returns list of dicts
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

    def answer_question(self, question: str) -> str:
        if not self.vector_store:
            self.load_vectorstore()

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs]).strip()

        # Primary: Groq
        groq_ans = self._answer_with_groq(context, question)
        if groq_ans:
            return groq_ans

        # Secondary: Hugging Face Inference
        hf_ans = self._answer_with_hf(context, question)
        if hf_ans:
            return hf_ans
        
        # Final fallback: basic extraction + context excerpt
        info = self._extract_basic_info(context or ("\n".join([d.page_content for d in docs]) if docs else ""))
        parts = ["Note: both LLM providers (HuggingFace and Groq) are unavailable or returned no usable response."]
        if info:
            parts.append("Extracted info:")
            for k, v in info.items():
                parts.append(f"- {k.capitalize()}: {v}")
        else:
            parts.append("No clear name/email/phone found.")
        excerpt = (context[:1200] + "...") if context and len(context) > 1200 else context
        if excerpt:
            parts.append("\nContext excerpt:\n" + excerpt)
        else:
            parts.append("\nNo document text available.")
        return "\n".join(parts)
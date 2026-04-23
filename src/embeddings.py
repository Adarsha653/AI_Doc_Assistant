from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src import config


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str | None = None):
        name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

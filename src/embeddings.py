from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode(text).tolist()
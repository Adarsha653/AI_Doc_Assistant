from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.embeddings import LocalEmbeddings
from src.qa import QASystem


class ChatInterface:
    def __init__(self, vector_store_path: str | None = None):
        self.embeddings = LocalEmbeddings()
        self.qa_system = QASystem(self.embeddings, vector_store_path=vector_store_path or "vectorstore")

    def has_documents(self) -> bool:
        return self.qa_system.has_index()

    def list_indexed_file_names(self) -> List[str]:
        return self.qa_system.list_indexed_file_names()

    def process_query(self, user_query: str) -> str:
        return self.qa_system.answer_question(user_query)

    def retrieve_with_scores(
        self, user_query: str, session_file_names: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        return self.qa_system.retrieve_with_scores(user_query, session_file_names=session_file_names)

    def stream_answer_tokens(self, user_query: str, context: str):
        return self.qa_system.iter_answer_tokens(user_query, context)

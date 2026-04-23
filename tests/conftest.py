import os

# Avoid rare FAISS/OpenMP aborts when running the suite on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


@pytest.fixture
def fake_embeddings():
    from langchain_core.embeddings import Embeddings

    class _Fake(Embeddings):
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    return _Fake()

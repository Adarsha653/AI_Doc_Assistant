from pathlib import Path

from langchain_core.documents import Document

from src import config
from src.qa import (
    QASystem,
    _max_output_tokens_for_question,
    _user_content_for_qa,
    diversify_chunks_by_file,
    humanize_excerpt_preview_line,
    is_overview_style_query,
    resume_chunk_priority,
    strip_leaked_paths,
    wants_broader_retrieval,
    wants_multi_file_retrieval,
)


def test_strip_leaked_paths_removes_user_home() -> None:
    raw = "Pokhara Un@/Users/someone/Library/App/pdfs/Resume.pdf iversity Nepal"
    out = strip_leaked_paths(raw)
    assert "/Users/" not in out
    assert "Resume.pdf" not in out


def test_is_overview_style_query() -> None:
    assert is_overview_style_query("Tell me about this resume")
    assert is_overview_style_query("Give a brief overview of the document")
    assert not is_overview_style_query("What city is Monash in?")


def test_wants_broader_retrieval_includes_fit_questions() -> None:
    assert wants_broader_retrieval("Ok, do you think this resume is good for a data scientist role?")
    assert wants_broader_retrieval("Is this resume suitable for an ML engineer role?")
    assert wants_broader_retrieval("Tell me about this resume")
    assert wants_broader_retrieval("Who does this resume belong to?")
    assert not wants_broader_retrieval("What is the capital of Nepal?")


def test_max_output_tokens_identity_capped() -> None:
    assert _max_output_tokens_for_question("Who does this resume belong to?") == min(
        config.GROQ_MAX_TOKENS_IDENTITY, config.GROQ_MAX_TOKENS
    )
    assert _max_output_tokens_for_question("What city is Monash in?") == config.GROQ_MAX_TOKENS


def test_user_content_identity_has_constraints() -> None:
    c = _user_content_for_qa("### Source file: r.pdf\nJane Doe", "Who is this?")
    assert "Do not summarize" in c or "not summarize" in c
    assert "name" in c.lower()


def test_resume_chunk_priority_header_wins() -> None:
    header = Document(
        page_content="Jane Doe\n+1 234 linkedin.com/in/jane jane@gmail.com\nSummary Data scientist",
        metadata={},
    )
    mid = Document(page_content="(Pandas, NumPy) to analyze GMV and seller performance.", metadata={})
    assert resume_chunk_priority(header) > resume_chunk_priority(mid)


def test_humanize_excerpt_preview_line_strips_orphan_paren() -> None:
    raw = "(Pandas, NumPy) to analyze GMV. ● Built an hourly Databricks pipeline for cities."
    out = humanize_excerpt_preview_line(raw)
    assert not out.startswith("(")
    assert "Databricks" in out


def test_normalize_chunk_text_collapses_pdf_spacing() -> None:
    raw = "Hello    world\t\tfoo\n\n  bar   \n  baz  "
    assert QASystem.normalize_chunk_text(raw) == "Hello world foo\nbar\nbaz"


def test_normalize_chunk_text_strips_zero_width() -> None:
    raw = "A\u200bB\u200bC   D"
    assert QASystem.normalize_chunk_text(raw) == "ABC D"


def test_context_from_docs_labels_and_normalizes() -> None:
    docs = [
        Document(page_content="Line   one\n\n  two  ", metadata={"file_name": "A.txt"}),
        Document(page_content="  three  ", metadata={"file_name": "B.txt"}),
    ]
    ctx = QASystem.context_from_docs(docs)
    assert "### Source file: A.txt" in ctx
    assert "### Source file: B.txt" in ctx
    assert "Line one\ntwo" in ctx
    assert "three" in ctx


def test_wants_multi_file_retrieval() -> None:
    assert wants_multi_file_retrieval("How are these 3 files related?")
    assert wants_multi_file_retrieval("The 3 files I uploaded back to back")
    assert wants_multi_file_retrieval("Compare the two resumes I uploaded")
    assert not wants_multi_file_retrieval("How are you today?")


def test_diversify_chunks_by_file_interleaves_sources() -> None:
    scored = [
        (Document(page_content="a1", metadata={"file_name": "A.pdf"}), 0.01),
        (Document(page_content="a2", metadata={"file_name": "A.pdf"}), 0.02),
        (Document(page_content="b1", metadata={"file_name": "B.pdf"}), 0.015),
        (Document(page_content="c1", metadata={"file_name": "C.pdf"}), 0.018),
    ]
    out = diversify_chunks_by_file(scored, k=4)
    labels = [d.metadata["file_name"] for d, _ in out]
    assert set(labels[:3]) == {"A.pdf", "B.pdf", "C.pdf"}


def test_answer_question_mocked_retrieve(monkeypatch, fake_embeddings, tmp_path: Path) -> None:
    qa = QASystem(fake_embeddings, vector_store_path=str(tmp_path / "vs"))

    def fake_retrieve(q: str, session_file_names=None):
        return [(Document(page_content="The secret code is 4242."), 0.12)]

    monkeypatch.setattr(qa, "retrieve_with_scores", fake_retrieve)
    monkeypatch.setattr(QASystem, "_answer_with_groq", lambda self, c, q: "mocked groq")
    monkeypatch.setattr(QASystem, "_answer_with_hf", lambda self, c, q: None)

    out = qa.answer_question("What is the secret code?")
    assert out == "mocked groq"


def test_list_indexed_file_names(fake_embeddings, tmp_path: Path) -> None:
    qa = QASystem(fake_embeddings, vector_store_path=str(tmp_path / "vs_names"))
    qa.ingest_documents(
        [
            Document(page_content="alpha", metadata={"file_name": "A.pdf"}),
            Document(page_content="beta", metadata={"file_name": "B.pdf"}),
        ]
    )
    assert qa.list_indexed_file_names() == ["A.pdf", "B.pdf"]


def test_has_index(fake_embeddings, tmp_path: Path) -> None:
    qa = QASystem(fake_embeddings, vector_store_path=str(tmp_path / "empty"))
    assert not qa.has_index()
    qa.ingest_documents([Document(page_content="hello")])
    assert qa.has_index()

from pathlib import Path

import pytest

from src.document_loader import DocumentLoader


def test_load_txt_chunks(tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("word " * 500, encoding="utf-8")
    loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
    docs = loader.load_documents(str(p))
    assert len(docs) >= 2
    assert all(d.page_content for d in docs)


def test_display_name_preserved_on_chunks(tmp_path: Path) -> None:
    """Upload paths are often temp IDs; UI should use the user's original filename."""
    staged = tmp_path / "ba1705c7-staged.txt"
    staged.write_text(("paragraph " * 40).strip(), encoding="utf-8")
    loader = DocumentLoader(chunk_size=80, chunk_overlap=10)
    docs = loader.load_documents(str(staged), display_name="Resume_Adarsha_Aryal.txt")
    assert docs
    for d in docs:
        assert d.metadata.get("file_name") == "Resume_Adarsha_Aryal.txt"
        assert d.metadata.get("source") == str(staged)


def test_load_from_folder_two_txt(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha content here", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "b.txt").write_text("beta more text " * 30, encoding="utf-8")
    loader = DocumentLoader(chunk_size=80, chunk_overlap=5)
    docs = loader.load_from_folder(str(tmp_path))
    text = " ".join(d.page_content for d in docs)
    assert "alpha" in text
    assert "beta" in text

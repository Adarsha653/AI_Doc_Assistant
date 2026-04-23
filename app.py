from __future__ import annotations

import os
from typing import Any, List, Sequence

import chainlit as cl
from chainlit import Action
from chainlit.context import context
from dotenv import load_dotenv

from src.chat import ChatInterface
from src.document_loader import DocumentLoader
from src.qa import (
    QASystem,
    resume_chunk_priority,
    wants_broader_retrieval,
    wants_multi_file_retrieval,
)

load_dotenv()

# Chainlit file picker: MIME → extensions (loading still uses the file path suffix).
WELCOME_OVERVIEW = """## Welcome to AI Doc Assistant

This chat helps you **review and question your own documents**. After you upload files, the app:

1. **Reads** the text (PDFs, Word, Excel, PowerPoint, CSV, Markdown, HTML, plain text, and more).
2. **Builds a search index** so it can find the right parts of your documents for each question.
3. **Answers in plain language**, using your files as the main source—so you can summarize, compare, or dig into details without rereading everything yourself.

**Add more files:** After each assistant answer, a short line and **Add more documents** appear at the bottom so you can merge more uploads without scrolling back.

**Tip:** Files are indexed on the server running this app. When you ask a question, the app finds relevant text from your files and sends it with your question to your configured AI provider (e.g. Groq) to generate the reply—check that provider’s privacy policy if needed.

---
"""

UPLOAD_ACCEPT: dict[str, list[str]] = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".log", ".rst"],
    "text/markdown": [".md", ".markdown"],
    "text/csv": [".csv"],
    "text/tab-separated-values": [".tsv"],
    "text/html": [".html", ".htm"],
    "application/json": [".json"],
    "application/xml": [".xml"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/vnd.ms-excel": [".xls"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/msword": [".doc"],
}


def _vector_store_dir_for_session() -> str:
    sid = context.session.id
    path = os.path.join("vectorstore", str(sid))
    os.makedirs(path, exist_ok=True)
    return path


def _append_session_file_names(names: List[str]) -> None:
    """Track successful uploads so multi-file questions can balance retrieval per file."""
    cur: List[str] = list(cl.user_session.get("indexed_file_names") or [])
    for n in names:
        if n and n not in cur:
            cur.append(n)
    cl.user_session.set("indexed_file_names", cur)


def _rerank_scored_for_readability(question: str, scored: List[tuple]) -> List[tuple]:
    """Put resume header / summary / education-style chunks first for broad or fit questions."""
    if not scored or not wants_broader_retrieval(question):
        return scored
    return sorted(
        scored,
        key=lambda pair: (-resume_chunk_priority(pair[0]), float(pair[1])),
    )


async def _ingest_uploaded_files(chat: ChatInterface, files: Sequence[Any]) -> None:
    loader = DocumentLoader()
    all_docs = []
    errors: List[str] = []
    ingested_names: List[str] = []
    for file in files:
        await cl.Message(content=f"Processing **{file.name}**…").send()
        try:
            all_docs.extend(loader.load_documents(file.path, display_name=file.name))
            ingested_names.append(file.name)
        except Exception as e:
            errors.append(f"{file.name}: {e}")
    if errors:
        await cl.Message(content="❌ Some files failed:\n" + "\n".join(errors)).send()
    if not all_docs:
        await cl.Message(content="No text could be extracted. Try another file.").send()
        return
    chat.qa_system.ingest_documents(all_docs)
    _append_session_file_names(ingested_names)
    ok = len(ingested_names)
    await cl.Message(
        content=(
            f"Indexed **{ok}** file(s) → **{len(all_docs)}** chunk(s). "
            "Ask a question below; answers are based on your indexed files. "
            "Use **Add more documents** anytime to merge more files into this session."
        ),
        actions=_follow_up_actions(),
    ).send()


def _follow_up_actions() -> List[Action]:
    # Stable `id` so `public/hide-readme.css` can style this like the pink file-upload control.
    return [
        Action(
            name="add_documents",
            payload={},
            label="Add more documents",
            tooltip="Upload more files in any supported format; they merge into this chat’s index.",
            icon="upload",
            id="add-documents-cta",
        )
    ]


@cl.on_chat_start
async def start() -> None:
    vs_path = _vector_store_dir_for_session()
    chat = ChatInterface(vector_store_path=vs_path)
    cl.user_session.set("chat", chat)

    await cl.Message(content=WELCOME_OVERVIEW).send()

    files = await cl.AskFileMessage(
        content=(
            "### Upload documents to get started\n\n"
            "Add the files you want to **review or ask questions about**. "
            "Supported types include **PDF**, **Word (.docx)**, **Excel (.xlsx / .xls)**, "
            "**PowerPoint (.pptx)**, **CSV**, **Markdown**, **HTML**, **JSON**, **TXT**, and similar text-based formats.\n\n"
            "**Note:** Old **.doc** Word files are not supported here—please save as **.docx** or **PDF** first.\n\n"
            "You can attach **up to 10 files**, **20 MB** each."
        ),
        accept=UPLOAD_ACCEPT,
        max_size_mb=20,
        max_files=10,
    ).send()

    if not files:
        await cl.Message(
            content="No files uploaded yet. Use the button when you have a document ready.",
            actions=_follow_up_actions(),
        ).send()
        return

    await _ingest_uploaded_files(chat, files)


@cl.action_callback("add_documents")
async def on_add_documents(action: Action) -> None:
    """AskFileMessage ends with `task_start` in Chainlit's ask flow; action callbacks use
    `with_task=False`, so we must emit `task_end` or the composer stays on Stop. See chainlit#2122.
    """
    need_task_end = False
    try:
        chat = cl.user_session.get("chat")
        if not isinstance(chat, ChatInterface):
            await cl.Message(content="Session expired; refresh the page and start again.").send()
            return

        need_task_end = True
        # Cancel beside Browse is injected via `custom_js` (see public/ask-file-in-plane-cancel.js).
        files = await cl.AskFileMessage(
            content=(
                "### Add more documents\n\n"
                "Upload additional files in **any supported format** (they do not need to match your first upload). They are **merged** into your current index."
            ),
            accept=UPLOAD_ACCEPT,
            max_size_mb=20,
            max_files=10,
        ).send()

        if not files:
            await cl.Message(content="Upload cancelled.", actions=_follow_up_actions()).send()
            return

        await _ingest_uploaded_files(chat, files)
    finally:
        if need_task_end:
            await context.emitter.task_end()


@cl.on_message
async def main(message: cl.Message) -> None:
    chat = cl.user_session.get("chat")
    if not isinstance(chat, ChatInterface):
        await cl.Message(
            content="Chat is not initialized yet. Refresh the page and upload a document to begin."
        ).send()
        return

    if not chat.has_documents():
        await cl.Message(
            content=(
                "Upload at least one supported document before asking questions. "
                "Use **Add more documents** if you skipped the first upload."
            ),
            actions=_follow_up_actions(),
        ).send()
        return

    try:
        session_files = list(cl.user_session.get("indexed_file_names") or [])
        for n in chat.list_indexed_file_names():
            if n and n not in session_files:
                session_files.append(n)
        if session_files:
            cl.user_session.set("indexed_file_names", session_files)

        scored = chat.retrieve_with_scores(message.content, session_file_names=session_files)
        if not wants_multi_file_retrieval(message.content):
            scored = _rerank_scored_for_readability(message.content, scored)
        docs = QASystem.dedupe_retrieved_documents([d for d, _ in scored])
        context = QASystem.context_from_docs(docs)

        # Stream the answer, finalize the step, then attach upload actions on their own row
        # (actions on the streaming bubble are not reliably shown until end-of-stream in Chainlit).
        out = cl.Message(content="")
        await out.send()
        for token in chat.stream_answer_tokens(message.content, context):
            await out.stream_token(token)
        await out.update()
        await cl.Message(content="_Add more files:_", actions=_follow_up_actions()).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e!s}", actions=_follow_up_actions()).send()

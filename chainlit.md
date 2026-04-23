# AI Doc Assistant

When you start a chat, you’ll see a short **overview** of what this tool does, then you can **upload documents** to index and ask questions about them.

**Formats:** PDF, Word (.docx), Excel (.xlsx / .xls), PowerPoint (.pptx), CSV, Markdown, HTML, JSON, TXT, and more. Save legacy **.doc** as **.docx** or **PDF** first.

The app chunks the text, builds a **FAISS** index with local embeddings, then answers using **Groq** (if `GROQ_API_KEY` is set) with Hugging Face fallbacks.

- Use **Add more documents** anytime to merge more files into this chat’s index. After each assistant answer, that control appears again on its own short line at the bottom of the thread.
- Each answer **streams** after the app pulls relevant text from your files in the background (you do not see that raw text in the chat).

See `.env.example` for configuration variables.

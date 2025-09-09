import os
import re

from langchain.docstore.document import Document


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ; ", "; ").replace(" , ", ", ")
    return text


def build_prompt(
    question: str,
    docs: list[Document],
    *,
    per_doc_chars: int | None = None,
    max_ctx_chars: int | None = None,
) -> str:
    """Assemble a compact, cited prompt from the question and retrieved docs."""
    per_doc_chars = per_doc_chars or int(os.getenv("PER_DOC_CHARS", "600"))
    max_ctx_chars = max_ctx_chars or int(os.getenv("MAX_CTX_CHARS", "2400"))

    snippets = []
    total = 0
    for d in docs:
        row = d.metadata.get("row", "?")
        chunk = _clean(d.page_content)[:per_doc_chars]
        block = f"[row {row}] {chunk}"
        if total + len(block) > max_ctx_chars:
            break
        snippets.append(block)
        total += len(block)

    context = "\n\n".join(snippets)

    if not context:
        return (
            "You are a careful data assistant.\n"
            "No relevant context was retrieved. If you cannot answer, say so clearly.\n\n"
            f"Question: {question}\n\n"
            "Return a concise answer. If unknown, reply: 'Insufficient context.'"
        )

    system_rules = (
        "You are a precise data assistant. Answer ONLY using the provided context.\n"
        "If the context is insufficient, say 'Insufficient context.' Do not invent values.\n"
        "When you reference items, cite row numbers like [row 12].\n"
        "Prefer concise sentences and, when helpful, a short bullet list."
    )

    answer_format = (
        "Format:\n"
        "1) Answer: <your concise answer>\n"
        "2) Sources: comma-separated row numbers, e.g., 'Sources: row 3, row 9'\n"
    )

    prompt = f"{system_rules}\n\nQuestion: {question}\n\nContext:\n{context}\n\n{answer_format}"
    return prompt

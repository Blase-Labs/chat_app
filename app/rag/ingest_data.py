# ingest_data.py

import json
import os
import time
from pathlib import Path

import pandas as pd
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

INDEX_DIR = Path(os.getenv("INDEX_DIR", "/workspace/data/index"))
META_PATH = INDEX_DIR / "metadata.json"

# ----- helpers -----


def _ensure_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _embedder() -> OllamaEmbeddings:
    base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(base_url=base, model=model)


def _df_to_docs(df: pd.DataFrame, source: str) -> list[Document]:
    df = df.fillna("").astype(str)
    docs: list[Document] = []
    for i, row in df.iterrows():
        txt = "\n".join(f"{k}: {row[k]}" for k in df.columns)
        docs.append(Document(page_content=txt, metadata={"source": source, "row": int(i)}))
    return docs


def _write_metadata(*, source_name: str, rows: int) -> None:
    md = {
        "source_name": source_name,
        "rows": rows,
        "embedding_provider": "ollama",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "created_utc": int(time.time()),
    }
    META_PATH.write_text(json.dumps(md, indent=2))


def _have_saved_index() -> bool:
    return INDEX_DIR.exists() and any(INDEX_DIR.iterdir())


# ----- public -----


def vectorize_data(df: pd.DataFrame, source_name: str) -> tuple[FAISS, int]:
    """Embed CSV rows and build a FAISS index; persist it and return (index, row_count)."""
    _ensure_dirs()
    emb = _embedder()
    docs = _df_to_docs(df, source_name)
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(INDEX_DIR))
    _write_metadata(source_name=source_name, rows=len(docs))
    return vs, len(docs)


def load_index_if_exists() -> FAISS | None:
    if not _have_saved_index():
        return None
    emb = _embedder()
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    return vs

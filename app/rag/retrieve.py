from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

def retriever(vs: FAISS, question: str, k: int = 6) -> Tuple[List[Document], List[int]]: 
    if not question or not question.strip():
        return [], []
    
    mode = os.getenv("RETRIEVAL_MODE", "mmr").lower()
    if mode == "mmr":
        fetch_k = int(os.getenv("FETCH_K", "24"))
        lambda_mult = float(os.getenv("MMR_LAMBDA", "0.7"))
        docs = vs.max_marginal_relevance_search(
            question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    else:
        docs = vs.similarity_search(question, k=k)

    seen = set()
    pruned: List[Document] = []
    for d in docs:
        row = d.metadata.get("row")
        if row in seen:
            continue
        seen.add(row)
        content = d.page_content
        if len(content) > 800:
            d = Document(page_content=content[:800], metadata=d.metadata)
        pruned.append(d)
        if len(pruned) >= k:
            break

    row_ids = [int(d.metadata.get("row", -1)) for d in pruned]
    return pruned, row_ids
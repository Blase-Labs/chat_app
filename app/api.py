# api.py

# FastAPI backend entrypoint

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd

from .rag.ingest_data import load_index_if_exists, vectorize_data
from .rag.retrieve import retriever
from .rag.prompt import build_prompt
from .rag.respond import respond

index = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vs = load_index_if_exists()
    yield
    app.state.vs = None

app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    question: str

@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/ingest")
async def ingest(csv: UploadFile = File(...)):
    if not csv.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV")
    df = pd.read_csv(csv.file)
    if df.empty:
        raise HTTPException(400, "CSV is empty")
    vs, rows = vectorize_data(df, source_name=csv.filename)
    app.state.vs = vs
    return {"rows": rows, "indexed": True}

@app.post("/ask")
async def ask(req: AskRequest):
    vs = getattr(app.state, "vs", None)
    if vs is None:
        raise HTTPException(422, "Index not built. Upload a CSV and /ingest first.")
    docs, row_ids = retriever(vs, req.question, k=6)
    prompt = build_prompt(req.question, docs)
    answer = respond(prompt)
    return {"answer": answer, "sources": row_ids}

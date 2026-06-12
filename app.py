import os
from typing import List, Optional, Dict
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize

from openai import OpenAI

# ================= CONFIG =================
load_dotenv()

EMBED_MODEL_PATH = "models/vietnamese-sbert"

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STUDENT_ID = os.getenv("STUDENT_ID", "B22DCAT082")

TOP_K = 20
FINAL_K = 5

PARENT_SIZE = 2000
CHILD_SIZE = 500
CHILD_OVERLAP = 100

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# ================= MODEL =================
model = SentenceTransformer(EMBED_MODEL_PATH)

def embed(texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

EMBED_DIM = model.get_sentence_embedding_dimension()
print("[embed] dim:", EMBED_DIM)

# ================= FAISS =================
base_index = faiss.IndexFlatIP(EMBED_DIM)
index = faiss.IndexIDMap(base_index)

# ================= STORAGE =================
chunk_store: Dict[int, str] = {}
chunk_parent_map: Dict[int, int] = {}
parent_map: Dict[int, str] = {}

bm25 = None
bm25_chunks: List[str] = []

# ================= RESET =================
def reset_db():
    global index, base_index, chunk_store, chunk_parent_map, parent_map, bm25, bm25_chunks

    base_index = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIDMap(base_index)

    chunk_store = {}
    chunk_parent_map = {}
    parent_map = {}

    bm25 = None
    bm25_chunks = []

# ================= CHUNKING =================
def build_parents(text: str) -> List[str]:
    sents = sent_tokenize(text)

    parents = []
    cur = ""

    for s in sents:
        if len(cur) + len(s) < PARENT_SIZE:
            cur += " " + s
        else:
            parents.append(cur.strip())
            cur = s

    if cur:
        parents.append(cur.strip())

    return parents


def build_children(parent: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(parent):
        chunks.append(parent[start:start + CHILD_SIZE])
        start += (CHILD_SIZE - CHILD_OVERLAP)

    return chunks

# ================= ADD =================
def add_chunk(vec: np.ndarray, text: str, chunk_id: int, parent_id: int):
    index.add_with_ids(vec, np.array([chunk_id], dtype="int64"))

    chunk_store[chunk_id] = text
    chunk_parent_map[chunk_id] = parent_id

# ================= BM25 =================
def build_bm25():
    global bm25
    tokenized = [c.split() for c in bm25_chunks]
    bm25 = BM25Okapi(tokenized)

# ================= INDEX BUILD =================
def build_index(text: str):
    reset_db()

    parents = build_parents(text)

    global bm25_chunks

    for pid, p in enumerate(parents):
        parent_map[pid] = p

        children = build_children(p)

        for c in children:
            vec = embed([c])[0]

            cid = len(chunk_store)
            add_chunk(vec, c, cid, pid)

            bm25_chunks.append(c)

    build_bm25()

# ================= SEARCH =================
def dense_search(query: str):
    vec = embed([query])
    scores, ids = index.search(vec, TOP_K)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        results.append((int(i), float(s)))
    return results


def bm25_search(query: str):
    scores = bm25.get_scores(query.split())

    results = []
    for i, s in enumerate(scores):
        results.append((i, float(s)))

    return sorted(results, key=lambda x: x[1], reverse=True)[:TOP_K]


def hybrid_search(question: str):
    score_map = {}

    # dense
    for idx, score in dense_search(question):
        score_map[idx] = score_map.get(idx, 0) + DENSE_WEIGHT * score

    # bm25
    for idx, score in bm25_search(question):
        score_map[idx] = score_map.get(idx, 0) + BM25_WEIGHT * score

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

# ================= RETRIEVAL =================
def retrieve(question: str):
    ranked = hybrid_search(question)

    parent_scores = {}

    for idx, score in ranked:
        pid = chunk_parent_map.get(idx)
        if pid is None:
            continue

        # max pooling (better than sum)
        parent_scores[pid] = max(parent_scores.get(pid, 0), score)

    top_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_K]

    return [parent_map[pid] for pid, _ in top_parents]

# ================= LLM =================
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=STUDENT_ID
)

def clean_answer(text: str) -> str:
    text = text.strip().upper()
    for c in ["A", "B", "C", "D"]:
        if c in text:
            return c
    return "A"


def ask_llm(question: str, context: str):
    system = "Bạn là trợ lý trắc nghiệm. Chỉ trả lời A, B, C hoặc D. Không giải thích."

    user = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=5,
        temperature=0
    )

    return clean_answer(res.choices[0].message.content)

# ================= FASTAPI =================
app = FastAPI()

class UploadRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None


class AskRequest(BaseModel):
    question: str


@app.post("/upload")
def upload(req: UploadRequest):
    build_index(req.text)

    return {
        "status": "ok",
        "chunks": len(chunk_store)
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not chunk_store:
        raise HTTPException(400, "No data uploaded")

    context = retrieve(req.question)
    context_text = "\n\n---\n\n".join(context)

    answer = ask_llm(req.question, context_text)

    return {"answer": answer}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(chunk_store),
        "model": EMBED_MODEL_PATH
    }
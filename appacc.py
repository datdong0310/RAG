import os
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
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
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35

# ================= MODELS =================
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=STUDENT_ID
)

def embed(texts: List[str]) -> np.ndarray:
    vecs = embed_model.encode(texts, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# ================= FAISS =================
base_index = faiss.IndexFlatIP(EMBED_DIM)
index = faiss.IndexIDMap(base_index)

# ================= STORAGE =================
chunk_store: Dict[int, str] = {}
chunk_parent_map: Dict[int, int] = {}
parent_store: Dict[int, str] = {}

bm25 = None
bm25_chunks: List[str] = []

# ================= RESET =================
def reset_db():
    global index, base_index, chunk_store, chunk_parent_map, parent_store, bm25, bm25_chunks

    base_index = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIDMap(base_index)

    chunk_store = {}
    chunk_parent_map = {}
    parent_store = {}

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


def build_chunks(parent: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(parent):
        chunks.append(parent[start:start + CHUNK_SIZE])
        start += (CHUNK_SIZE - CHUNK_OVERLAP)

    return chunks

# ================= INDEX =================
def add_chunk(chunk_id: int, text: str, parent_id: int):
    vec = embed([text])[0]
    index.add_with_ids(np.array([vec]), np.array([chunk_id], dtype="int64"))

    chunk_store[chunk_id] = text
    chunk_parent_map[chunk_id] = parent_id


def build_bm25():
    global bm25
    tokenized = [c.split() for c in bm25_chunks]
    bm25 = BM25Okapi(tokenized)

# ================= QUERY REWRITE =================
def rewrite_query(q: str) -> List[str]:
    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the question into 3 short search queries in Vietnamese. Return one per line."
                },
                {"role": "user", "content": q}
            ],
            temperature=0
        )

        out = res.choices[0].message.content
        queries = [x.strip() for x in out.split("\n") if x.strip()]
        return queries[:3]

    except:
        return [q]

# ================= HYBRID RETRIEVAL =================
def dense_search(q: str):
    vec = embed([q])
    scores, ids = index.search(vec, TOP_K)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        results.append((int(i), float(s)))
    return results


def bm25_search(q: str):
    scores = bm25.get_scores(q.split())
    return sorted([(i, float(s)) for i, s in enumerate(scores)],
                  key=lambda x: x[1], reverse=True)[:TOP_K]


def hybrid_search(question: str):
    queries = rewrite_query(question)

    score_map = {}

    for q in queries:
        for idx, s in dense_search(q):
            score_map[idx] = score_map.get(idx, 0) + DENSE_WEIGHT * s

        for idx, s in bm25_search(q):
            score_map[idx] = score_map.get(idx, 0) + BM25_WEIGHT * s

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

# ================= RERANK =================
def rerank(question: str, chunks: List[str]) -> List[str]:
    pairs = [(question, c) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:FINAL_K]]

# ================= RETRIEVAL =================
def retrieve(question: str) -> List[str]:
    ranked = hybrid_search(question)

    candidates = []
    for idx, _ in ranked:
        candidates.append(chunk_store[idx])

    if not candidates:
        return []

    return rerank(question, candidates)

# ================= LLM ANSWER =================
def clean_answer(text: str) -> str:
    text = text.strip().upper()
    for c in ["A", "B", "C", "D"]:
        if c in text:
            return c
    return "A"


def ask_llm(question: str, context: str):
    system = (
        "Bạn là trợ lý trắc nghiệm. "
        "Chỉ trả lời A, B, C hoặc D. Không giải thích."
    )

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
        temperature=0,
        max_tokens=5
    )

    return clean_answer(res.choices[0].message.content)

# ================= FASTAPI =================
app = FastAPI()

class UploadRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None


class AskRequest(BaseModel):
    question: str

# ================= BUILD INDEX =================
def build_index(text: str):
    reset_db()

    parents = build_parents(text)

    global bm25_chunks

    chunk_id = 0

    for pid, parent in enumerate(parents):
        parent_store[pid] = parent

        chunks = build_chunks(parent)

        for c in chunks:
            add_chunk(chunk_id, c, pid)
            bm25_chunks.append(c)
            chunk_id += 1

    build_bm25()

# ================= API =================
@app.post("/upload")
def upload(req: UploadRequest):
    build_index(req.text)
    return {"status": "ok", "chunks": len(chunk_store)}


@app.post("/ask")
def ask(req: AskRequest):
    if not chunk_store:
        raise HTTPException(400, "No data uploaded")

    context_chunks = retrieve(req.question)
    context = "\n\n---\n\n".join(context_chunks)

    answer = ask_llm(req.question, context)

    return {"answer": answer}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(chunk_store),
        "model": EMBED_MODEL_PATH
    }
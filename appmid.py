import os
import re
from typing import List, Optional, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize

from openai import OpenAI
from functools import lru_cache
from collections import defaultdict

# ================= CONFIG =================
load_dotenv()

EMBED_MODEL_PATH = "models/vietnamese-sbert"

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STUDENT_ID = os.getenv("STUDENT_ID", "B22DCAT082")

TOP_K = 10
CANDIDATE_K = 12
FINAL_K = 5

PARENT_SIZE = 2000
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# ================= MODELS =================
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=STUDENT_ID
)

EMBED_DIM = embed_model.get_sentence_embedding_dimension()

# ================= FAISS (HNSW faster) =================
index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
index.hnsw.efSearch = 32
index = faiss.IndexIDMap(index)

# ================= STORAGE =================
chunk_store: Dict[int, str] = {}
chunk_parent_map: Dict[int, int] = {}
parent_store: Dict[int, str] = {}

bm25 = None
bm25_chunks: List[str] = []
bm25_chunk_ids: List[int] = []

# ================= RESET =================
def reset_db():
    global index, chunk_store, chunk_parent_map, parent_store, bm25, bm25_chunks, bm25_chunk_ids

    base = faiss.IndexHNSWFlat(EMBED_DIM, 32)
    base.hnsw.efSearch = 32
    index = faiss.IndexIDMap(base)

    chunk_store = {}
    chunk_parent_map = {}
    parent_store = {}

    bm25 = None
    bm25_chunks = []
    bm25_chunk_ids = []

# ================= EMBEDDING =================
def embed(texts: List[str]) -> np.ndarray:
    vecs = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vecs.astype("float32")


# cached query embedding (BIG SPEED WIN)
@lru_cache(maxsize=2048)
def cached_embed(text: str):
    vec = embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    return vec.copy()


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


# ================= INDEXING (BATCHED) =================
def add_chunks_batch(chunks: List[str], parent_id: int, start_id: int):
    vecs = embed(chunks)
    ids = np.arange(start_id, start_id + len(chunks), dtype="int64")

    index.add_with_ids(vecs, ids)

    for i, c in zip(ids, chunks):
        chunk_store[int(i)] = c
        chunk_parent_map[int(i)] = parent_id

    return start_id + len(chunks)


def build_bm25():
    global bm25
    tokenized = [c.split() for c in bm25_chunks]
    bm25 = BM25Okapi(tokenized)


# ================= QUERY (SIMPLIFIED FOR SPEED) =================
def rewrite_query(q: str) -> List[str]:
    # removed LLM call → big latency reduction
    return [q]


# ================= RETRIEVAL =================
def dense_search(q: str):
    vec = cached_embed(q)
    scores, ids = index.search(vec, TOP_K)

    return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


def bm25_search(q: str):
    if bm25 is None:
        return []
    scores = bm25.get_scores(q.split())
    # Map list indices back to chunk IDs so they match dense_search IDs
    id_score_pairs = [(bm25_chunk_ids[i], s) for i, s in enumerate(scores)]
    return sorted(id_score_pairs, key=lambda x: x[1], reverse=True)[:TOP_K]


def hybrid_search(question: str):
    queries = rewrite_query(question)

    score_map = defaultdict(float)

    for q in queries:
        for idx, s in dense_search(q):
            score_map[idx] += DENSE_WEIGHT * s

        for idx, s in bm25_search(q):
            score_map[idx] += BM25_WEIGHT * s

    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:TOP_K]


# ================= RERANK =================
def rerank(question: str, chunks: List[str]) -> List[str]:
    chunks = chunks[:CANDIDATE_K]  # IMPORTANT SPEED CUT

    pairs = [(question, c) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:FINAL_K]]


# ================= RETRIEVAL =================
def retrieve(question: str) -> List[str]:
    ranked = hybrid_search(question)

    candidates = [chunk_store[idx] for idx, _ in ranked if idx in chunk_store]

    if not candidates:
        return []

    reranked = rerank(question, candidates)

    # Expand to parent documents for richer context
    seen_parents = set()
    parent_docs = []
    for chunk in reranked:
        # Find the chunk's parent
        for cid, text in chunk_store.items():
            if text == chunk and cid in chunk_parent_map:
                pid = chunk_parent_map[cid]
                if pid not in seen_parents:
                    seen_parents.add(pid)
                    parent_docs.append(parent_store[pid])
                break

    return parent_docs if parent_docs else reranked


# ================= LLM =================
def clean_answer(text: str) -> str:
    text = text.strip().upper()
    # Match a standalone answer letter (at word boundary or alone)
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    # Fallback: check if text starts with a valid letter
    if text and text[0] in "ABCD":
        return text[0]
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

    global bm25_chunks, bm25_chunk_ids
    bm25_chunks = []
    bm25_chunk_ids = []

    chunk_id = 0

    for pid, parent in enumerate(parents):
        parent_store[pid] = parent

        chunks = build_chunks(parent)

        # Track chunk IDs for BM25 index alignment
        for i in range(len(chunks)):
            bm25_chunk_ids.append(chunk_id + i)

        chunk_id = add_chunks_batch(chunks, pid, chunk_id)

        bm25_chunks.extend(chunks)

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
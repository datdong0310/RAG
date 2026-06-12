import os, re, uuid
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize

from openai import OpenAI
from functools import lru_cache
from collections import defaultdict

# ================= CONFIG =================
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "keepitreal/vietnamese-sbert")
MODEL_CACHE_DIR  = os.getenv("MODEL_CACHE_DIR", os.path.join(os.path.dirname(__file__), "models"))

# Proxy LLM theo slide: dùng MSSV làm API key
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.50.218:8000/api/v1/proxy")
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")
STUDENT_ID   = os.getenv("STUDENT_ID", "B22DCAT082")  # MSSV — dùng làm API key

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 32))
TOP_K         = int(os.getenv("TOP_K", 5))

# Khi chạy offline (trong LAN thi), buộc transformers/HF KHÔNG gọi mạng.
# Model phải đã được tải sẵn vào MODEL_CACHE_DIR (chạy download_model.py trước).
if os.getenv("OFFLINE", "0") == "1":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

# ── Embedding setup ───────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
_st_model = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=MODEL_CACHE_DIR)



def get_embeddings(texts: List[str]) -> List[List[float]]:
    return _st_model.encode(texts, convert_to_numpy=True).tolist()

EMBED_DIM = _st_model.get_sentence_embedding_dimension()
print(f"[embed] local — {EMBED_MODEL_NAME}  dim={EMBED_DIM}")

# ── Vector store (FAISS in-memory) ───────────────────────────────────────────
import faiss, numpy as np

_index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))
_store: dict[str, str] = {}   # chunk_id -> text
_id_map: dict[int, str] = {}
next_id = 0


def db_upsert(chunk_id: str, vector: List[float], text: str):
    global next_id   # ✅ REQUIRED

    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)

    faiss_id = next_id
    next_id += 1

    _id_map[faiss_id] = chunk_id
    _store[chunk_id] = text

    _index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))

def db_search(vector: List[float], k: int):
    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)

    scores, idxs = _index.search(vec, min(k, _index.ntotal))

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        chunk_id = _id_map.get(int(i))
        if chunk_id:
            results.append({
                "chunk_id": chunk_id,
                "text": _store[chunk_id],
                "score": float(score)
            })

    return results

def db_count() -> int:
    return _index.ntotal

def db_reset():
    global _index, _store, _id_map, _bm25, _corpus_tokens, _corpus_chunks, next_id

    _index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))
    _store = {}
    _id_map = {}

    _bm25 = None
    _corpus_tokens = []
    _corpus_chunks = []

    next_id = 0   # ✅ correct reset
    
print("[db] FAISS in-memory")

from rank_bm25 import BM25Okapi

_bm25 = None
_corpus_tokens = []
_corpus_chunks = []

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Competition — Student Server")

# ── Schemas (đúng slide) ──────────────────────────────────────────────────────
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []

# ── Helpers ───────────────────────────────────────────────────────────────────
r"""def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter for Vietnamese + English
    # Handles ., ?, ! and line breaks
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]
    
   def chunk_text(text: str) -> List[str]:
    sentences = split_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0

    step_overlap_sentences = max(1, CHUNK_OVERLAP // 50)  # rough heuristic

    for sent in sentences:
        sent_len = len(sent)

        # if adding sentence exceeds limit → flush chunk
        if current_length + sent_len > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))

            # keep overlap
            current_chunk = current_chunk[-step_overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
    """
def chunk_text(text: str) -> List[str]:
    chunks, start = [], 0
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += step
    return chunks

_VALID_LETTERS = {"A", "B", "C", "D"}

def normalize(scores: dict):
    if not scores:
        return scores
    min_v = min(scores.values())
    max_v = max(scores.values())
    if max_v == min_v:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

def hybrid_retrieve(query: str, q_vec, top_k=10):
    if _bm25 is None:
        hits = db_search(q_vec, top_k)
        return [{"text": h["text"], "chunk_id": h["chunk_id"]} for h in hits]

    faiss_hits = db_search(q_vec, top_k * 2)
    bm25_scores = _bm25.get_scores(tokenize(query))

    faiss_scores = {
        h["chunk_id"]: h["score"]
        for h in faiss_hits
    }

    bm25_scores_dict = {
        _id_map[i]: float(bm25_scores[i])
        for i in range(len(bm25_scores))
        if i in _id_map
    }

    faiss_scores = normalize(faiss_scores)
    bm25_scores_dict = normalize(bm25_scores_dict)

    merged = {}

    for k, v in faiss_scores.items():
        merged[k] = merged.get(k, 0) + 0.6 * v

    for k, v in bm25_scores_dict.items():
        merged[k] = merged.get(k, 0) + 0.4 * v

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    return [
        {"chunk_id": k, "text": _store[k]}
        for k, _ in ranked[:top_k]
    ]


def extract_letter(raw: str) -> str:
    """LLM có thể trả 'A', 'A.', 'Đáp án: B', '**C**'... → chuẩn hoá về 1 ký tự."""
    if not raw:
        return "A"
    s = raw.strip().upper()

    # Nếu chuỗi chỉ có 1 ký tự và là A, B, C, D
    if len(s) == 1 and s in _VALID_LETTERS:
        return s

    # Tìm pattern rõ ràng: 'ĐÁP ÁN: X' hoặc 'ANSWER: X' hoặc 'CHỌN X'
    m = re.search(r"(?:ĐÁP\s*ÁN|ANSWER|CHỌN)\s*(?:LÀ\s*)?[:\-]?\s*(?:[*_]*)([ABCD])", s)
    if m:
        return m.group(1)

    # Nếu bắt đầu bằng A, B, C, D theo sau là dấu phân cách (ví dụ "A.", "B)", "C -")
    if len(s) >= 2 and s[0] in _VALID_LETTERS and not s[1].isalnum():
        return s[0]

    # Tìm chữ A, B, C, D đứng độc lập (không dính liền chữ/số khác)
    m = re.search(r"(?<!\w)([ABCD])(?!\w)", s)
    if m:
        return m.group(1)

    return "A"  # fallback an toàn

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed": EMBED_MODEL_NAME,
        "db": "faiss-memory",
        "indexed": db_count(),
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "student_id": STUDENT_ID,
    }

@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    # Reset DB mỗi lần nhận document mới để tránh nhiễu giữa các lượt thi
    db_reset()

    doc_id = req.doc_id if req.doc_id and req.doc_id != "none" else str(uuid.uuid4())[:8]
    chunks = chunk_text(req.text)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Empty document.")
    
    global _bm25, _corpus_tokens, _corpus_chunks

    _corpus_chunks.extend(chunks)
    _corpus_tokens.extend([tokenize(c) for c in chunks])
    _bm25 = BM25Okapi(_corpus_tokens)

    vectors = get_embeddings(chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        db_upsert(f"{doc_id}_chunk_{i}", vec, chunk)

    return UploadResponse(status="success", doc_id=doc_id, chunks=len(chunks))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if db_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    q_vec = get_embeddings([req.question])[0]

    candidates = hybrid_retrieve(req.question, q_vec, top_k=5)

    context = "\n\n---\n\n".join(
        c["text"] for c in candidates
    )

    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=STUDENT_ID)

    system_prompt = (
        "Bạn là trợ lý trả lời trắc nghiệm. "
        "Dựa CHỈ vào tài liệu được cung cấp để chọn đáp án đúng. "
        "BẮT BUỘC chỉ trả lời bằng MỘT ký tự duy nhất: A, B, C hoặc D. "
        "Không giải thích, không viết gì thêm."
    )

    user_prompt = f"""Tài liệu tham khảo:
{context}

Câu hỏi trắc nghiệm:
{req.question}

Đáp án (chỉ 1 ký tự A/B/C/D):"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    answer = extract_letter(raw)

    return AskResponse(
        answer=answer,
        sources=[c["chunk_id"] for c in candidates]
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

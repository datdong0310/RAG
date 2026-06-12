import os, re, uuid
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from typing import List


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
_id_list: list[str] = []

def db_upsert(chunk_id: str, vector: List[float], text: str):
    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)

    faiss_id = len(_id_list)  # or hash-based ID
    _id_list.append(chunk_id)
    _store[chunk_id] = text

    _index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))

def db_search(vector: List[float], k: int) -> List[dict]:
    if _index.ntotal == 0:
        return []

    vec = np.array([vector], dtype="float32")
    faiss.normalize_L2(vec)  # IMPORTANT

    _, idxs = _index.search(vec, min(k, _index.ntotal))

    return [
        {"chunk_id": _id_list[i], "text": _store[_id_list[i]]}
        for i in idxs[0] if i < len(_id_list)
    ]

def db_count() -> int:
    return _index.ntotal

def db_reset():
    global _index, _store, _id_list
    _index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))
    _store = {}
    _id_list = []
    
print("[db] FAISS in-memory")

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
"""def split_sentences(text: str) -> List[str]:
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

def extract_letter(raw: str) -> str:
    """LLM có thể trả 'A', 'A.', 'Đáp án: B', '**C**'... → chuẩn hoá về 1 ký tự."""
    if not raw:
        return "A"
    s = raw.strip().upper()

    # ký tự đầu nếu hợp lệ
    if s[0] in _VALID_LETTERS:
        return s[0]

    # tìm pattern 'ĐÁP ÁN: X' hoặc 'ANSWER: X'
    m = re.search(r"(?:ĐÁP\s*ÁN|ANSWER|CHỌN)\s*[:\-]?\s*([ABCD])", s)
    if m:
        return m.group(1)

    # tìm ký tự A/B/C/D đầu tiên bị bao bởi non-letter
    m = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", s)
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

    vectors = get_embeddings(chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        db_upsert(f"{doc_id}_chunk_{i}", vec, chunk)

    return UploadResponse(status="success", doc_id=doc_id, chunks=len(chunks))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if db_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    q_vec = get_embeddings([req.question])[0]
    hits = db_search(q_vec, TOP_K)

    context_chunks = [h["text"] for h in hits]
    context = "\n\n---\n\n".join(context_chunks)

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
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=4,
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    answer = extract_letter(raw)

    return AskResponse(answer=answer, sources = [h["chunk_id"] for h in hits])




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

import os
from dotenv import load_dotenv

load_dotenv()

# ─── APIs ────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# ─── ChromaDB ────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ─── Chunking ────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ─── LLM ─────────────────────────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"

# ─── Embeddings ──────────────────────────────────────────
# all-MiniLM-L6-v2: small (90MB), fast, very efficient
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─── Reranking ───────────────────────────────────────────
# cross-encoder/ms-marco-MiniLM-L-6-v2: lightweight, free
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-large-v2"

# ─── Retrieval ───────────────────────────────────────────
TOP_K_RETRIEVAL = 10   # Number of chunks retrieved by Hybrid Search
TOP_K_RERANKED = 5     # Number of chunks kept after reranking
HYBRID_ALPHA = 0.5     # 0 = pure BM25, 1 = pure dense, 0.5 = balanced

# ─── Multi-query ─────────────────────────────────────────
MULTI_QUERY_COUNT = 3  # Number of query reformulations

# ─── Supported file types ────────────────────────────────
SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "md",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.google-apps.document": "gdoc",  # Native Google Docs
}

# ─── LangSmith Tracing ───────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "DriveRAG")
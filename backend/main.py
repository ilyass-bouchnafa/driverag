"""
DriveRAG — FastAPI Backend
Wraps existing src/ modules, exposes REST API for the React frontend.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import uuid
import time
from pathlib import Path

# Force ffmpeg - cherche dans tous les endroits possibles
ffmpeg_path = r"C:\ffmpeg-8.1\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    
# Point Python at the project root so src.* imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.llm_chain import ask, clear_memory
from src.generation.llm_direct import ask_direct
from src.retrieval.vectorstore import get_indexed_files, add_chunks_to_store
from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.ingestion.sync_manager import start_auto_sync, get_auto_sync_status, smart_sync
from src.config import GOOGLE_DRIVE_FOLDER_ID

app = FastAPI(title="DriveRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"
    history: List[Message] = []
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None

class SyncResponse(BaseModel):
    synced: int
    errors: int
    files: List[str]


@app.on_event("startup")
def startup_event():
    """FastAPI startup event: start background auto-sync.

    This hooks into application startup to kick off the periodic
    synchronization with Google Drive (non-blocking).
    """

    start_auto_sync(interval_seconds=1800)


@app.get("/health")
def health():
    """Health check endpoint.

    Returns basic status information including sync state, number of
    indexed files and Redis statistics to help monitoring the service.
    """

    from src.retrieval.redis_corpus import get_redis_stats
    return {
        "status": "ok",
        "sync": get_auto_sync_status(),
        "indexed_files": len(get_indexed_files()),
        "redis": get_redis_stats(),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Handle chat requests from the frontend.

    Supports two modes:
        - "rag": Retrieval-Augmented Generation using indexed documents
        - "direct": Direct LLM mode (no document grounding)

    The request is converted to the expected internal format and dispatched
    to the appropriate handler in `src.generation`.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    conversation_id = req.conversation_id or f"conv_{int(time.time())}"
    try:
        if req.mode == "rag":
            from langchain.schema import HumanMessage, AIMessage
            lc_history = []
            for m in req.history:
                if m.role == "user":
                    lc_history.append(HumanMessage(content=m.content))
                else:
                    lc_history.append(AIMessage(content=m.content))
            result = ask(req.message, external_history=lc_history,
                        thread_id=thread_id, conversation_id=conversation_id)
        else:
            history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
            result = ask_direct(req.message, history=history_dicts,
                               thread_id=thread_id, conversation_id=conversation_id)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "mode": req.mode,
            "thread_id": thread_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def get_files():
    """Return the list of indexed files available in the vector store."""

    return get_indexed_files()


@app.post("/sync", response_model=SyncResponse)
async def sync_drive():
    """Trigger a manual sync with Google Drive and return sync stats.

    Calls the same logic as the automatic sync but exposes it via API.
    """

    try:
        files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drive error: {e}")

    stats = smart_sync()
    names = [f["name"] for f in files]
    return SyncResponse(
        synced=stats.get("new", 0) + stats.get("updated", 0),
        errors=len(stats.get("errors", [])),
        files=names,
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file from the frontend and ingest it into the corpus.

    Validates the file extension, uploads the file to Drive and ingests
    the resulting pages/chunks into the vector store.
    """

    try:
        from src.ingestion.drive_uploader import upload_and_ingest
        file_bytes = await file.read()
        file_name = file.filename
        file_extension = Path(file_name).suffix.lower()
        ALLOWED = {".pdf", ".docx", ".txt", ".md", ".pptx"}
        if file_extension not in ALLOWED:
            raise HTTPException(status_code=400,
                detail=f"Unsupported format: {file_extension}. Accepted: {', '.join(ALLOWED)}")
        result = upload_and_ingest(file_bytes, file_name, file_extension)
        if result.get("error") and not result.get("upload"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {
            "success": True,
            "file_name": file_name,
            "drive_id": result.get("upload", {}).get("id"),
            "drive_link": result.get("upload", {}).get("webViewLink"),
            "chunks": result.get("chunks", 0),
            "warning": result.get("error"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
def clear_conversation():
    """Clear the in-memory conversation history (development utility)."""

    try:
        clear_memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive an audio file, run Whisper transcription, and return text."""

    import whisper, traceback
    data = await audio.read()
    print(f"Received audio: {len(data)} bytes")
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time())}.webm")
    with open(tmp_path, "wb") as f:
        f.write(data)
    print(f"Saved to: {tmp_path}, exists: {os.path.exists(tmp_path)}")
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Transcribing...")
        result = model.transcribe(tmp_path, fp16=False, language="fr")
        print(f"Result: {result['text']}")
        return {"text": result["text"].strip()}
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

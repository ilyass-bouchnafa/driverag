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
os.environ["PATH"] = r"C:\ffmpeg-8.1\bin" + os.pathsep + os.environ.get("PATH", "")

# Point Python at the project root so src.* imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.llm_chain import ask, clear_memory
from src.generation.llm_direct import ask_direct
from src.retrieval.vectorstore import get_indexed_files, add_chunks_to_store
from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.ingestion.sync_manager import start_auto_sync, get_auto_sync_status
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
    start_auto_sync(interval_seconds=300)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sync": get_auto_sync_status(),
        "indexed_files": len(get_indexed_files()),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
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
    return get_indexed_files()


@app.post("/sync", response_model=SyncResponse)
async def sync_drive():
    try:
        files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drive error: {e}")
    synced, errors, names = 0, 0, []
    for f in files:
        try:
            file_bytes = download_file(f["id"], f["name"], f["mimeType"])
            pages = extract_text_from_bytes(file_bytes, f["name"], f["format"])
            for page in pages:
                page["drive_path"] = f["path"]
                page["file_format"] = f["format"]
                page["drive_modified_time"] = f.get("modifiedTime", "")
            chunks = chunk_pages(pages)
            add_chunks_to_store(chunks)
            synced += 1
            names.append(f["name"])
        except Exception:
            errors += 1
    return SyncResponse(synced=synced, errors=errors, files=names)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        from src.ingestion.drive_uploader import upload_and_ingest
        file_bytes = await file.read()
        file_name = file.filename
        file_extension = Path(file_name).suffix.lower()
        ALLOWED = {".pdf", ".docx", ".txt", ".md", ".pptx"}
        if file_extension not in ALLOWED:
            raise HTTPException(status_code=400,
                detail=f"Format non supporté : {file_extension}. Acceptés : {', '.join(ALLOWED)}")
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
    try:
        clear_memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    import whisper, traceback
    data = await audio.read()
    print(f"Received audio: {len(data)} bytes")
    tmp_path = os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), f"audio_{int(time.time())}.webm")
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

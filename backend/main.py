"""
backend/main.py

Corrections appliquees par rapport a l'original :
  - /chat utilise desormais reellement thread_id pour scoper l'historique
    (avant: thread_id etait genere/recu mais jamais utilise pour le
    stockage, qui passait par une liste globale unique).
  - /transcribe utilise le service Whisper corrige (modele en cache,
    langue auto-detectee avec biais FR/EN en cas d'ambiguite -- voir
    src/transcription/whisper_service.py).
  - /clear prend desormais un thread_id et n'efface QUE ce thread.
  - Les stats BM25 globales (vocab/df/n_docs) sont chargees une fois au
    demarrage et reutilisees par toutes les requetes /chat, rafraichies
    uniquement apres un /sync (voir sync_manager.get_current_bm25_stats).
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

ffmpeg_path = r"C:\ffmpeg-8.1\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generation.llm_chain import ask_async, clear_memory
from src.generation.llm_direct import ask_direct
from src.retrieval.qdrant_store import get_indexed_files
from src.ingestion.gdrive_loader import list_files_recursive
from src.ingestion.sync_manager import (
    start_auto_sync,
    get_auto_sync_status,
    smart_sync,
    get_current_bm25_stats,
)
from src.transcription.whisper_service import transcribe_audio
from src.config import GOOGLE_DRIVE_FOLDER_ID

app = FastAPI(title="DriveRAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"
    thread_id: Optional[str] = None


class ClearRequest(BaseModel):
    thread_id: str


class SyncResponse(BaseModel):
    synced: int
    errors: int
    files: List[str]


@app.on_event("startup")
def startup_event():
    """
    Au demarrage : lance l'auto-sync en arriere-plan ET pre-charge les
    stats BM25 (vocab/df/n_docs) en memoire pour que la toute premiere
    requete /chat n'ait pas a les recalculer a la volee.
    """
    start_auto_sync(interval_seconds=1800)
    get_current_bm25_stats()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sync": get_auto_sync_status(),
        "indexed_files": len(get_indexed_files()),
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    NOTE IMPORTANTE sur thread_id :
    ---------------------------------
    En mono-utilisateur local, un thread_id stable et persistant cote
    FRONTEND (genere une fois au lancement d'une nouvelle conversation,
    stocke par le frontend, PAS regenere a chaque message) est suffisant
    et necessaire : c'est lui qui permet de retrouver l'historique de
    CETTE conversation specifique apres un refresh de page ou un
    redemarrage du serveur (l'historique vit dans SQLite, scope par
    thread_id -- voir conversation_store.py).

    Si ce projet passait un jour en multi-utilisateur (plusieurs
    etudiants, plusieurs Drive), thread_id devrait alors etre couple a
    un user_id (ex: thread_id = f"{user_id}:{conversation_uuid}") pour
    eviter qu'un utilisateur puisse, via un thread_id devine ou reutilise,
    lire l'historique de conversation d'un autre utilisateur. Ce n'est
    pas le cas ici (mono-utilisateur, un Drive par instance), donc le
    thread_id seul suffit comme cle d'isolation.
    """
    thread_id = req.thread_id or str(uuid.uuid4())

    try:
        if req.mode == "rag":
            vocab, df, n_docs = get_current_bm25_stats()
            result = await ask_async(
                question=req.message,
                vocab=vocab,
                df=df,
                n_docs=n_docs,
                thread_id=thread_id,
            )
        else:
            result = ask_direct(req.message, thread_id=thread_id)
            # plus besoin de calculer has_rag_history ici, ask_direct() s'en occupe

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
    """
    Sync manuel. Recalcule aussi les stats BM25 globales a la fin
    (delegue a smart_sync -> _refresh_bm25_stats), donc la prochaine
    requete /chat utilisera des stats a jour automatiquement.
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
    try:
        from src.ingestion.drive_uploader import upload_and_ingest
        file_bytes = await file.read()
        file_name = file.filename
        file_extension = Path(file_name).suffix.lower()
        ALLOWED = {".pdf", ".docx", ".txt", ".md", ".pptx"}
        if file_extension not in ALLOWED:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporte: {file_extension}. Accepte: {', '.join(ALLOWED)}",
            )
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
def clear_conversation(req: ClearRequest):
    """
    Efface l'historique d'UN SEUL thread (avant : effacait TOUTE la
    memoire globale partagee entre toutes les conversations).
    """
    try:
        clear_memory(req.thread_id)
        return {"status": "cleared", "thread_id": req.thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcription avec auto-detection de langue (biais FR/EN en cas
    d'ambiguite, voir whisper_service.py). Le modele Whisper est mis en
    cache (singleton), donc seul le PREMIER appel le charge depuis le
    disque -- avant, il etait recharge a chaque requete.
    """
    data = await audio.read()
    try:
        result = transcribe_audio(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads")
def list_conversation_threads():
    """Liste les conversations RAG passées (mode direct exclu : préfixe 'direct:')."""
    from src.generation.conversation_store import list_threads
    threads = list_threads()
    # On exclut les threads du mode direct de la liste affichée (pas pertinent
    # pour une sidebar "historique de conversation RAG")
    return [t for t in threads if not t["thread_id"].startswith("direct:")]


@app.get("/history/{thread_id}")
def get_thread_history(thread_id: str):
    from src.generation.conversation_store import get_history
    history = get_history(thread_id, max_turns=1000)  # tout l'historique, pas la limite LLM
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in history
    ]
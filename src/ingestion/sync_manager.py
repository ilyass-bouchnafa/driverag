"""
src/ingestion/sync_manager.py

Corrige deux points lies a la migration Qdrant + collision de noms :

1. L'ancien get_indexed_file_timestamps() indexait par `file_name`
   (nom de fichier). C'est exactement la collision documentee : deux
   fichiers Drive de meme nom dans des dossiers differents se
   melangeaient. Desormais on indexe par `drive_id` (identifiant Drive
   stable, jamais reutilise -- voir src/ingestion/chunk_identity.py).

2. Le sync ecrivait dans ChromaDB sans jamais recalculer les stats BM25
   globales (vocab/df/avgdl). Avec la migration Qdrant, ces stats DOIVENT
   etre recalculees a la fin de CHAQUE sync (pas a chaque requete, voir
   src/retrieval/qdrant_store.py) et persistees pour etre reutilisees
   par toutes les requetes utilisateur jusqu'au prochain sync.
"""

import logging
import threading
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages

from src.retrieval.qdrant_store import (
    index_chunks,
    get_all_chunks,
    delete_chunks_by_drive_id,
    rebuild_bm25_stats,
)

from src.config import GOOGLE_DRIVE_FOLDER_ID

logger = logging.getLogger(__name__)

# Persistance simple des stats BM25 sur disque : evite de devoir tout
# rescanner Qdrant au redemarrage du serveur avant la premiere requete.
BM25_STATS_PATH = Path(__file__).parent.parent.parent / "data" / "bm25_stats.pkl"

# Cache en memoire des stats BM25 courantes, partage par toutes les
# requetes jusqu'au prochain sync (c'est ce qui evite le recalcul a
# chaque appel utilisateur).
_current_vocab = None
_current_df = None
_current_avgdl = None
_current_n_docs = 0


def get_current_bm25_stats():
    """
    Retourne les stats BM25 actuellement en memoire. Si aucune n'a
    encore ete calculee dans ce process (ex: juste apres un redemarrage
    serveur), tente de les charger depuis le disque, sinon les recalcule
    depuis Qdrant une seule fois.
    """
    global _current_vocab, _current_df, _current_avgdl, _current_n_docs

    if _current_vocab is not None:
        return _current_vocab, _current_df, _current_n_docs

    if BM25_STATS_PATH.exists():
        with open(BM25_STATS_PATH, "rb") as f:
            _current_vocab, _current_df, _current_avgdl, _current_n_docs = pickle.load(f)
        logger.info("BM25 stats chargees depuis le disque")
        return _current_vocab, _current_df, _current_n_docs

    logger.warning("Aucune stat BM25 en cache, recalcul depuis Qdrant (peut etre lent une fois)")
    _refresh_bm25_stats()
    return _current_vocab, _current_df, _current_n_docs


def _refresh_bm25_stats():
    """Recalcule les stats BM25 depuis Qdrant et les persiste sur disque."""
    global _current_vocab, _current_df, _current_avgdl, _current_n_docs

    vocab, df, avgdl = rebuild_bm25_stats()
    n_docs = len(get_all_chunks())

    _current_vocab, _current_df, _current_avgdl, _current_n_docs = vocab, df, avgdl, n_docs

    BM25_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_STATS_PATH, "wb") as f:
        pickle.dump((vocab, df, avgdl, n_docs), f)

    logger.info(f"BM25 stats recalculees : {len(vocab)} tokens, {n_docs} chunks")


def get_indexed_file_state() -> dict:
    """
    Retourne {drive_id: drive_modified_time} pour tous les fichiers
    indexes -- scope par drive_id, PAS par nom de fichier (fix de la
    collision). Deux fichiers de meme nom dans des dossiers differents
    ont des drive_id distincts et ne se melangent plus jamais.
    """
    try:
        all_chunks = get_all_chunks()
        state = {}
        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            drive_id = meta.get("drive_id")
            mod_time = meta.get("drive_modified_time", "")
            if drive_id and drive_id not in state:
                state[drive_id] = mod_time
        logger.info(f"{len(state)} fichiers indexes (scope par drive_id)")
        return state
    except Exception as e:
        logger.warning(f"Echec recuperation etat indexe: {e}")
        return {}


def smart_sync(force_all: bool = False) -> dict:
    """
    Synchronisation intelligente avec Google Drive, scopee par drive_id.

    A la fin du sync (que des fichiers aient change ou non, sauf si
    rien n'a ete touche), les stats BM25 globales sont recalculees UNE
    FOIS -- c'est le seul moment ou ce recalcul a lieu, plus jamais a la
    requete utilisateur.
    """
    logger.info("Demarrage de la synchronisation intelligente...")

    stats = {"new": 0, "updated": 0, "skipped": 0, "errors": []}

    indexed_state = {} if force_all else get_indexed_file_state()

    try:
        drive_files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
        logger.info(f"{len(drive_files)} fichiers trouves sur Drive")
    except Exception as e:
        logger.error(f"Echec de listing Drive: {e}")
        stats["errors"].append(str(e))
        return stats

    any_change = False

    for file in drive_files:
        drive_id = file["id"]
        drive_mod_time = file.get("modifiedTime", "")

        if drive_id in indexed_state:
            stored_timestamp = indexed_state[drive_id]
            if stored_timestamp and stored_timestamp == drive_mod_time and not force_all:
                stats["skipped"] += 1
                continue
            action = "updated"
            delete_chunks_by_drive_id(drive_id)  # scope par drive_id, pas par nom
        else:
            action = "new"

        try:
            file_bytes = download_file(drive_id, file["name"], file["mimeType"])
            pages = extract_text_from_bytes(file_bytes, file["name"], file["format"])

            for page in pages:
                page["drive_id"] = drive_id
                page["drive_path"] = file["path"]
                page["file_format"] = file["format"]
                page["drive_modified_time"] = drive_mod_time

            chunks = chunk_pages(pages)

            # NOTE IMPORTANTE : index_chunks() a besoin du vocab/df/avgdl
            # COURANTS pour encoder les sparse vectors. On utilise les
            # stats encore valides de l'INDEXATION PRECEDENTE pour ce
            # batch ; elles seront recalculees globalement a la fin du
            # sync complet (voir plus bas), donc legerement obsoletes
            # pendant le sync lui-meme -- acceptable car le recalcul
            # final corrige tout avant que la moindre requete utilisateur
            # n'arrive.
            vocab, df, n_docs = get_current_bm25_stats()
            avgdl = _current_avgdl or 200.0
            index_chunks(chunks, vocab, df, avgdl)

            stats[action] += 1
            any_change = True
            logger.info(f"{file['name']} indexe ({len(chunks)} chunks)")

        except Exception as e:
            error_msg = f"Erreur sur {file['name']}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    if any_change:
        _refresh_bm25_stats()

    logger.info(f"Sync termine: {stats}")
    return stats


# =========================================================
# AUTO SYNC (THREAD DE FOND) -- inchangé dans sa logique
# =========================================================

_auto_sync_thread: Optional[threading.Thread] = None
_auto_sync_running = False
_last_sync_result = None
_last_sync_time = None


def start_auto_sync(interval_seconds: int = 1800):
    global _auto_sync_running, _auto_sync_thread

    if _auto_sync_running:
        logger.info("Auto-sync deja en cours")
        return

    _auto_sync_running = True

    def _sync_loop():
        global _last_sync_result, _last_sync_time
        logger.info(f"Auto-sync demarre (intervalle={interval_seconds}s)")
        while _auto_sync_running:
            time.sleep(interval_seconds)
            if not _auto_sync_running:
                break
            try:
                _last_sync_result = smart_sync()
                _last_sync_time = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                logger.error(f"Erreur auto-sync: {e}")
                _last_sync_result = {"error": str(e)}

    _auto_sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    _auto_sync_thread.start()
    logger.info("Thread auto-sync demarre")


def stop_auto_sync():
    global _auto_sync_running
    _auto_sync_running = False
    logger.info("Auto-sync arrete")


def get_auto_sync_status() -> dict:
    return {
        "running": _auto_sync_running,
        "last_result": _last_sync_result,
        "last_time": _last_sync_time,
    }
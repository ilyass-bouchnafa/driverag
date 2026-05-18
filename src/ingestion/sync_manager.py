"""
Smart Synchronization Manager

Overview
--------
This module implements an intelligent synchronization system between:
- Google Drive (source of truth)
- ChromaDB (vector database)

Problem
-------
Re-indexing all files every time is inefficient and slow.

Solution
--------
We use the Google Drive `modifiedTime` field to detect changes:
- New file      → Index it
- Modified file → Re-index it
- Unchanged     → Skip it

Key Idea
--------
Each chunk stored in ChromaDB contains metadata:
    {
        "source": "file.pdf",
        "drive_modified_time": "2024-01-01T10:00:00"
    }

During sync, we compare:
    Drive modifiedTime vs Stored modifiedTime

If different → file has changed.
"""

# ---------------------------------------------------------
# STANDARD LIBRARIES
# ---------------------------------------------------------
import logging
import threading
import time
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------
# GOOGLE DRIVE INGESTION
# ---------------------------------------------------------
from src.ingestion.gdrive_loader import (
    list_files_recursive,
    download_file
)

# ---------------------------------------------------------
# FILE PROCESSING PIPELINE
# ---------------------------------------------------------
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages

# ---------------------------------------------------------
# VECTOR STORE (ChromaDB)
# ---------------------------------------------------------
from src.retrieval.vectorstore import (
    add_chunks_to_store,
    get_all_chunks,
    delete_chunks_by_source
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
from src.config import GOOGLE_DRIVE_FOLDER_ID

logger = logging.getLogger(__name__)


# =========================================================
# FUNCTION: Get Indexed File Timestamps
# =========================================================
def get_indexed_file_timestamps() -> dict[str, str]:
    """
    Retrieve a mapping of indexed files and their last known modified time.

    Why?
    ----
    We store metadata in ChromaDB for each chunk.
    This allows us to reconstruct which files were indexed
    and when they were last modified.

    Returns
    -------
    dict[str, str]
        {
            "file1.pdf": "2024-01-01T10:00:00",
            "file2.docx": "2024-01-02T12:00:00"
        }
    """

    try:
        # -------------------------------------------------
        # STEP 1: Retrieve all chunks from ChromaDB
        # -------------------------------------------------
        all_chunks = get_all_chunks()

        timestamps = {}

        # -------------------------------------------------
        # STEP 2: Extract metadata from chunks
        # -------------------------------------------------
        for chunk in all_chunks:
            metadata = chunk.get("metadata", {})

            source = metadata.get("source")
            mod_time = metadata.get("drive_modified_time")

            # -------------------------------------------------
            # STEP 3: Store only one entry per file
            # -------------------------------------------------
            if source and mod_time and source not in timestamps:
                timestamps[source] = mod_time

        return timestamps

    except Exception as e:
        logger.warning(f"Failed to retrieve timestamps: {e}")
        return {}


# =========================================================
# FUNCTION: Smart Sync
# =========================================================
def smart_sync(force_all: bool = False) -> dict:
    """
    Perform intelligent synchronization with Google Drive.

    Algorithm
    ---------
    1. Get indexed files (from ChromaDB)
    2. Get current files (from Google Drive)
    3. For each file:
        - If not indexed → NEW → index it
        - If modified → UPDATED → delete + re-index
        - If unchanged → SKIP

    Parameters
    ----------
    force_all : bool
        If True, re-index all files regardless of modification

    Returns
    -------
    dict
        {
            "new": int,
            "updated": int,
            "skipped": int,
            "errors": list
        }
    """

    logger.info("🔄 Starting smart synchronization...")

    stats = {
        "new": 0,
        "updated": 0,
        "skipped": 0,
        "errors": []
    }

    # ---------------------------------------------------------
    # STEP 1: Load indexed file timestamps
    # ---------------------------------------------------------
    indexed_files = {} if force_all else get_indexed_file_timestamps()

    logger.info(f"📊 Indexed files: {len(indexed_files)}")

    # ---------------------------------------------------------
    # STEP 2: Retrieve files from Google Drive
    # ---------------------------------------------------------
    try:
        drive_files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
        logger.info(f"📂 Drive files: {len(drive_files)}")

    except Exception as e:
        logger.error(f"Failed to list Drive files: {e}")
        stats["errors"].append(str(e))
        return stats

    # ---------------------------------------------------------
    # STEP 3: Process each file
    # ---------------------------------------------------------
    for file in drive_files:

        file_name = file["name"]
        drive_mod_time = file.get("modifiedTime", "")

        # -----------------------------------------------------
        # STEP 4: Decide action (NEW / UPDATED / SKIP)
        # -----------------------------------------------------
        if file_name in indexed_files:

            # Case 1: Unchanged file
            if indexed_files[file_name] == drive_mod_time and not force_all:
                stats["skipped"] += 1
                continue

            # Case 2: Modified file
            logger.info(f"🔄 Updated file: {file_name}")

            delete_chunks_by_source(file_name)
            action = "updated"

        else:
            # Case 3: New file
            logger.info(f"🆕 New file: {file_name}")
            action = "new"

        # -----------------------------------------------------
        # STEP 5: Download file
        # -----------------------------------------------------
        try:
            file_bytes = download_file(
                file["id"],
                file["name"],
                file["mimeType"]
            )

            # -------------------------------------------------
            # STEP 6: Extract text
            # -------------------------------------------------
            pages = extract_text_from_bytes(
                file_bytes,
                file["name"],
                file["format"]
            )

            # -------------------------------------------------
            # STEP 7: Enrich metadata
            # -------------------------------------------------
            for page in pages:
                page["drive_path"] = file["path"]
                page["file_format"] = file["format"]

                # CRITICAL: store modifiedTime
                page["drive_modified_time"] = drive_mod_time

            # -------------------------------------------------
            # STEP 8: Chunk pages
            # -------------------------------------------------
            chunks = chunk_pages(pages)

            # -------------------------------------------------
            # STEP 9: Store in ChromaDB
            # -------------------------------------------------
            add_chunks_to_store(chunks)

            stats[action] += 1

            logger.info(f"✅ {file_name} indexed ({len(chunks)} chunks)")

        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    logger.info(f"✅ Sync completed: {stats}")
    return stats


# =========================================================
# AUTO SYNC (BACKGROUND THREAD)
# =========================================================

_auto_sync_thread: Optional[threading.Thread] = None
_auto_sync_running = False
_last_sync_result = None
_last_sync_time = None


def start_auto_sync(interval_seconds: int = 300):
    """
    Start automatic synchronization in the background.

    Why?
    ----
    - Avoid manual sync
    - Keep vector DB always up-to-date
    - Run without blocking Streamlit UI

    Parameters
    ----------
    interval_seconds : int
        Time between sync runs (default: 300s = 5 minutes)
    """

    global _auto_sync_running, _auto_sync_thread

    if _auto_sync_running:
        logger.info("Auto-sync already running")
        return

    _auto_sync_running = True

    # ---------------------------------------------------------
    # BACKGROUND LOOP
    # ---------------------------------------------------------
    def _sync_loop():
        global _last_sync_result, _last_sync_time

        logger.info(f"🤖 Auto-sync started (interval={interval_seconds}s)")

        while _auto_sync_running:
            time.sleep(interval_seconds)

            if not _auto_sync_running:
                break

            try:
                logger.info("🔄 Auto-sync running...")

                _last_sync_result = smart_sync()
                _last_sync_time = datetime.now().strftime("%H:%M:%S")

                logger.info(f"✅ Auto-sync result: {_last_sync_result}")

            except Exception as e:
                logger.error(f"❌ Auto-sync error: {e}")
                _last_sync_result = {"error": str(e)}

            

    # ---------------------------------------------------------
    # START THREAD
    # ---------------------------------------------------------
    _auto_sync_thread = threading.Thread(
        target=_sync_loop,
        daemon=True
    )

    _auto_sync_thread.start()

    logger.info("✅ Auto-sync thread started")


def stop_auto_sync():
    """
    Stop automatic synchronization.
    """
    global _auto_sync_running
    _auto_sync_running = False
    logger.info("⏹️ Auto-sync stopped")


def get_auto_sync_status() -> dict:
    """
    Retrieve current auto-sync status.

    Returns
    -------
    dict
        {
            "running": bool,
            "last_result": dict,
            "last_time": str
        }
    """
    return {
        "running": _auto_sync_running,
        "last_result": _last_sync_result,
        "last_time": _last_sync_time
    }
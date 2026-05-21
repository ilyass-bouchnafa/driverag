import io
import logging
from googleapiclient.http import MediaIoBaseUpload
from src.ingestion.gdrive_loader import get_drive_service
from src.config import GOOGLE_DRIVE_FOLDER_ID, SUPPORTED_MIME_TYPES

logger = logging.getLogger(__name__)

EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

def upload_file_to_drive(
    file_bytes: bytes,
    file_name: str,
    file_extension: str
) -> dict:
    service = get_drive_service()

    mime_type = EXTENSION_TO_MIME.get(
        file_extension.lower(), 
        "application/octet-stream"
    )

    file_metadata = {
        "name": file_name,
        "parents": [GOOGLE_DRIVE_FOLDER_ID]
    }

    media = MediaIoBaseUpload(
        io.BytesIO(file_bytes),
        mimetype=mime_type,
        resumable=True    # Allows large file uploads
    )

    logger.info(f"Drive upload: {file_name} ({mime_type})")

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, webViewLink"
    ).execute()

    logger.info(f"✅ Uploaded: {uploaded['name']} (id: {uploaded['id']})")
    return uploaded

def upload_and_ingest(
        file_bytes: bytes,
        file_name: str,
        file_extension: str
) -> dict:
    
    from src.ingestion.file_router import extract_text_from_bytes
    from src.ingestion.chunker import chunk_pages
    from src.retrieval.vectorstore import add_chunks_to_store

    result = {"upload": None, "chunks": 0, "error": None}

    try:
        uploaded = upload_file_to_drive(file_bytes, file_name, file_extension)
        result["upload"] = uploaded
        logger.info(f"Drive upload OK: {uploaded['name']}")
    except Exception as e:
        result["error"] = f"Drive upload error: {str(e)}"
        logger.error(result["error"])
        return result
    
    try:
        format_key = file_extension.lower().strip(".")
        pages = extract_text_from_bytes(file_bytes, file_name, format_key)

        for page in pages:
            page["drive_path"] = file_name
            page["file_format"] = format_key
        
        chunks = chunk_pages(pages)
        add_chunks_to_store(chunks)
        result["chunks"] = len(chunks)
        logger.info(f"Ingestion OK: {len(chunks)} chunks")
    
    except Exception as e:
        result["error"] = f"Upload OK but ingestion error: {str(e)}"
        logger.warning(result["error"])

    return result
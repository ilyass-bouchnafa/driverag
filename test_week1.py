"""
test_week1.py
--------------
Quick local test script to validate ingestion from Google Drive.
Run from the project root. This script lists Drive files, downloads
one sample file, extracts text, chunks it, and indexes the chunks.
"""

from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.retrieval.vectorstore import add_chunks_to_store, get_indexed_files
from src.config import GOOGLE_DRIVE_FOLDER_ID

def test_ingestion():
    print("=" * 50)
    print("TEST WEEK 1 — Multi-format Ingestion")
    print("=" * 50)

    # 1. List all files (recursive)
    print("\n1. Exploring Google Drive...")
    files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
    print(f"   → {len(files)} files found")
    for f in files:
        print(f"   · {f['path']} ({f['format']})")

    if not files:
        print("   ⚠️ No files found. Add documents to your Drive folder.")
        return

    # 2. Tester avec le premier fichier
    test_file = files[12]
    print(f"\n2. Test with: {test_file['path']}")

    file_bytes = download_file(test_file['id'], test_file['name'], test_file['mimeType'])
    pages = extract_text_from_bytes(file_bytes, test_file['name'], test_file['format'])
    chunks = chunk_pages(pages)

    print(f"   → {len(chunks)} chunks created")
    print(f"   → Example: {chunks[0]['text'][:200]}...")

    # 3. Indexer
    print("\n3. Indexing into ChromaDB...")
    add_chunks_to_store(chunks)
    indexed = get_indexed_files()
    print(f"\n✅ Indexed files: {[f['name'] for f in indexed]}")

if __name__ == "__main__":
    test_ingestion()
# test_week1.py — lancer depuis le dossier racine du projet
from src.ingestion.gdrive_loader import list_files_recursive, download_file
from src.ingestion.file_router import extract_text_from_bytes
from src.ingestion.chunker import chunk_pages
from src.retrieval.vectorstore import add_chunks_to_store, get_indexed_files
from src.config import GOOGLE_DRIVE_FOLDER_ID

def test_ingestion():
    print("=" * 50)
    print("TEST SEMAINE 1 — Ingestion Multi-format")
    print("=" * 50)

    # 1. Lister tous les fichiers (récursif)
    print("\n1. Exploration de Google Drive...")
    files = list_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
    print(f"   → {len(files)} fichiers trouvés")
    for f in files:
        print(f"   · {f['path']} ({f['format']})")

    if not files:
        print("   ⚠️ Aucun fichier. Ajoute des documents dans ton dossier Drive.")
        return

    # 2. Tester avec le premier fichier
    test_file = files[12]
    print(f"\n2. Test avec : {test_file['path']}")

    file_bytes = download_file(test_file['id'], test_file['name'], test_file['mimeType'])
    pages = extract_text_from_bytes(file_bytes, test_file['name'], test_file['format'])
    chunks = chunk_pages(pages)

    print(f"   → {len(chunks)} chunks créés")
    print(f"   → Exemple : {chunks[0]['text'][:200]}...")

    # 3. Indexer
    print("\n3. Indexation dans ChromaDB...")
    add_chunks_to_store(chunks)

    indexed = get_indexed_files()
    print(f"\n✅ Fichiers indexés : {[f['name'] for f in indexed]}")

if __name__ == "__main__":
    test_ingestion()
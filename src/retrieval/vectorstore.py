# Import ChromaDB for persistent vector storage
import chromadb

# Import ChromaDB for persistent vector storage
from src.config import CHROMA_PERSIST_DIR

# Import embedding functions
from src.retrieval.embedder import embed_texts, embed_query


# Name of the ChromaDB collection used in this project
COLLECTION_NAME = "driverag_docs"

# ---------------------------------------------------------
# In-memory document store
# ---------------------------------------------------------
# Stores full original chunk texts.
#
# Why needed?
# ----------
# In multi-representation indexing:
#
# - Search is performed on summaries (shorter semantic representations)
# - Final answer generation uses full original text
#
# This dictionary maps:
# chunk_id → full chunk text
_doc_store: dict[str, str] = {}

def get_collection():
    """
    Return the ChromaDB collection.

    If the collection does not exist, it is created automatically.

    Why PersistentClient?
    ---------------------
    - Vectors are stored on disk
    - Data survives application restart
    - No need to rebuild embeddings every time

    Cosine similarity is used because:
    - Embeddings are directional semantic vectors
    - Cosine works best for semantic search
    """

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,

        # HNSW index configured for cosine similarity
        metadata={"hnsw:space": "cosine"}
    )



def add_chunks_to_store(chunks: list[dict], summaries: dict[str, str] = None):
    """
    Index chunks into ChromaDB.

    Two modes exist:

    1) Simple mode:
       - Original chunk text is indexed directly

    2) Multi-representation mode:
       - Summary is indexed for search
       - Full original text is kept separately for generation

    Parameters
    ----------
    chunks : list[dict]
        Output produced by chunker.py

    summaries : dict[str, str], optional
        Mapping:
        chunk_id → summary

        Used when summaries are available
    """

    global _doc_store

    # Load ChromaDB collection
    collection = get_collection()

    # Prepare data containers
    texts_to_index = []
    metadatas = []
    ids = []
    full_texts = []

    # ---------------------------------------------------------
    # Process each chunk
    # ---------------------------------------------------------
    for chunk in chunks:

        # Unique chunk identifier
        # Combines:
        # source file + page number + chunk number
        chunk_id = (
            f"{chunk['metadata']['source']}"
            f"_p{chunk['metadata']['page']}"
            f"_c{chunk['metadata']['chunk_index']}"
        )

        # ---------------------------------------------------------
        # Decide what text is indexed
        # ---------------------------------------------------------
        # If summaries exist:
        # use summary for semantic retrieval
        #
        # Otherwise:
        # use original chunk text
        text_for_search = summaries.get(chunk_id, chunk["text"]) if summaries else chunk["text"]

        # Store indexed text
        texts_to_index.append(text_for_search)

        # Store indexed text
        metadatas.append(chunk["metadata"])

        # Store metadata
        ids.append(chunk_id)

        # Keep original text separately
        full_texts.append(chunk["text"])

        # Store full original chunk in memory
        _doc_store[chunk_id] = chunk["text"]

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------
    embeddings = embed_texts(texts_to_index)

    # ---------------------------------------------------------
    # Insert into ChromaDB
    # ---------------------------------------------------------
    collection.upsert(
        ids=ids,
        documents=texts_to_index,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"✅ {len(chunks)} chunks indexed in ChromaDB")

    # Also synchronize chunks to Redis to speed up BM25 retrieval
    from src.retrieval.redis_corpus import save_chunks_to_redis
    save_chunks_to_redis(chunks)



def dense_search(query: str, k: int = 10) -> list[dict]:
    """
    Perform semantic search inside ChromaDB.

    Search uses:
    ----------
    query embedding → nearest vectors

    Returns:
    --------
    Full original text from _doc_store
    (not indexed summaries)
    """

    collection = get_collection()

    # Encode query into embedding vector
    query_embedding = embed_query(query)

    # Perform nearest-neighbor vector search
    results = collection.query(
        query_embeddings=[query_embedding],

        # Number of nearest results
        n_results=k,

        # Return:
        # indexed text + metadata + distances
        include=["documents", "metadatas", "distances"]
    )

    docs = []

    # ---------------------------------------------------------
    # Rebuild result list
    # ---------------------------------------------------------
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        
        # Recreate chunk id
        chunk_id = (
            f"{meta['source']}_p{meta['page']}_c{meta['chunk_index']}"
        )

        # Retrieve full original text
        # If missing, fallback to indexed text
        full_text = _doc_store.get(chunk_id, doc)

        docs.append({
            "text": full_text,

            "metadata": meta,

            # Convert distance into similarity score
            "score": 1 - dist
        })

    return docs

def get_all_chunks() -> list[dict]:
    """
    Return all stored chunks.

    Used for BM25 because:
    ----------------------
    BM25 requires full corpus access
    """
    collection = get_collection()
    results = collection.get(include=["documents", "metadatas"])

    chunks = []

    for doc, meta in zip(results["documents"], results["metadatas"]):

        # Recreate chunk id
        chunk_id = f"{meta['source']}_p{meta['page']}_c{meta['chunk_index']}"

        # Retrieve original full text
        full_text = _doc_store.get(chunk_id, doc)

        chunks.append({
            "text": full_text,
            "metadata": meta
        })

    return chunks

def get_indexed_files() -> list[dict]:
    """
    Return indexed files list.

    Used for UI display:
    -------------------
    Shows:
    - file name
    - drive path
    - file format
    """
    all_chunks = get_all_chunks()
    seen = set()
    files = []
    
    for c in all_chunks:

        # Avoid duplicates:
        # one file may contain many chunks
        key = c["metadata"]["source"]
        if key not in seen:
            seen.add(key)
            files.append({
                "name": c["metadata"]["source"],
                "path": c["metadata"].get("drive_path", key),
                "format": c["metadata"].get("file_format", "?")
            })
    
    # Sort by path for better readability
    return sorted(files, key=lambda x: x["path"])

def delete_chunks_by_source(source_name: str):
    """
    Delete all chunks associated with a given source file.

    Why is this important?
    ----------------------
    In a RAG pipeline, each document is split into multiple chunks.
    When a file is updated in Google Drive, we must:

        1. Remove old chunks from the vector database
        2. Re-index the updated content

    This prevents:
        - Duplicate chunks
        - Outdated information
        - Retrieval inconsistencies

    Parameters
    ----------
    source_name : str
        Name of the source file (e.g., "Chapter4_Management.pdf")
    """

    # ---------------------------------------------------------
    # STEP 1: Get ChromaDB collection
    # ---------------------------------------------------------
    collection = get_collection()

    # ---------------------------------------------------------
    # STEP 2: Retrieve all chunks linked to this source
    # ---------------------------------------------------------
    # We filter using metadata: {"source": source_name}
    results = collection.get(
        where={"source": source_name},
        include=["documents"]
    )

    # ---------------------------------------------------------
    # STEP 3: Delete chunks from vector store
    # ---------------------------------------------------------
    if results["ids"]:
        collection.delete(ids=results["ids"])

        print(f"🗑️ Deleted {len(results['ids'])} chunks for {source_name}")

        # -----------------------------------------------------
        # STEP 4: Clean in-memory document store
        # -----------------------------------------------------
        # Some systems keep a local cache (doc_store)
        # We must also remove entries to avoid stale references
        global _doc_store

        keys_to_delete = [
            key for key in _doc_store
            if key.startswith(source_name)
        ]

        # Remove matching entries from the in-memory store and Redis
        from src.retrieval.redis_corpus import delete_chunks_by_source_redis
        if keys_to_delete:
            delete_chunks_by_source_redis(source_name)

        for key in keys_to_delete:
            del _doc_store[key]
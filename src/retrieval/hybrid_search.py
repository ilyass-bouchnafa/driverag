# Import BM25 implementation for lexical (keyword-based) search
from rank_bm25 import BM25Okapi

# Import dense semantic search and full corpus access
from src.retrieval.vectorstore import dense_search, get_all_chunks

# Import configuration parameters
from src.config import TOP_K_RETRIEVAL, HYBRID_ALPHA

def hybrid_search(query: str, k: int = TOP_K_RETRIEVAL, alpha: float = HYBRID_ALPHA) -> list[dict]:
    """
    Perform hybrid retrieval combining:
    - BM25 (lexical keyword matching)
    - Dense search (semantic similarity)

    Why hybrid search?
    ------------------
    - BM25 is good for exact keyword matching
    - Dense search is good for semantic understanding
    - Combining both improves retrieval quality

    Parameters
    ----------
    query : str
        User query

    k : int
        Number of results to return

    alpha : float
        Weight for dense score (between 0 and 1)
        final_score = alpha * dense + (1 - alpha) * bm25

    Returns
    -------
    list[dict]
        Top-k ranked chunks with combined scores
    """

    # ---------------------------------------------------------
    # STEP 1: Load all indexed chunks
    # ---------------------------------------------------------
    # Required for BM25 because it needs the full corpus
    all_chunks = get_all_chunks()

    # If no data is available, return empty list
    if not all_chunks:
        return []
    
    # ---------------------------------------------------------
    # STEP 2: BM25 (keyword-based retrieval)
    # ---------------------------------------------------------
    # Tokenize each chunk text into words
    tokenized = [c["text"].lower().split() for c in all_chunks]

    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized)

    # Compute BM25 scores for the query
    bm25_scores = bm25.get_scores(query.lower().split())
    
    # Normalize BM25 scores between 0 and 1
    # Avoid division by zero
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = [float(s) / max_bm25 for s in bm25_scores]

    # ---------------------------------------------------------
    # STEP 3: Dense search (semantic retrieval)
    # ---------------------------------------------------------
    # Retrieve top results using embeddings
    dense_results = dense_search(query, k=min(len(all_chunks), 100))

    # Helper function to build unique chunk identifier
    def chunk_key(meta):
        return f"{meta['source']}_p{meta['page']}_c{meta['chunk_index']}"
    
    # Create a mapping:
    # chunk_id → dense score
    dense_map = {chunk_key(r["metadata"]): r["score"] for r in dense_results}

    # ---------------------------------------------------------
    # STEP 4: Score fusion (BM25 + Dense)
    # ---------------------------------------------------------
    scored = []

    for i, chunk in enumerate(all_chunks):

        # Recreate chunk identifier
        key = chunk_key(chunk["metadata"])

        # Get dense score (default = 0 if not retrieved)
        d_score = dense_map.get(key, 0.0)

        # Get BM25 score
        b_score = bm25_norm[i]

        # Combine both scores using weighted sum
        combined = alpha * d_score + (1 - alpha) * b_score

        # Store chunk with final score
        scored.append({**chunk, "score": combined})
    # ---------------------------------------------------------
    # STEP 5: Sort and return top-k results
    # ---------------------------------------------------------
    # Sort chunks by score (highest first)
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    return scored[:k]


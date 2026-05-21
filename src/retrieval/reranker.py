from  sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, TOP_K_RERANKED

# Global variable to cache the reranker model (singleton pattern)
_reranker = None

def get_reranker() -> CrossEncoder:
    """
    Load and return the CrossEncoder model (loaded only once).

    Why?
    ----
    - Loading a model is expensive
    - We reuse the same instance across calls
    - Improves performance significantly

    Returns
    -------
    CrossEncoder
        The loaded reranker model
    """

    global _reranker

    # Load the model only once
    if _reranker is None:
        print(f"⚖️  Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker

def rerank(question: str, chunks: list[dict], top_k: int = TOP_K_RERANKED) -> list[dict]:
    """
    Rerank retrieved chunks using a CrossEncoder model.

    Key Idea:
    ---------
    - Dense retrieval compares embeddings independently (approximate)
    - CrossEncoder reads (question + chunk) together → more precise scoring

    Analogy:
    --------
    - Dense search = comparing two images separately
    - CrossEncoder = looking at both images side by side

    Parameters
    ----------
    question : str
        The original user query

    chunks : list[dict]
        List of retrieved chunks (usually top 10-20 from retrieval)

    top_k : int
        Number of top chunks to keep after reranking

    Returns
    -------
    list[dict]
        Top-k reranked chunks based on CrossEncoder scores
    """

    # ---------------------------------------------------------
    # STEP 1: Handle empty input
    # ---------------------------------------------------------
    if not chunks:
        return []

    # ---------------------------------------------------------
    # STEP 2: Load reranker model
    # ---------------------------------------------------------
    reranker = get_reranker()

    # ---------------------------------------------------------
    # STEP 3: Build (question, chunk) pairs
    # ---------------------------------------------------------
    # Each pair is evaluated jointly by the CrossEncoder
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # ---------------------------------------------------------
    # STEP 4: Compute relevance scores
    # ---------------------------------------------------------
    # The model returns a score for each (question, chunk) pair
    scores = reranker.predict(pairs)

    # ---------------------------------------------------------
    # STEP 5: Attach scores to chunks
    # ---------------------------------------------------------
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # ---------------------------------------------------------
    # STEP 6: Sort by rerank score (descending)
    # ---------------------------------------------------------
    # Sort and apply a lightweight quality filter
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Keep chunks with a reasonable reranker score (> -5.0) as a simple quality threshold
    good_chunks = [c for c in sorted_chunks if c.get("rerank_score", 0) > -5.0]

    # If the quality filter removes everything, fall back to the top results to guarantee output
    if not good_chunks:
        good_chunks = sorted_chunks[:top_k]

    return good_chunks[:top_k]
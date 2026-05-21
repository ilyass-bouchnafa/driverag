# Import the SentenceTransformer model from the sentence-transformers library
from sentence_transformers import SentenceTransformer

# Import embedding model configuration from project settings
from src.config import EMBEDDING_MODEL

# Global variable to store the loaded model (singleton pattern)
_model = None

def get_embedder() -> SentenceTransformer:
    """
    Load and return the embedding model as a singleton.

    Why singleton?
    --------------
    - Loading the model can be slow (hundreds of MBs)
    - Reusing the same model avoids multiple loads in memory
    - Ensures consistent embeddings across the application

    The chosen model converts any text into a fixed-size vector:
    - Texts with similar meaning → vectors close in the embedding space
    - Sentence-transformers provides multilingal support for FR + EN

    Returns
    -------
    SentenceTransformer
        The embedding model ready to encode text
    """

    global _model
    if _model is None:
        # Load the model only once and print a message
        print(f"🤖 Loading embedder: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Encode a list of texts into vector embeddings.

    Args
    ----
    texts : list of str
        The textual content to encode, e.g., chunks of PDF pages

    Returns
    -------
    list of list of float
        Each inner list represents the vector embedding of a text
        These vectors can later be stored in a vector database for retrieval
    """

    model = get_embedder()

    # show_progress_bar=True → visual feedback when encoding many chunks
    # batch_size=32 → balance between speed and memory usage on CPU
    return model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

def embed_query(query: str) -> list[float]:
    """
    Encode a single query or question into a vector embedding.

    Args
    ----
    query : str
        The search or question text to embed

    Returns
    -------
    list of float
        A vector representing the query in the same embedding space as the document chunks
        Useful for semantic search or nearest-neighbor retrieval
    """
    
    model = get_embedder()
    return model.encode(query).tolist()


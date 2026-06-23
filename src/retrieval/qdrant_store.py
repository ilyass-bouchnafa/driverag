"""
src/retrieval/qdrant_store.py

Remplace src/retrieval/vectorstore.py + une grosse partie de
src/retrieval/hybrid_search.py.

Pourquoi cette migration ?
----------------------------
L'ancienne architecture (Chroma pour le dense + rank_bm25 reconstruit en
Python a chaque requete) a deux limites qui deviennent bloquantes au-dela
de quelques milliers de chunks :

  1. hybrid_search() relisait TOUT le corpus et retokenizait TOUT pour
     reconstruire un BM25Okapi() a CHAQUE appel -> O(corpus) par requete.
  2. La fusion sparse/dense se faisait via un score alpha calcule cote
     Python, sur deux echelles de score non comparables.

Qdrant resout les deux :
  - Les sparse vectors BM25 sont indexes UNE FOIS au moment du sync (pas
    a chaque requete), Qdrant fait le produit scalaire sparse cote
    serveur avec un index inverse -> O(log n) au lieu de O(corpus).
  - La fusion sparse+dense est native (qdrant Query API, mode RRF ou
    DBSF), plus de score alpha a regler a la main.

Une seule collection Qdrant par etudiant/corpus (mono-utilisateur local :
voir QDRANT_COLLECTION dans la config, scope par dossier Drive racine).
"""

from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    NamedVector,
    NamedSparseVector,
    PointStruct,
    FieldCondition,
    MatchValue,
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
)

from src.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    DENSE_VECTOR_SIZE,
)
from src.retrieval.sparse_encoder import (
    Vocabulary,
    tokenize,
    compute_corpus_stats,
    encode_document_sparse,
    encode_query_sparse,
)
from src.retrieval.embedder import embed_query, embed_texts

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    """Singleton client Qdrant (evite de rouvrir une connexion par appel)."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def ensure_collection():
    """
    Cree la collection Qdrant si elle n'existe pas, avec deux espaces
    vectoriels nommes : un dense (embeddings semantiques) et un sparse
    (BM25). Idempotent : ne fait rien si la collection existe deja.
    """
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    print(f"qdrant: collection '{QDRANT_COLLECTION}' creee")


def _stable_point_id(chunk_uid: str) -> str:
    """
    Qdrant exige soit un int, soit un UUID comme point id. On derive un
    UUID deterministe depuis le chunk_uid (meme chunk_uid -> meme UUID,
    donc un re-sync remplace le point au lieu de le dupliquer).
    """
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_uid))


def index_chunks(chunks: List[dict], vocab: Vocabulary, df: Dict[str, int], avgdl: float):
    """
    Indexe une liste de chunks dans Qdrant (dense + sparse) en un seul
    batch upsert. A appeler depuis le pipeline de sync, PAS a la requete.

    Parameters
    ----------
    chunks : chunks produits par chunker.py (avec metadata.chunk_uid)
    vocab, df, avgdl : statistiques BM25 globales du corpus, calculees
        une fois par sync_manager.py via rebuild_bm25_stats() ci-dessous.
    """
    ensure_collection()
    client = get_client()

    texts = [c["text"] for c in chunks]
    dense_vectors = embed_texts(texts)  # batch, plus rapide qu'un par un
    n_docs = len(df) and sum(df.values()) or len(chunks)  # approx n_docs total

    points = []
    for chunk, dense_vec in zip(chunks, dense_vectors):
        tokens = tokenize(chunk["text"])
        sp_indices, sp_values = encode_document_sparse(
            tokens, vocab, df, max(len(chunks), 1), avgdl
        )

        points.append(PointStruct(
            id=_stable_point_id(chunk["metadata"]["chunk_uid"]),
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: SparseVector(indices=sp_indices, values=sp_values),
            },
            payload={
                "text": chunk["text"],
                **chunk["metadata"],
            },
        ))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"qdrant: {len(points)} chunks indexes")


def hybrid_search(
    query: str,
    vocab: Vocabulary,
    df: Dict[str, int],
    n_docs: int,
    k: int = 30,
) -> List[dict]:
    """
    Recherche hybride NATIVE Qdrant : sparse (BM25) + dense, fusionnes
    cote serveur par RRF. Remplace l'ancienne hybrid_search() qui
    recalculait BM25 en Python a chaque appel.

    Returns
    -------
    list[dict] : chunks au format {"text", "metadata", "score"}
        (meme format que l'ancien pipeline, pour ne rien casser en aval)
    """
    client = get_client()

    dense_vec = embed_query(query)
    q_indices, q_values = encode_query_sparse(query, vocab, df, n_docs)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=k * 2,
            ),
            Prefetch(
                query=SparseVector(indices=q_indices, values=q_values),
                using=SPARSE_VECTOR_NAME,
                limit=k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        payload = dict(point.payload)
        text = payload.pop("text")
        chunks.append({
            "text": text,
            "metadata": payload,
            "score": float(point.score),
        })
    return chunks


def get_all_chunks() -> List[dict]:
    """
    Recupere tous les chunks indexes (utilise par sync_manager.py pour
    recalculer les stats BM25 globales df/avgdl apres chaque sync, PAS
    par le chemin de requete utilisateur).
    """
    client = get_client()
    ensure_collection()

    all_chunks = []
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )
        for record in records:
            payload = dict(record.payload)
            text = payload.pop("text")
            all_chunks.append({"text": text, "metadata": payload})
        if next_offset is None:
            break
    return all_chunks


def get_indexed_files() -> List[dict]:
    """Liste des fichiers indexes, deduplique par drive_id (pas par nom)."""
    all_chunks = get_all_chunks()
    seen = set()
    files = []
    for c in all_chunks:
        drive_id = c["metadata"].get("drive_id", c["metadata"]["source"])
        if drive_id not in seen:
            seen.add(drive_id)
            files.append({
                "name": c["metadata"]["source"],
                "path": c["metadata"].get("drive_path", c["metadata"]["source"]),
                "format": c["metadata"].get("file_format", "?"),
                "drive_id": drive_id,
            })
    return sorted(files, key=lambda x: x["path"])


def delete_chunks_by_drive_id(drive_id: str):
    """
    Supprime tous les chunks d'un fichier, identifie par drive_id (PAS par
    nom de fichier -> corrige la collision quand deux fichiers Drive
    differents partagent le meme nom).
    """
    client = get_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="drive_id", match=MatchValue(value=drive_id))]
        ),
    )
    print(f"qdrant: chunks supprimes pour drive_id={drive_id}")


def rebuild_bm25_stats() -> tuple:
    """
    Recalcule les statistiques globales BM25 (vocabulaire, document
    frequency, longueur moyenne) sur tout le corpus actuel.

    A appeler UNE FOIS apres chaque sync complet (pas a chaque requete).
    Le vocabulaire est cumulatif (voir Vocabulary.get_or_add) : un sync
    partiel qui ajoute peu de nouveaux documents ne rebat pas les
    indices existants.
    """
    all_chunks = get_all_chunks()
    if not all_chunks:
        return Vocabulary(), {}, 1.0

    tokenized = [tokenize(c["text"]) for c in all_chunks]
    df, avgdl = compute_corpus_stats(tokenized)

    vocab = Vocabulary()
    for tokens in tokenized:
        for t in set(tokens):
            vocab.get_or_add(t)

    return vocab, df, avgdl
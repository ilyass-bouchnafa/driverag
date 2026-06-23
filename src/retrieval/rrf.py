"""
src/retrieval/rrf.py

Reciprocal Rank Fusion (RRF), utilisee a deux niveaux dans le pipeline :

  Niveau 1 (par variante de requete) : fusion sparse (BM25) + dense
            -> remplace l'ancien score alpha (alpha*dense + (1-alpha)*bm25)

  Niveau 2 (entre variantes)         : fusion des resultats de
            [requete originale, reformulations multi-query, HyDE]

Pourquoi RRF plutot que la somme ponderee (alpha) ?
----------------------------------------------------
- Le score alpha melange deux echelles de score totalement differentes
  (BM25 brut vs cosine similarity), normalisees "a la main" -> fragile,
  sensible au corpus, demande un reglage manuel de alpha par dataset.
- RRF ne regarde QUE le rang de chaque document dans chaque liste, pas le
  score brut. Un document classe 1er en dense ET 3eme en sparse remonte
  naturellement, sans avoir besoin de comparer des echelles incomparables.
- C'est la methode utilisee en production par la plupart des moteurs de
  recherche hybrides (Elasticsearch RRF, Qdrant fusion native, etc.).
"""

from collections import defaultdict
from typing import Dict, List


def _chunk_key(chunk: dict) -> str:
    """Identifiant unique et stable d'un chunk (voir chunk_identity.py)."""
    meta = chunk["metadata"]
    return meta.get("chunk_uid") or (
        f"{meta.get('drive_id', meta.get('source'))}_"
        f"p{meta.get('page', 0)}_c{meta.get('chunk_index', 0)}"
    )


def reciprocal_rank_fusion(
    results_list: List[List[dict]],
    k: int = 60,
    top_n: int = 20,
) -> List[dict]:
    """
    Fusionne plusieurs listes classees de chunks en une seule liste,
    via Reciprocal Rank Fusion.

    Parameters
    ----------
    results_list : liste de listes de chunks, chaque sous-liste DEJA
        triee par pertinence decroissante (rang 1 = le plus pertinent).
    k : constante de lissage RRF (60 est la valeur standard de la
        litterature ; plus k est grand, plus les rangs bas comptent).
    top_n : nombre de chunks a garder apres fusion.

    Returns
    -------
    Liste de chunks tries par score RRF, dans le meme format d'entree
    (dict avec "metadata", "text", ...), enrichis d'un champ "score".
    """
    score_dict: Dict[str, float] = defaultdict(float)
    chunk_dict: Dict[str, dict] = {}

    for ranked_list in results_list:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = _chunk_key(chunk)
            score_dict[key] += 1.0 / (k + rank)
            if key not in chunk_dict:
                chunk_dict[key] = chunk

    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for key, rrf_score in sorted_items[:top_n]:
        chunk = dict(chunk_dict[key])
        chunk["score"] = round(rrf_score, 5)
        final_results.append(chunk)

    return final_results
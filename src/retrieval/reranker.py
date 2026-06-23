"""
src/retrieval/reranker.py

Probleme de l'ancien seuil `rerank_score > -5.0` :
------------------------------------------------------
Ce seuil est une constante absolue choisie "a vue de nez", jamais
calibree sur des donnees reelles. Or l'echelle de score d'un CrossEncoder
depend ENTIEREMENT du modele utilise : -5.0 peut etre "tres mauvais" pour
un modele et "moyen" pour un autre. Deux consequences observees :
  - Si TOUS les chunks retrieves sont moyennement pertinents (score autour
    de -6 a -8), le seuil rejette tout -> fallback brutal vers le top_k
    brut, donc le filtre ne sert a rien dans ce cas.
  - Si un seul chunk est tres pertinent (-1) et les autres mediocres
    (-4.9), le seuil absolu garde des chunks limites qui n'auraient pas
    du passer.

Fix : seuil RELATIF au meilleur score de CETTE requete.
-----------------------------------------------------------
On garde les chunks dont le score est au-dessus de :
    best_score - RELATIVE_MARGIN
plutot qu'au-dessus d'une constante absolue. Ca s'adapte automatiquement
a l'echelle du modele ET au niveau de difficulte de chaque question
(si la meilleure correspondance est deja faible, on ne garde que ce qui
est proche d'elle ; si elle est tres bonne, on garde plus de marge).

On garde aussi un plancher minimal absolu (FLOOR_SCORE) pour ecarter le
cas degenere ou MEME le meilleur chunk est clairement hors-sujet (toute
la liste est mauvaise) -- dans ce cas, mieux vaut dire "je ne sais pas"
que de forcer une reponse a partir de contexte non pertinent.
"""

import threading
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL, TOP_K_RERANKED

_reranker = None
_lock = threading.Lock()

# Marge relative : on garde tout chunk dont le score est a moins de
# RELATIVE_MARGIN points du meilleur score de la requete courante.
RELATIVE_MARGIN = 2.5

# Plancher absolu de secours : si MEME le meilleur chunk est sous ce
# seuil, aucun chunk n'est considere pertinent (corpus hors-sujet).
# Calibre empiriquement sur cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 :
# en dessous de -8, le modele indique typiquement une non-correspondance
# nette. A re-valider avec evaluation/ragas_eval.py si tu changes de
# modele de reranking.
FLOOR_SCORE = -8.0


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                print(f"reranker: chargement du modele {RERANKER_MODEL}")
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(question: str, chunks: list[dict], top_k: int = TOP_K_RERANKED) -> list[dict]:
    """
    Reordonne les chunks par pertinence (question, chunk) avec un
    CrossEncoder, puis filtre par seuil RELATIF au meilleur score de
    cette requete (voir docstring du module pour le raisonnement).
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    best_score = sorted_chunks[0]["rerank_score"]

    if best_score < FLOOR_SCORE:
        # Meme le meilleur chunk est mauvais : le corpus ne contient
        # probablement pas de reponse pertinente a cette question.
        # On retourne quand même le top_k pour laisser le LLM répondre
        # "information non trouvée" en connaissance de cause plutôt que
        # de planter sur une liste vide.
        return sorted_chunks[:top_k]

    relative_threshold = best_score - RELATIVE_MARGIN
    good_chunks = [c for c in sorted_chunks if c["rerank_score"] >= relative_threshold]

    if not good_chunks:
        good_chunks = sorted_chunks[:top_k]

    return good_chunks[:top_k]
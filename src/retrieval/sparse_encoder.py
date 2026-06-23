"""
src/retrieval/sparse_encoder.py

BM25 -> sparse vector, compatible avec Qdrant SparseVector.

Pourquoi pas fastembed/BM25 tout fait ?
----------------------------------------
On garde le controle total sur la tokenization (multilingue FR/EN/AR sans
dependance lourde), et le vocabulaire est persiste dans Qdrant lui-meme
(pas de fichier pickle externe a synchroniser).

Principe :
----------
1. On construit un vocabulaire global (token -> index) au moment du sync.
2. Chaque chunk est encode en SparseVector (indices = tokens presents,
   values = poids BM25 du token dans CE document).
3. La requete utilisateur est encodee avec les memes indices de vocabulaire.
4. Qdrant calcule lui-meme le produit scalaire sparse cote serveur :
   plus de "recharger tout le corpus et retokenizer" a chaque appel.
"""

import re
import math
import json
from collections import Counter
from typing import Dict, List, Tuple

# BM25 standard (Robertson/Sparck Jones), parametres usuels
K1 = 1.5
B = 0.75

# Tokenizer simple mais multilingue : on garde lettres unicode (accents FR,
# arabe, etc.) et chiffres, on ignore la ponctuation. Pas de stemming pour
# rester language-agnostic (le stemming FR casserait l'EN et inversement).
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize un texte en minuscules, multilingue (FR/EN/AR/...)."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """
    Vocabulaire persistant token -> index entier stable.

    Stable = important : si l'index d'un token change entre deux syncs,
    les sparse vectors deja stockes dans Qdrant deviennent incoherents.
    On n'efface donc jamais un index existant, on ne fait qu'ajouter.
    """

    def __init__(self, token_to_id: Dict[str, int] = None):
        self.token_to_id: Dict[str, int] = token_to_id or {}

    def get_or_add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def get(self, token: str) -> int:
        """Retourne l'index existant, ou -1 si le token est inconnu
        (cas d'un mot de la requete jamais vu a l'indexation)."""
        return self.token_to_id.get(token, -1)

    def to_json(self) -> str:
        return json.dumps(self.token_to_id, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Vocabulary":
        return cls(json.loads(raw) if raw else {})

    def __len__(self) -> int:
        return len(self.token_to_id)


def compute_corpus_stats(documents_tokens: List[List[str]]) -> Tuple[Dict[str, int], float]:
    """
    Calcule les statistiques globales du corpus necessaires a BM25 :
    - document frequency (df) par token : dans combien de documents il apparait
    - longueur moyenne des documents (avgdl)

    Ces stats doivent etre recalculees a chaque sync complet (pas a chaque
    requete), c'est la difference fondamentale avec l'ancienne implementation.
    """
    df: Counter = Counter()
    total_len = 0
    for tokens in documents_tokens:
        total_len += len(tokens)
        for token in set(tokens):
            df[token] += 1
    avgdl = total_len / len(documents_tokens) if documents_tokens else 1.0
    return dict(df), avgdl


def encode_document_sparse(
    tokens: List[str],
    vocab: Vocabulary,
    df: Dict[str, int],
    n_docs: int,
    avgdl: float,
) -> Tuple[List[int], List[float]]:
    """
    Encode un document en sparse vector BM25 (poids par terme DANS ce doc).

    Formule BM25 du poids d'un terme t dans le document d :
        idf(t) * ( tf(t,d) * (k1+1) ) / ( tf(t,d) + k1*(1 - b + b*|d|/avgdl) )
    """
    if not tokens:
        return [], []

    tf = Counter(tokens)
    doc_len = len(tokens)
    indices: List[int] = []
    values: List[float] = []

    for token, freq in tf.items():
        token_df = df.get(token, 1)
        # idf classique BM25 (toujours positif, lisse les termes tres frequents)
        idf = math.log(1 + (n_docs - token_df + 0.5) / (token_df + 0.5))
        denom = freq + K1 * (1 - B + B * doc_len / avgdl)
        weight = idf * (freq * (K1 + 1)) / denom
        if weight > 0:
            indices.append(vocab.get_or_add(token))
            values.append(float(weight))

    return indices, values


def encode_query_sparse(
    query: str,
    vocab: Vocabulary,
    df: Dict[str, int],
    n_docs: int,
) -> Tuple[List[int], List[float]]:
    """
    Encode la requete utilisateur en sparse vector (poids idf simple,
    on n'a pas de "longueur de document" pour une requete).

    Les tokens absents du vocabulaire (jamais vus a l'indexation) sont
    ignores : ils ne peuvent matcher aucun document de toute facon.
    """
    tokens = tokenize(query)
    tf = Counter(tokens)
    indices: List[int] = []
    values: List[float] = []

    for token, freq in tf.items():
        idx = vocab.get(token)
        if idx == -1:
            continue  # mot inconnu du corpus, ignore (pas d'erreur)
        token_df = df.get(token, 1)
        idf = math.log(1 + (n_docs - token_df + 0.5) / (token_df + 0.5))
        if idf > 0:
            indices.append(idx)
            values.append(float(idf * freq))

    return indices, values
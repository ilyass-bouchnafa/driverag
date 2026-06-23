"""
src/generation/citation_guard.py

Corrige le probleme : "le LLM cite des pages/fichiers qui ne correspondent
pas a la realite ni au chunk reellement injecte dans le contexte".

Principe (citations forcees par ID, pas verification a posteriori) :
------------------------------------------------------------------------
Au lieu de laisser le LLM ecrire librement "[NomFichier.pdf, Page 12]" et
esperer qu'il invente pas, on lui donne CHAQUE chunk avec un identifiant
court et explicite (ex: [C1], [C2], ...) et on lui IMPOSE de citer
exclusivement ces identifiants. Le LLM ne peut plus halluciner un nom de
fichier ou une page : il ne fait que pointer vers une etiquette qu'on lui
a fournie, fermee, finie, verifiable.

Apres generation, on fait un controle de coherence (filet de securite,
pas le mecanisme principal) : on extrait tous les [Cx] presents dans la
reponse et on verifie qu'ils existent bien parmi les chunks injectes. Si
le LLM invente un [C99] qui n'existe pas (rare mais possible), on le
retire de la reponse plutot que de laisser une fausse reference.
"""

import re
from typing import List, Dict, Tuple

_CITATION_RE = re.compile(r"\[C(\d+)\]")


def build_context_with_ids(chunks: List[dict]) -> Tuple[str, Dict[str, dict]]:
    """
    Construit le contexte textuel envoye au LLM, avec un identifiant
    court [C1], [C2], ... par chunk, et retourne la table de mapping
    id -> metadata pour le remapping apres generation.

    Returns
    -------
    (context_str, id_to_chunk) :
        context_str : texte pret a injecter dans le prompt
        id_to_chunk : {"C1": chunk_dict, "C2": chunk_dict, ...}
    """
    parts = []
    id_to_chunk: Dict[str, dict] = {}

    for i, chunk in enumerate(chunks, start=1):
        cid = f"C{i}"
        id_to_chunk[cid] = chunk
        meta = chunk["metadata"]
        section = f" — {meta['section_title']}" if meta.get("section_title") else ""
        header = f"[{cid}] (source: {meta['source']}, page {meta['page']}{section})"
        parts.append(f"{header}\n{chunk['text']}")

    context_str = ("\n\n" + "─" * 40 + "\n\n").join(parts)
    return context_str, id_to_chunk


def resolve_citations(answer: str, id_to_chunk: Dict[str, dict]) -> Tuple[str, List[dict]]:
    """
    Post-traite la reponse du LLM :
      1. Retire les citations [Cx] qui ne correspondent a AUCUN chunk
         reellement injecte (hallucination d'ID -> filet de securite).
      2. Remplace chaque [Cx] valide par sa reference lisible humaine
         [NomFichier, Page X].
      3. Construit la liste structuree des sources reellement citees
         (pour affichage UI, distincte de "tous les chunks retrieves" :
         seules les sources EFFECTIVEMENT citees dans le texte apparaissent).

    Returns
    -------
    (clean_answer, cited_sources)
    """
    cited_sources = []
    seen_ids = set()

    def _replace(match: re.Match) -> str:
        cid = f"C{match.group(1)}"
        chunk = id_to_chunk.get(cid)
        if chunk is None:
            # Le LLM a cite un identifiant qui n'existe pas dans le
            # contexte fourni : on supprime silencieusement plutot que
            # d'afficher une fausse reference a l'utilisateur.
            return ""

        meta = chunk["metadata"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            cited_sources.append({
                "file": meta["source"],
                "path": meta.get("drive_path", meta["source"]),
                "page": meta["page"],
                "section_title": meta.get("section_title"),
                "total_pages": meta.get("total_pages", "?"),
                "format": meta.get("file_format", "?"),
                "score": round(chunk.get("rerank_score", chunk.get("score", 0)), 3),
            })

        return f"[{meta['source']}, Page {meta['page']}]"

    clean_answer = _CITATION_RE.sub(_replace, answer)
    # Nettoyage des doubles espaces laisses par une citation supprimee
    clean_answer = re.sub(r"[ \t]{2,}", " ", clean_answer).strip()

    return clean_answer, cited_sources


def detect_dominant_language(chunks: List[dict]) -> str:
    """
    Heuristique legere pour estimer la langue dominante des chunks
    retrieves (utile pour decider si la reponse doit etre en FR/EN/etc
    quand la question et les documents ne sont pas dans la meme langue,
    voir docstring de llm_chain.py pour la regle complete).

    On ne fait pas une vraie detection de langue (couteux, dependance
    lourde) : un comptage de mots-outils frequents FR vs EN suffit a
    distinguer les deux cas dominants du corpus academique ENSA/IMT.
    """
    fr_markers = {"le", "la", "les", "de", "des", "et", "est", "une", "un", "dans", "pour", "que"}
    en_markers = {"the", "is", "are", "and", "of", "in", "for", "to", "this", "that"}

    fr_score, en_score = 0, 0
    for chunk in chunks[:5]:  # un echantillon suffit, pas besoin du tout le contexte
        words = chunk["text"].lower().split()
        fr_score += sum(1 for w in words if w in fr_markers)
        en_score += sum(1 for w in words if w in en_markers)

    if fr_score == 0 and en_score == 0:
        return "unknown"
    return "fr" if fr_score >= en_score else "en"


_CITATION_LABEL_RE = re.compile(r"\[([^\]]+),\s*Page\s*\d+\]")

def strip_citations_for_eval(answer: str) -> str:
    """Retire les tags [Fichier, Page X] pour l'évaluation RAGAS."""
    clean = _CITATION_LABEL_RE.sub("", answer)
    return re.sub(r"[ \t]{2,}", " ", clean).strip()

_CX_RE = re.compile(r"\s*\[C\d+\]")

def strip_cx_for_eval(raw_answer: str) -> str:
    """
    Retire les [Cx] de la réponse brute du LLM (avant resolve_citations).
    Préserve la ponctuation : "[C1]." devient "." et non " ."
    Utilisé exclusivement pour l'évaluation RAGAS — jamais pour l'UI.
    """
    clean = _CX_RE.sub("", raw_answer)
    return re.sub(r"[ \t]{2,}", " ", clean).strip()
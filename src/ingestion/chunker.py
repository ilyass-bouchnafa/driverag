"""
src/ingestion/chunker.py

Probleme de l'ancienne implementation :
-----------------------------------------
RecursiveCharacterTextSplitter avec separateurs generiques
["\\n\\n", "\\n", ". ", "! ", "? ", " ", ""] decoupe par TAILLE DE CARACTERES
sans aucune notion de structure du document. Resultat concret observe :
un chunk peut commencer au milieu d'une definition, d'un theoreme, ou
d'un tableau -> le rerank et la citation pointent vers un fragment qui,
hors contexte, ne veut plus rien dire.

Principe du fix :
------------------
On decoupe en respectant la structure logique du document AVANT de
decouper par taille :

  1. Detection de structure : titres markdown (#, ##), titres numerotes
     ("1.2.3 Introduction"), ou a defaut paragraphes vides comme
     separateurs de section.
  2. Chaque SECTION garde son titre en prefixe de chunk (le titre fait
     partie du contexte semantique du chunk, meme si on doit le re-couper
     ensuite par taille).
  3. A l'interieur d'une section, on decoupe par taille SEULEMENT si la
     section depasse CHUNK_SIZE, avec RecursiveCharacterTextSplitter en
     filet de securite (jamais l'inverse).
  4. Chaque chunk recoit en metadata son "section_title" : utile pour le
     reranking (le titre de section peut etre concatene a la requete)
     et pour l'affichage des citations a l'utilisateur (plus parlant
     qu'un simple numero de page).

Limites assumees :
-------------------
Cette heuristique fonctionne bien sur des supports de cours structures
(PDF avec titres, markdown, slides). Sur du texte brut sans aucune
structure (ex: un .txt sans titres), le comportement degenere proprement
vers l'ancien decoupage par taille -- pas de regression.
"""

import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.chunk_identity import build_chunk_uid

# Detecte les titres markdown ("# Titre", "## Sous-titre", ...)
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Detecte les titres numerotes academiques classiques :
# "1. Introduction", "1.2 Definitions", "Chapitre 3 : ...", "III. ..."
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,3})|(?:[IVX]{1,4}\.)|(?:Chapitre\s+\d+))\s*[:\-\.]?\s+[A-ZÀ-Ü].{0,80}$",
    re.MULTILINE,
)

# Taille minimale pour considerer qu'une section vaut la peine d'etre
# gardee comme unite (sinon on la fusionne avec la suivante : evite des
# micro-chunks de type juste-un-titre-sans-contenu)
MIN_SECTION_CHARS = 40


def _split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Decoupe un texte en sections logiques en se basant sur les titres
    detectes (markdown ou numerotation academique).

    Retourne une liste de {"title": str | None, "content": str}.
    Si aucun titre n'est detecte, retourne une seule section sans titre
    (fallback transparent vers le comportement "texte brut").
    """
    headings = []
    for match in _MD_HEADING_RE.finditer(text):
        headings.append((match.start(), match.group(2).strip()))
    for match in _NUMBERED_HEADING_RE.finditer(text):
        headings.append((match.start(), match.group(0).strip()))

    if not headings:
        return [{"title": None, "content": text}]

    headings.sort(key=lambda h: h[0])

    sections = []
    for i, (start, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append({"title": title, "content": content})

    # Texte avant le premier titre detecte (souvent un en-tete ou intro)
    if headings[0][0] > 0:
        leading = text[: headings[0][0]].strip()
        if len(leading) >= MIN_SECTION_CHARS:
            sections.insert(0, {"title": None, "content": leading})

    return sections


def chunk_pages(pages: List[dict]) -> List[dict]:
    """
    Decoupe une liste de pages extraites en chunks, en respectant la
    structure logique du document (titres/sections) avant la taille.

    Parameters
    ----------
    pages : list[dict]
        Produit par file_router.py. Chaque page doit desormais contenir
        un champ "drive_id" (identifiant Drive stable du fichier source),
        en plus des champs existants ("text", "source", "page", ...).

    Returns
    -------
    list[dict]
        Chunks structures : {"text", "metadata": {..., "chunk_uid",
        "section_title"}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = []

    for page in pages:
        drive_id = page.get("drive_id") or page["source"]  # fallback si absent
        sections = _split_into_sections(page["text"])

        chunk_index = 0
        for section in sections:
            section_text = section["content"]

            # Si la section tient dans une seule taille de chunk, on la
            # garde intacte (pas de decoupage artificiel d'une definition
            # courte ou d'un theoreme).
            if len(section_text) <= CHUNK_SIZE:
                sub_chunks = [section_text]
            else:
                sub_chunks = splitter.split_text(section_text)
                # Le splitter peut isoler la ligne de titre dans son propre
                # micro-chunk (ex: juste "2. Types de filtres"). On la
                # recolle au chunk suivant pour ne jamais avoir un chunk
                # qui ne contient qu'un titre sans aucun contenu exploitable.
                if (
                    len(sub_chunks) >= 2
                    and section["title"]
                    and sub_chunks[0].strip() == section["title"].strip()
                ):
                    sub_chunks = [sub_chunks[0] + "\n\n" + sub_chunks[1]] + sub_chunks[2:]

            for sub_text in sub_chunks:
                cleaned = sub_text.strip()
                if not cleaned:
                    continue

                chunks.append({
                    "text": cleaned,
                    "metadata": {
                        "source": page["source"],
                        "drive_id": drive_id,
                        "page": page["page"],
                        "chunk_index": chunk_index,
                        "chunk_uid": build_chunk_uid(drive_id, page["page"], chunk_index),
                        "section_title": section["title"],
                        "total_pages": page["total_pages"],
                        "drive_path": page.get("drive_path", page["source"]),
                        "file_format": page.get("file_format", "unknown"),
                        "drive_modified_time": page.get("drive_modified_time", ""),
                    },
                })
                chunk_index += 1

    print(f"chunker: {len(chunks)} chunks crees ({sum(1 for c in chunks if c['metadata']['section_title'])} avec titre de section detecte)")
    return chunks
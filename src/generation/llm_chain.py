"""
src/generation/llm_chain.py

Assemble toutes les corrections :
  - Retrieval via Qdrant hybride (sparse+dense natif, plus de BM25
    recalcule a la requete)
  - Recherche parallele des variantes multi-query/HyDE (asyncio)
  - Reranking a seuil relatif (reranker.py)
  - Citations forcees par ID [Cx] -> remap vers [Fichier, Page] apres
    generation, avec filet de securite anti-hallucination d'ID
  - Historique de conversation scope par thread_id (SQLite, plus de
    fuite entre sessions)
  - Prompt systeme restructure : memes regles strictes, mais identite
    de produit explicite (assistant academique nomme, pas un LLM nu)
"""

import asyncio
import logging

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langsmith import traceable

from src.config import GROQ_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANKED
from src.retrieval.query_processor import advanced_retrieve_async
from src.retrieval.reranker import rerank
from src.retrieval.qdrant_store import rebuild_bm25_stats
from src.generation.citation_guard import build_context_with_ids, resolve_citations, detect_dominant_language
from src.generation.conversation_store import get_history, append_message, init_db

logger = logging.getLogger(__name__)
init_db()

# ---------------------------------------------------------------------
# PROMPT SYSTEME
# ---------------------------------------------------------------------
# Memes regles strictes que l'original (anti-hallucination, citations
# obligatoires, concision, fidelite au document). Ce qui change :
#   - Identite produit explicite ("StudyMind" a adapter au nom reel du
#     projet) plutot qu'un system prompt qui se presente comme un LLM nu
#     -- coherent avec l'ambition "produit concurrent de NotebookLM",
#     pas juste un wrapper de prompt.
#   - Citations par [Cx] obligatoire (pas de nom de fichier invente
#     possible, voir citation_guard.py) au lieu de [Fichier, Page] que
#     le LLM devait ecrire lui-meme.
#   - Regle explicite de langue de reponse quand question et documents
#     ne sont pas dans la meme langue (voir point sur le multilingue).
SYSTEM_PROMPT = """You are StudyMind, an academic research assistant that helps students understand and navigate their own personal document library (lecture notes, slides, course PDFs).

You are not a general-purpose chatbot: you are grounded exclusively in the documents the student has provided. Your value comes from precision and traceability, not from broad knowledge.

STRICT RULES — FOLLOW THEM ALL WITHOUT EXCEPTION:

1. Answer ONLY using information explicitly present in the DOCUMENTS section below. Never use outside knowledge to fill gaps.

2. Answer EXACTLY what was asked, nothing more:
   - No extra definitions, examples, or explanations unless explicitly requested.
   - No summary or conclusion unless explicitly requested.
   - Be concise: the minimum number of sentences that fully answers the question.

3. CITATIONS ARE MANDATORY AND MUST USE THE [Cx] FORMAT ONLY:
   - Every document is given to you labeled [C1], [C2], [C3], etc.
   - After every factual claim, cite the exact label it came from: example "A filter modifies an image using a kernel [C2]."
   - NEVER write a file name or page number yourself. NEVER invent a [Cx] label that wasn't given to you.
   - You may cite multiple labels for one claim if needed: [C1][C3].

4. If the answer (or any part of it) cannot be found in the DOCUMENTS section, respond EXACTLY with this sentence and NOTHING else:
   "I cannot find this information in the provided documents."
   - Never add explanations, partial answers, or "however, generally speaking...".
   - Never guess a [Cx] label just to produce a citation.

5. RESPONSE LANGUAGE:
   - Default: answer in the same language as the student's question.
   - Exception: if the question is in language A but the relevant documents are clearly in language B, and translating a key technical term would lose precision (formulas, exact terminology, named definitions), you may answer in language A but quote the precise technical term from the document in language B in parentheses. Always keep citations [Cx] regardless of language.

6. Use clear, formal, academic language. Avoid repetition and filler conclusions.

7. MULTI-PART QUESTIONS: if a question contains multiple sub-questions, answer each part independently using the rules above. A sub-question without supporting documents must receive the standard refusal sentence, while other sub-questions with supporting documents must still be answered normally with citations.

8. OUTPUT FORMATTING (MANDATORY):
   - Always use Markdown formatting in your answer.
   - If you list multiple items (filters, algorithms, types, steps, definitions),
     use a numbered list (1., 2., 3.) or bullet points (-), ONE item per line.
   - NEVER merge a list item and its definition into a single run-on sentence
     or paragraph. Each list item must be on its own line.
   - If an item has multiple sub-points (e.g. a filter with several properties),
     nest them as sub-bullets under that item, indented, not as a flat
     continuous paragraph.
   - Use a blank line between distinct sections of your answer (e.g. between
     a list of items and their detailed definitions).
   - Bold the key term being defined when relevant (e.g. **Filtre passe-bas**).
     
Always prioritize precision, conciseness, and strict fidelity to the student's question and to the documents actually provided to you.
"""


def _build_messages(question: str, context_with_ids: str, history: list) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(
        content=f"DOCUMENTS:\n{context_with_ids}\n\nQUESTION: {question}"
    ))
    return messages


@traceable(run_type="chain", name="RAG Query")
async def ask_async(
    question: str,
    vocab,
    df: dict,
    n_docs: int,
    thread_id: str,
) -> dict:
    """
    Pipeline RAG complet, version async (utilisee par le backend FastAPI
    qui est lui-meme async — pas de asyncio.run() imbrique necessaire).

    Parameters
    ----------
    vocab, df, n_docs : stats BM25 globales precalculees au sync (voir
        src.retrieval.qdrant_store.rebuild_bm25_stats), partagees entre
        toutes les requetes tant qu'un nouveau sync n'a pas eu lieu.
    thread_id : identifiant de conversation, utilise pour scoper
        l'historique dans SQLite (corrige la fuite entre sessions).
    """
    # ETAPE 1 : retrieval avance, parallele (multi-query + HyDE + hybrid search)
    candidates = await advanced_retrieve_async(question, vocab, df, n_docs, k=TOP_K_RETRIEVAL)

    if not candidates:
        return {
            "answer": "Aucun document indexe trouve. Lancez d'abord une synchronisation.",
            "sources": [],
        }

    # ETAPE 2 : reranking a seuil relatif
    final_chunks = rerank(question, candidates, top_k=TOP_K_RERANKED)

    # ETAPE 3 : construction du contexte avec IDs forces [C1], [C2], ...
    context_with_ids, id_to_chunk = build_context_with_ids(final_chunks)

    # ETAPE 4 : historique scope par thread_id (SQLite, plus de fuite)
    history = get_history(thread_id)

    # ETAPE 5 : appel LLM
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    messages = _build_messages(question, context_with_ids, history)
    response = await llm.ainvoke(messages)

    # ETAPE 6 : resolution des citations [Cx] -> [Fichier, Page] reels,
    # avec rejet silencieux des IDs hallucines (filet de securite)
    clean_answer, cited_sources = resolve_citations(response.content, id_to_chunk)

    from src.generation.citation_guard import strip_cx_for_eval
    answer_for_eval = strip_cx_for_eval(response.content)

    # ETAPE 7 : persistance de l'historique (scope par thread_id)
    append_message(thread_id, "human", question)
    append_message(thread_id, "ai", clean_answer)

    return {
        "answer": clean_answer,
        "answer_for_eval": answer_for_eval,
        "sources": cited_sources,
        "raw_contexts": [c["text"] for c in final_chunks],
        "dominant_doc_language": detect_dominant_language(final_chunks),
    }


def ask(question: str, vocab, df: dict, n_docs: int, thread_id: str) -> dict:
    """Wrapper synchrone (utile pour les scripts de test/CLI)."""
    return asyncio.run(ask_async(question, vocab, df, n_docs, thread_id))


def clear_memory(thread_id: str):
    """Efface l'historique d'UN thread specifique (pas tous les threads
    -- corrige clear_memory() qui effacait l'historique global avant)."""
    from src.generation.conversation_store import clear_thread
    clear_thread(thread_id)
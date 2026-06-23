"""
src/retrieval/query_processor.py

Corrige deux problemes lies a la latence :
---------------------------------------------
1. hybrid_search() etait appelee SEQUENTIELLEMENT pour chaque variante
   (question originale + 3 reformulations + HyDE = 5 appels l'un apres
   l'autre), et chacun recalculait BM25 depuis zero sur tout le corpus.
   -> Fix : les variantes de requete sont generees, PUIS toutes les
   recherches hybrides sont lancees EN PARALLELE (asyncio.gather), ET
   chaque recherche utilise les stats BM25 globales precalculees au sync
   (vocab/df/avgdl, voir qdrant_store.rebuild_bm25_stats), plus de
   reconstruction a la requete.

2. La fusion des resultats entre variantes se faisait par un simple
   tri sur un score deja fusionne alpha -- maintenant on utilise RRF a
   deux niveaux :
     Niveau 1 (DANS qdrant_store.hybrid_search) : sparse+dense fusionnes
       nativement par Qdrant (RRF cote serveur).
     Niveau 2 (ICI) : les listes resultats de chaque VARIANTE de requete
       (originale, reformulations, HyDE) sont a leur tour fusionnees par
       RRF -- coherent avec le niveau 1, pas de score alpha residuel.

Pourquoi generer multi-query ET HyDE en parallele aussi (pas seulement
les recherches) ?
--------------------------------------------------------------------------
Les deux sont des appels LLM independants (l'un ne depend pas de l'autre
pour etre genere) : il n'y a aucune raison de les attendre l'un apres
l'autre. On les lance avec asyncio.gather egalement.
"""

import asyncio
from typing import List, Dict, Tuple

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

from src.config import GROQ_API_KEY, LLM_MODEL, MULTI_QUERY_COUNT, TOP_K_RETRIEVAL
from src.retrieval.qdrant_store import hybrid_search
from src.retrieval.sparse_encoder import Vocabulary
from src.retrieval.rrf import reciprocal_rank_fusion


async def _generate_multi_queries_async(llm: ChatGroq, question: str) -> List[str]:
    """Version async de la generation de reformulations (voir docstring
    originale conservee ci-dessous pour le detail du prompt)."""
    prompt = f"""You are an expert information retrieval assistant.

Generate exactly {MULTI_QUERY_COUNT} reformulations of the following question.

IMPORTANT:
- Keep the SAME language as the original question
- Do NOT translate unless necessary

Rules:
- Do NOT add numbering
- Do NOT add explanations
- Do NOT repeat the original question
- Vary wording and phrasing

Original question:
{question}

Reformulations:"""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        reformulations = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ][:MULTI_QUERY_COUNT]
        return [question] + reformulations
    except Exception as e:
        print(f"multi-query LLM failed: {e}")
        return [question, question + " explanation", question + " definition"]


async def _generate_hyde_async(llm: ChatGroq, question: str) -> str:
    """Version async de la generation HyDE."""
    prompt = f"""You are an expert academic writer.

Write a detailed academic-style passage (150-200 words) that directly answers the following question.

IMPORTANT:
- Keep the SAME language as the original question
- Do NOT translate unless necessary

Requirements:
- Use formal academic tone
- Include precise technical vocabulary
- Do NOT mention that this is hypothetical
- Write as if this is a real document excerpt

Question:
{question}

Document excerpt:"""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"HyDE LLM failed: {e}")
        return question


async def _hybrid_search_async(
    query: str, vocab: Vocabulary, df: dict, n_docs: int, k: int
) -> List[dict]:
    """
    Wrap synchrone -> async via un thread pool : qdrant-client est
    synchrone, mais on veut lancer N recherches en parallele sans
    bloquer la boucle asyncio sur des appels reseau I/O-bound.
    """
    return await asyncio.to_thread(hybrid_search, query, vocab, df, n_docs, k)


async def advanced_retrieve_async(
    question: str,
    vocab: Vocabulary,
    df: dict,
    n_docs: int,
    k: int = TOP_K_RETRIEVAL,
) -> List[dict]:
    """
    Pipeline complet de retrieval avance, parallelise :

      1. Generation EN PARALLELE de : reformulations multi-query + HyDE
      2. Recherche hybride EN PARALLELE pour chaque variante de requete
         (utilise les stats BM25 globales deja calculees, pas de
         reconstruction a la requete)
      3. Fusion RRF (niveau 2) de tous les resultats de toutes les
         variantes en une seule liste classee

    Parameters
    ----------
    vocab, df, n_docs : stats BM25 globales, calculees une fois par
        sync (voir src.retrieval.qdrant_store.rebuild_bm25_stats),
        PAS recalculees ici.
    """
    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.7)
    llm_hyde = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

    # ETAPE 1 : generation des variantes, en parallele
    multi_queries, hyde_doc = await asyncio.gather(
        _generate_multi_queries_async(llm, question),
        _generate_hyde_async(llm_hyde, question),
    )
    all_variants = multi_queries + [hyde_doc]

    # ETAPE 2 : recherche hybride pour chaque variante, en parallele
    search_tasks = [
        _hybrid_search_async(variant, vocab, df, n_docs, k)
        for variant in all_variants
    ]
    results_per_variant = await asyncio.gather(*search_tasks)

    # ETAPE 3 : fusion RRF niveau 2 (entre variantes)
    fused = reciprocal_rank_fusion(list(results_per_variant), top_n=k)

    print(f"advanced_retrieve: {len(all_variants)} variantes interrogees en parallele, {len(fused)} chunks fusionnes")
    return fused


def advanced_retrieve(
    question: str,
    vocab: Vocabulary,
    df: dict,
    n_docs: int,
    k: int = TOP_K_RETRIEVAL,
) -> List[dict]:
    """
    Wrapper synchrone pour les appelants qui ne sont pas eux-memes async
    (ex: si llm_chain.ask() reste synchrone). Utilise asyncio.run en
    interne.
    """
    return asyncio.run(advanced_retrieve_async(question, vocab, df, n_docs, k))
"""
src/generation/llm_direct.py

Continuite asymetrique RAG <-> Direct, par choix delibere :

  RAG -> Direct : le mode Direct LIT le contenu de la conversation RAG
                  precedente (pas juste un signal "il y a eu du RAG avant").
                  Permet "explique-moi encore" de fonctionner naturellement
                  apres un switch de mode.

  Direct -> RAG : AUCUNE lecture, dans aucun sens. Le mode RAG ne voit
                  JAMAIS le contenu Direct, meme apres switch.

Pourquoi l'asymetrie est volontaire (et pas un compromis bancal) :
----------------------------------------------------------------------
Le mode RAG a une garantie stricte : "Answer ONLY using information
explicitly present in the DOCUMENTS section". Si l'historique Direct
(reponses non sourcees, non verifiees contre les documents) s'infiltrait
dans le contexte du RAG, le LLM RAG pourrait traiter une affirmation
Direct non verifiee comme un fait acquis -- ca casserait la garantie de
tracabilite qui fait toute la valeur ajoutee du mode RAG par rapport a
un chatbot generaliste.

Le mode Direct, lui, n'a AUCUNE contrainte de tracabilite (il dit lui
meme "answer from general knowledge"). Lui faire lire le contexte RAG
n'affaiblit aucune garantie : ca ameliore juste la continuite percue
par l'etudiant, sans risque de contamination dans l'autre sens.
"""

import logging
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

from src.config import GROQ_API_KEY, LLM_MODEL
from src.generation.conversation_store import get_history, append_message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_DIRECT = """You are an expert academic assistant.
Answer in a clear, structured, and precise way, in the same language as the question.

You can use your general knowledge to answer.
You can refer to previous messages in the conversation normally.
If you are not sure about something, say it clearly.
NEVER use citation formats like [File Name, Page X] — there are no source documents in this mode.
NEVER reference files, documents, or pages — answer from general knowledge only.

OUTPUT FORMATTING (MANDATORY):
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
"""

_RAG_CONTEXT_PREFIX = (
    "CONTEXT FROM A PREVIOUS DOCUMENT-GROUNDED CONVERSATION (for continuity only — "
    "this came from a different mode that only used the student's course documents. "
    "Treat it as conversational context, not as verified fact; if the student asks "
    "to continue or clarify something from it, you may use general knowledge to do so):"
)

RAG_CONTEXT_TURNS = 3


def _direct_thread_key(thread_id: str) -> str:
    return f"direct:{thread_id}"


@traceable(run_type="chain", name="Direct LLM Query")
def ask_direct(question: str, thread_id: str) -> dict:
    direct_thread_id = _direct_thread_key(thread_id)
    direct_history = get_history(direct_thread_id)

    messages = [SystemMessage(content=SYSTEM_PROMPT_DIRECT)]

    if not direct_history:
        rag_history = get_history(thread_id, max_turns=RAG_CONTEXT_TURNS)
        if rag_history:
            rag_transcript = "\n".join(
                f"{'Student' if m.type == 'human' else 'Assistant (RAG mode)'}: {m.content}"
                for m in rag_history
            )
            messages.append(SystemMessage(content=f"{_RAG_CONTEXT_PREFIX}\n\n{rag_transcript}"))

    messages.extend(direct_history)
    messages.append(HumanMessage(content=question))

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.7)

    try:
        response = llm.invoke(messages)
        answer = response.content

        append_message(direct_thread_id, "human", question)
        append_message(direct_thread_id, "ai", answer)

        logger.info(f"Direct mode: reponse generee pour '{question[:60]}'")
        return {"answer": answer, "sources": [], "mode": "direct"}
    except Exception as e:
        logger.error(f"Erreur Direct LLM: {e}")
        return {"answer": f"Erreur: {str(e)}", "sources": [], "mode": "direct"}
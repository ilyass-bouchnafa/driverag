import logging

# Import Groq LLM (fast inference)
from langchain_groq import ChatGroq

# Import message schema for structured chat input
from langchain.schema import SystemMessage, HumanMessage

from langchain.callbacks.manager import collect_runs

# Import project configuration
from src.config import GROQ_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANKED

# Import retrieval pipeline
from src.retrieval.query_processor import advanced_retrieve

# Import reranker
from src.retrieval.reranker import rerank

from langsmith import traceable, Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# SYSTEM PROMPT (MULTILINGUAL CONTROL)
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a rigorous, precise, and expert academic assistant specialized in the user's personal academic documents.

STRICT RULES - FOLLOW THEM ALL WITHOUT EXCEPTION:

1. You MUST answer ONLY using the information explicitly present in the provided context documents.

2. Answer the question EXACTLY as asked. Do not add any information that was not explicitly requested by the user.
   - Do not give extra definitions, advantages, examples, or explanations unless the question specifically asks for them.
   - Do not make summaries or conclusions unless the user explicitly asks for a summary.
   - BE CONCISE: answer in the minimum number of sentences necessary to fully answer the question.

3. For every important statement, claim, definition, or explanation, you MUST cite the exact source immediately after the sentence using this format: [File Name, Page X].
   Example: A filter is an operation that modifies an image using a kernel. [Chapitre I-II_Pr AMINE.pdf, Page 65]

4. If the answer (or any part of it) cannot be directly found or supported in the provided documents, respond **EXACTLY** with this sentence and add absolutely nothing else:
   "I cannot find this information in the provided documents."

5. PREVIOUS MESSAGES HANDLING:
   - If any previous assistant message starts with "[STRONG RESTRICTION:", it means this message comes from Direct LLM mode.
     → You MUST completely IGNORE that entire message.
     → Do NOT use any facts, explanations, or ideas from it.

6. NEVER invent, extrapolate, add extra details, or over-explain. Stick strictly to what the question asks and what the documents provide.
7. NEVER cite a source that does not explicitly appear in the provided documents.
8. Use clear, formal, precise, and academic language. Be concise and direct. Avoid unnecessary repetitions and conclusions.
9. RESPONSE LENGTH: Match the response length to the complexity of the question. Simple questions deserve short answers. Never pad responses with redundant information.

Always prioritize precision, conciseness, and strict fidelity to the user's question and the source documents.
"""

# ---------------------------------------------------------
# SIMPLE CONVERSATION MEMORY (last 5 exchanges)
# ---------------------------------------------------------
_history = []

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context string.

    Each chunk includes:
    - Source file name
    - Page number
    - Chunk text

    This helps the LLM cite sources correctly.
    """

    parts = []
    for chunk in chunks:
        meta = chunk["metadata"]
        parts.append(f"[{meta['source']}, Page {meta['page']}]\n{chunk['text']}")
    
    # Add visual separators between chunks
    return "\n\n" + "─" * 40 + "\n\n".join(parts)

@traceable(run_type="chain", name="RAG Query")
def ask(question: str, external_history: list = None, thread_id: str = None, conversation_id: str = None, langsmith_extra: dict = None) -> dict:
    """
    Full RAG pipeline:

    Question
        → Advanced Retrieval (Multi-query + HyDE + Hybrid Search)
        → Reranking (CrossEncoder)
        → Context building
        → LLM (Groq)
        → Answer with citations

    Parameters
    ----------
    question : str
        User query
    
    external_history : list
        liste de langchain.schema messages (optionnel)

    Returns
    -------
    dict
        {
            "answer": generated response,
            "sources": list of source metadata
        }
    """

    # ---------------------------------------------------------
    # STEP 1: Advanced Retrieval
    # ---------------------------------------------------------
    # Retrieve candidate chunks using multi-query + HyDE + hybrid search
    candidates = advanced_retrieve(question, k=TOP_K_RETRIEVAL)

    if not candidates:
        return {
            "answer": "⚠️ No indexed documents found. Please sync your data first.",
            "sources": []
        }
    
    # ---------------------------------------------------------
    # STEP 1: Advanced Retrieval
    # ---------------------------------------------------------
    # Retrieve candidate chunks using multi-query + HyDE + hybrid search
    final_chunks = rerank(question, candidates, top_k=TOP_K_RERANKED)

    # ---------------------------------------------------------
    # STEP 3: Build context
    # ---------------------------------------------------------
    # Convert chunks into a structured prompt context
    context = format_context(final_chunks)

    # ---------------------------------------------------------
    # STEP 4: Conversation history !!!
    # ---------------------------------------------------------
    # Keep only the last 5 exchanges (10 messages total)
    if external_history is not None:
        history = external_history[-10:]
    else:
        history = _history[-10:]

    # ---------------------------------------------------------
    # STEP 5: LLM call (Groq)
    # ---------------------------------------------------------
    llm = ChatGroq(
        model=LLM_MODEL,

        api_key=GROQ_API_KEY,

        # Low temperature for factual, grounded answers
        temperature=0.1,

    )

    # Build message list
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history)
    messages.append(HumanMessage(
        content=f"DOCUMENTS :\n{context}\n\nQUESTION : {question}"
    ))

    # Generate response
    response = llm.invoke(messages)

    # ---------------------------------------------------------
    # STEP 6: Save conversation memory
    # ---------------------------------------------------------
    _history.append(HumanMessage(content=question))
    _history.append(response)

    # ---------------------------------------------------------
    # STEP 7: Extract unique sources
    # ---------------------------------------------------------
    sources =[]
    seen = set()

    for chunk in final_chunks:
        key = (chunk["metadata"]["source"], chunk["metadata"]["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": chunk["metadata"]["source"],
                "path": chunk["metadata"].get("drive_path", chunk["metadata"]["source"]),
                "page": chunk["metadata"]["page"],
                "total_pages": chunk["metadata"].get("total_pages", "?"),
                "format": chunk["metadata"].get("file_format", "?"),

                # Prefer rerank score, fallback to retrieval score
                "score": round(chunk.get("rerank_score", chunk.get("score", 0)), 3)
            })

    return {"answer": response.content, "sources": sources}

def clear_memory():
    """
    Clear conversation memory.
    Useful to reset context between sessions.
    """

    global _history
    _history.clear()
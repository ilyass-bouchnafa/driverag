# Import Groq LLM (fast inference)
from langchain_groq import ChatGroq

# Import message schema for structured chat input
from langchain.schema import SystemMessage, HumanMessage

# Import project configuration
from src.config import GROQ_API_KEY, TOP_K_RETRIEVAL, TOP_K_RERANKED

# Import retrieval pipeline
from src.retrieval.query_processor import advanced_retrieve

# Import reranker
from src.retrieval.reranker import rerank


# ---------------------------------------------------------
# SYSTEM PROMPT (MULTILINGUAL CONTROL)
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are a rigorous and expert academic assistant.

STRICT RULES:
1. You MUST answer ONLY using the provided context documents.
2. For EVERY important statement, you MUST cite the source using this format: [File, Page X]
3. If the answer is not found in the documents, respond EXACTLY:
   "I cannot find this information in the provided documents."
4. NEVER invent information that is not present in the context.
5. Your answer MUST be in the SAME LANGUAGE as the retrieved documents (context).
6. If the question refers to previous conversation, take it into account.
7. Be structured, precise, and academic in your response.
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

def ask(question: str) -> dict:
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
    # STEP 4: Conversation history
    # ---------------------------------------------------------
    # Keep only the last 5 exchanges (10 messages total)
    history = _history[-10:]

    # ---------------------------------------------------------
    # STEP 5: LLM call (Groq)
    # ---------------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",

        api_key=GROQ_API_KEY,

        # Low temperature for factual, grounded answers
        temperature=0.1
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
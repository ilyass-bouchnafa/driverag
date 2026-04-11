# Import Groq chat model (fast and free alternative to Gemini)
from langchain_groq import ChatGroq

# Import message schema for LLM interaction
from langchain.schema import HumanMessage

# Import project configuration
from src.config import GROQ_API_KEY, MULTI_QUERY_COUNT

# Import hybrid search (used later in pipeline)
from src.retrieval.hybrid_search import hybrid_search

def generate_multi_queries(question: str) -> list[str]:
    """
    Generate multiple reformulations of a user query using an LLM.

    Why Multi-Query?
    ----------------
    - A single query may fail to retrieve relevant documents
    - Different phrasings improve recall in vector search
    - Helps mitigate embedding limitations

    Example:
    --------
    "How does RAG work?"
    → "What is Retrieval-Augmented Generation?"
    → "Explain the RAG pipeline step by step"
    → "What are the components of a RAG system?"

    Parameters
    ----------
    question : str
        The original user query

    Returns
    -------
    list[str]
        List of reformulated queries (including original query)
    """
    
    # ---------------------------------------------------------
    # STEP 1: Initialize the LLM (Groq)
    # ---------------------------------------------------------
    # Llama 3.1 is fast and free via Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",

        api_key=GROQ_API_KEY,

        # Higher temperature = more diverse reformulations
        temperature=0.7
    )

    # ---------------------------------------------------------
    # STEP 2: Build the prompt 
    # ---------------------------------------------------------
    prompt = prompt = f"""You are an expert information retrieval assistant.

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
        # ---------------------------------------------------------
        # STEP 3: Call the LLM
        # ---------------------------------------------------------
        response = llm.invoke([HumanMessage(content=prompt)])

        # ---------------------------------------------------------
        # STEP 4: Parse LLM output
        # ---------------------------------------------------------
        reformulations = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ][:MULTI_QUERY_COUNT]

        # ---------------------------------------------------------
        # STEP 5: Add original query
        # ---------------------------------------------------------
        all_queries = [question] + reformulations

        print(f"🔄 Multi-query: {len(all_queries)} variants generated")

        return all_queries

    except Exception as e:
        # ---------------------------------------------------------
        # FALLBACK (IMPORTANT for production systems)
        # ---------------------------------------------------------
        print(f"⚠️ LLM failed: {e}")

        return [
            question,
            question + " explanation",
            question + " definition"
        ]


def generate_hyde_document(question: str) -> str:
    """
    Generate a hypothetical document using HyDE (Hypothetical Document Embeddings).

    Why HyDE?
    ---------
    - User queries are often short and ambiguous
    - Documents in the vector store are long and information-dense
    - Generating a "fake but relevant" document bridges the semantic gap

    Idea:
    -----
    Instead of embedding the query directly, we:
    1. Generate a realistic document that answers the question
    2. Embed this document
    3. Retrieve similar documents from the vector database

    This significantly improves retrieval quality.

    Parameters
    ----------
    question : str
        The original user query

    Returns
    -------
    str
        A hypothetical document (~150-200 words)
    """
    # ---------------------------------------------------------
    # STEP 1: Initialize the LLM (Groq)
    # ---------------------------------------------------------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.1 #Low temperature → more factual and structured output
    )

    # ---------------------------------------------------------
    # STEP 2: Build the prompt 
    # ---------------------------------------------------------
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
        # ---------------------------------------------------------
        # STEP 3: Call the LLM
        # ---------------------------------------------------------
        response = llm.invoke([HumanMessage(prompt)])

        # ---------------------------------------------------------
        # STEP 4: Clean output
        # ---------------------------------------------------------
        hyde_doc = response.content.strip()

        print(f"📝 HyDE: generated hypothetical document ({len(hyde_doc.split())} words)")

        return hyde_doc
        
    except Exception as e:
        # ---------------------------------------------------------
        # FALLBACK (important for robustness)
        # ---------------------------------------------------------
        print(f"⚠️ HyDE generation failed: {e}")

        # Fallback: use the original question as a pseudo-document
        return question

def advanced_retrieve(question: str, k: int = 10) -> list[dict]:
    """
    Advanced retrieval pipeline combining Multi-Query, HyDE, Hybrid Search, and result fusion.

    Pipeline Overview:
    ------------------
    1. Generate multiple query reformulations (Multi-Query)
    2. Generate a hypothetical document (HyDE)
    3. Perform hybrid search for each variant
    4. Merge and deduplicate results
    5. Return top-k most relevant chunks

    Parameters
    ----------
    question : str
        The original user query

    k : int
        Number of final chunks to return after merging

    Returns
    -------
    list[dict]
        Top-k unique chunks aggregated from all retrieval strategies
    """

    # ---------------------------------------------------------
    # STEP 1: Generate query reformulations (Multi-Query)
    # ---------------------------------------------------------
    # Produces multiple variations of the original question
    # Improves recall in retrieval
    all_queries = generate_multi_queries(question)
    
    # ---------------------------------------------------------
    # STEP 2: Generate hypothetical document (HyDE)
    # ---------------------------------------------------------
    # Creates a dense, informative pseudo-document
    # Helps bridge the semantic gap between query and documents
    hyde_doc = generate_hyde_document(question)


    # ---------------------------------------------------------
    # STEP 3: Combine all query variants
    # ---------------------------------------------------------
    # Includes:
    # - Original query + reformulations
    # - HyDE-generated document
    all_variants = all_queries + [hyde_doc]

    # ---------------------------------------------------------
    # STEP 4: Perform hybrid search for each variant
    # ---------------------------------------------------------
    # Use a set to track unique chunks and avoid duplicates
    seen_keys = set()
    merged_chunks = []

    for variant in all_variants:

        # Perform hybrid search (BM25 + Dense retrieval)
        results = hybrid_search(variant)

        for chunk in results:
            # Create a unique identifier for each chunk
            key = (
                chunk["metadata"]["source"],
                chunk["metadata"]["page"],
                chunk["metadata"]["chunk_index"]
            )

            # Deduplicate chunks based on metadata
            if key not in seen_keys:
                seen_keys.add(key)
                merged_chunks.append(chunk)
    
    # ---------------------------------------------------------
    # STEP 5: Rank merged results
    # ---------------------------------------------------------
    # Sort chunks by combined score (descending)
    merged_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top-k most relevant chunks
    top_chunks = merged_chunks[:k]

    # ---------------------------------------------------------
    # STEP 6: Logging
    # ---------------------------------------------------------
    print(f"🔍 Advanced Retrieve : {len(top_chunks)} chunks uniques fusionnés")

    return top_chunks

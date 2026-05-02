# Import LangChain text splitter designed for semantic chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import chunk configuration values from the project settings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split extracted pages into smaller chunks while preserving metadata.

    Why chunking is necessary:
    --------------------------
    - Large Language Models have context size limits
    - Embeddings are more accurate on smaller text segments
    - Retrieval systems send only the most relevant chunks to the LLM

    RecursiveCharacterTextSplitter follows a progressive splitting strategy:
    paragraph → line → sentence → word

    Parameters
    ----------
    pages : list[dict]
        List of extracted pages produced by file_router.py

    Returns
    -------
    list[dict]
        A list of structured chunks containing:
        - text
        - metadata
    """

    # ---------------------------------------------------------
    # STEP 1: Initialize the text splitter
    # ---------------------------------------------------------
    # RecursiveCharacterTextSplitter tries to preserve semantic coherence
    # by splitting text using natural separators before forcing cuts.
    splitter = RecursiveCharacterTextSplitter(
        # Maximum size allowed for each chunk
        chunk_size=CHUNK_SIZE,

        # Number of overlapping characters shared between chunks
        # This helps preserve context continuity
        chunk_overlap=CHUNK_OVERLAP,

        # Ordered splitting strategy:
        # Try paragraphs first, then lines, then sentences, then words
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    # Initialize the final list that will store all generated chunks
    chunks = []

    # ---------------------------------------------------------
    # STEP 2: Process each extracted page
    # ---------------------------------------------------------
    for page in pages:

        # Split the page text into smaller sub-chunks
        sub_chunks = splitter.split_text(page["text"])

        # ---------------------------------------------------------
        # STEP 3: Process each generated sub-chunk
        # ---------------------------------------------------------
        for i, chunk_text in enumerate(sub_chunks):

            # Ignore empty chunks
            if chunk_text.strip(): 

                # Store chunk text and full metadata
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": i,
                        "total_pages": page["total_pages"],
                        "drive_path": page.get("drive_path", page["source"]),
                        "file_format": page.get("file_format", "unknown"),
                        "drive_modified_time": page.get("drive_modified_time", ""),  # ← AJOUT
                    }
                })

    # ---------------------------------------------------------
    # STEP 4: Display chunking summary
    # ---------------------------------------------------------
    print(f"✂️  {len(chunks)} chunks créés")

    # Return all structured chunks
    return chunks
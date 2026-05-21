# import io
import tempfile
import os

# Import document loaders from LangChain # Each loader is specialized for one document format
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    # UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)

def extract_text_from_bytes(file_bytes: bytes, file_name: str, file_format: str) -> list[dict]:
    """ 
    Extract text content from a file according to its format. 
    Strategy: 
    --------- 
    Since most LangChain loaders expect a physical file path, 
    the file is first written into a temporary file.

    The appropriate loader is then selected depending on the format. After extraction: 
    - text content is collected 
    - page numbers are preserved when possible 
    - the temporary file is deleted

    Parameters 
    ---------- 
    file_bytes : bytes 
        Binary content of the file. 
        
    file_name : str 
        Original file name used for metadata tracking. 
    
    file_format : str 
        File format identifier: "pdf", "docx", "txt", "md", "pptx"

    Returns 
    ------- 
    list[dict] 
        A list of dictionaries containing: 
        - text 
        - page 
        - source 
        - total_pages 
    """

    # Initialize the result list 
    # Each extracted page/section will be stored here
    pages = []

    # ---------------------------------------------------------
    # STEP 1: Create a temporary file 
    # ---------------------------------------------------------
    # LangChain loaders require a real file path, 
    # so the bytes must first be written to disk temporarily.

    # Build the file extension dynamically
    suffix = f".{file_format}"

    # Create a temporary file with the correct extension
    # delete=False ensures the file remains accessible after closing
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:

        # Write the binary content into the temporary file
        tmp.write(file_bytes)

        # Store the generated temporary path
        tmp_path = tmp.name
    
    try:

        # ---------------------------------------------------------
        # STEP 2: Process PDF files
        # ---------------------------------------------------------
        if file_format == "pdf":

            # Use PyPDFLoader to split the PDF page by page
            loader = PyPDFLoader(tmp_path)

            # Load all pages into LangChain documents
            docs = loader.load()

            # Iterate through each extracted page
            for doc in docs:

                # Ignore empty pages
                if doc.page_content.strip():

                    # Store page content and metadata
                    pages.append({
                        "text": doc.page_content.strip(),

                        # LangChain starts page numbering at 0 so +1 gives human-readable numbering
                        # if no page found it take 0 in default
                        "page": doc.metadata.get("page", 0) + 1,

                        "source": file_name,

                        # Store total number of pages
                        "total_pages": len(docs)
                    })

        # ---------------------------------------------------------
        # STEP 3: Process DOCX and Google Docs exported files
        # ---------------------------------------------------------
        elif file_format in ("docs", "gdoc"):

            # Use Docx2txtLoader for Word files
            loader = Docx2txtLoader(tmp_path)

            docs = loader.load()

            # DOCX does not preserve real page numbers,
            # so sections are indexed manually
            for i, doc in enumerate(docs):

                if doc.page_content.strip():

                    pages.append({
                        "text": doc.page_content.strip(),

                        # Simulated page numbering using section index
                        "page": i + 1,

                        "source": file_name,

                        "total_pages": len(docs)
                    })
        
        # ---------------------------------------------------------
        # STEP 4: Process TXT and Markdown files
        # ---------------------------------------------------------
        elif file_format in ("text", "md"):

            # Load plain text using UTF-8 encoding
            loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()

            # TXT and Markdown files do not contain pages,
            # so everything is merged into one page
            full_text = "\n".join([d.page_content for d in docs])

            if full_text.strip():

                pages.append({

                    "text": full_text.strip(),

                    "page": 1,

                    "source": file_name,

                    "total_pages": 1
                })
        
        # ---------------------------------------------------------
        # STEP 5: Process PowerPoint files
        # ---------------------------------------------------------
        elif file_format == "pptx":

            # Use PowerPoint loader
            loader = UnstructuredPowerPointLoader(tmp_path)

            docs = loader.load()

            # Each slide is treated as one page
            for i, doc in enumerate(docs):

                if doc.page_content.strip():

                    pages.append({
                        "text": doc.page_content.strip(),

                        # Slide numbering starts at 1
                        "page": i + 1,

                        "source": file_name,

                        "total_pages": len(docs)
                    })

    finally:

        # ---------------------------------------------------------
        # STEP 6: Delete temporary file
        # ---------------------------------------------------------
        # Always remove the temporary file,
        # even if an error occurs during extraction
        os.unlink(tmp_path)
    
    # ---------------------------------------------------------
    # STEP 7: Display extraction summary
    # ---------------------------------------------------------
    print(f"📄 {file_name} ({file_format}): {len(pages)} page(s)/section(s) extracted")

    # Return structured extracted content
    return pages




                
            
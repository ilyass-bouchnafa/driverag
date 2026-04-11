from src.ingestion.file_router import extract_text_from_bytes

# Open a local PDF file in binary mode
with open("test.pdf", "rb") as f:
    file_bytes = f.read()

# Call the extraction function
result = extract_text_from_bytes(
    file_bytes=file_bytes,
    file_name="test.pdf",
    file_format="pdf"
)

# Display extracted pages
for page in result:
    print(page)


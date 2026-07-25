from pypdf import PdfReader

def get_pdf_text(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but overlap a bit
    return chunks

# Test it
text = get_pdf_text("Intelli-Helmet_CSP.pdf")
chunks = chunk_text(text)

print(f"Total chunks created: {len(chunks)}")
print("\n--- First chunk ---\n")
print(chunks[0])
print("\n--- Second chunk (notice the overlap) ---\n")
print(chunks[1])
print("\n--- Third chunk (notice the overlap) ---\n")
print(chunks[2])
print("\n--- fourth chunk (notice the overlap) ---\n")
print(chunks[3])

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# --- Reuse our earlier functions ---
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
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Step 1: Extract + chunk the PDF ---
text = get_pdf_text("Intelli-Helmet_CSP.pdf")
chunks = chunk_text(text)
print(f"Created {len(chunks)} chunks")

# --- Step 2: Load embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 3: Set up ChromaDB (local, saved to disk) ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documind")

# --- Step 4: Embed and store each chunk ---
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )

print(f"Stored {len(chunks)} chunks in ChromaDB!")
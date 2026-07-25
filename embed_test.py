from sentence_transformers import SentenceTransformer

# Load a small, fast embedding model (downloads once, then cached locally)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Try embedding two similar sentences and one different one
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Stock prices fell sharply today"
]

embeddings = model.encode(sentences)

print(f"Shape of embeddings: {embeddings.shape}")
print(f"First 10 numbers of sentence 1's embedding:\n{embeddings[0][:10]}")
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

# Setup
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="documind")

def get_relevant_chunks(question, n_results=3):
    query_embedding = embed_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results['documents'][0]

def ask_rag(question):
    # Step 1: Retrieve relevant chunks
    chunks = get_relevant_chunks(question)
    context = "\n\n".join(chunks)

    # Step 2: Build augmented prompt
    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have that information in this document."

Context:
{context}

Question: {question}

Answer:"""

    # Step 3: Send to Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# --- Chat loop ---
print("DocuMind ready! Ask questions about your PDF. Type 'exit' to quit.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = ask_rag(question)
    print(f"\nDocuMind: {answer}\n")
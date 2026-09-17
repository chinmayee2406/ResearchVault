import chromadb
import ollama

from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)


# -----------------------------
# 2. Load embedding model
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# 3. User question
# -----------------------------

query = input("\nAsk a question about the paper: ")


# -----------------------------
# 4. Convert question to embedding
# -----------------------------

query_embedding = model.encode(query)


# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

# -----------------------------
# DEBUG: Inspect retrieved chunks
# -----------------------------

print("\nRETRIEVED CHUNKS:")

for i in range(5):

    print("\n" + "=" * 60)

    print("Rank:", i + 1)

    print(
        "Paper:",
        results["metadatas"][0][i]["paper"]
    )

    print(
        "Page:",
        results["metadatas"][0][i]["page"]
    )

    print(
        "Chunk ID:",
        results["metadatas"][0][i]["chunk_id"]
    )

    print("\nText:")

    print(
        results["documents"][0][i]
    )
# -----------------------------
# 6. Build context
# -----------------------------

context_parts = []

for i in range(5):

    paper = results["metadatas"][0][i]["paper"]
    page = results["metadatas"][0][i]["page"]
    text = results["documents"][0][i]

    context_parts.append(
        f"[Paper: {paper}, Page: {page}]\n{text}"
    )


context = "\n\n".join(context_parts)


# -----------------------------
# 7. Build RAG prompt
# -----------------------------

prompt = f"""
You are a research assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided paper."

Context:
{context}

Question:
{query}

Answer:
"""


# -----------------------------
# 8. Generate answer locally
# -----------------------------

response = ollama.chat(
    model="lfm2.5:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 9. Display answer
# -----------------------------

print("\n" + "=" * 60)

print("ANSWER:")

print(response["message"]["content"])

print("=" * 60)


# -----------------------------
# 10. Display sources
# -----------------------------

print("\nSOURCES:")

for i in range(5):

    metadata = results["metadatas"][0][i]

    print(
        f"- {metadata['paper']} "
        f"(Page {metadata['page']}, "
        f"Chunk {metadata['chunk_id']})"
    )
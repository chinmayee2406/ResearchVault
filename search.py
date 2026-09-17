import chromadb

from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


query = "What is NeMo Guardrails?"

query_embedding = model.encode(query)


results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)


print("\nQuery:")
print(query)

print("\nTop results:")

for i in range(3):

    print("\n" + "=" * 60)

    print("Rank:", i + 1)

    print("Distance:", results["distances"][0][i])

    print("Paper:", results["metadatas"][0][i]["paper"])

    print("Page:", results["metadatas"][0][i]["page"])

    print("Chunk ID:", results["metadatas"][0][i]["chunk_id"])

    print("Text:")
    print(results["documents"][0][i][:500])
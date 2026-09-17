import chromadb

client = chromadb.PersistentClient(path="data/chroma")

collection = client.get_or_create_collection(
    name="research_papers"
)

results = collection.get(
    ids=["chunk_0"]
)

print("Retrieved document:")
print(results["documents"][0])

print("\nMetadata:")
print(results["metadatas"][0])
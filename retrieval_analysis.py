import chromadb
from sentence_transformers import SentenceTransformer


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)


# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Test questions
questions = [
    "What is NeMo Guardrails?",
    "What are the five reference Guardrails applications?",
    "What is Colang?",
    "What are the main concepts in Colang?"
]


# Test each question
for query in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(query)

    # Convert question into embedding
    query_embedding = model.encode(query)

    # Retrieve top 5 chunks
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    print("\nRETRIEVED CHUNKS:")

    for i in range(5):

        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print("\n" + "-" * 60)

        print("Rank:", i + 1)
        print("Distance:", round(distance, 4))
        print("Page:", metadata["page"])
        print("Chunk ID:", metadata["chunk_id"])
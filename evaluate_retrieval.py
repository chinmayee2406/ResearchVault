import chromadb
from sentence_transformers import SentenceTransformer

from evaluation_dataset import evaluation_questions


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


reciprocal_ranks = []


for item in evaluation_questions:

    query = item["question"]
    relevant_page = item["relevant_page"]

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    retrieved_pages = [
        metadata["page"]
        for metadata in results["metadatas"][0]
    ]

    rank = None

    for i, page in enumerate(retrieved_pages):

        if page == relevant_page:
            rank = i + 1
            break

    if rank is not None:

        reciprocal_rank = 1 / rank

    else:

        reciprocal_rank = 0

    reciprocal_ranks.append(reciprocal_rank)

    print("\n" + "=" * 60)
    print("Question:", query)
    print("Expected page:", relevant_page)
    print("Retrieved pages:", retrieved_pages)
    print("Relevant page rank:", rank)
    print("Reciprocal rank:", round(reciprocal_rank, 2))


mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


print("\n" + "=" * 60)
print("MRR:", round(mrr, 2))
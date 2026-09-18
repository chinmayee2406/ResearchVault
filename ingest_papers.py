import pymupdf
from sentence_transformers import SentenceTransformer

from ingestion.chunker import load_chunks
from embeddings.vector_store import create_collection, add_documents


def main():

    print("Loading papers...")

    chunks = load_chunks()

    print(
        "Total chunks:",
        len(chunks)
    )

    print("\nChunks by paper:")

    paper_counts = {}

    for chunk in chunks:

        paper = chunk["paper"]

        paper_counts[paper] = (
            paper_counts.get(paper, 0) + 1
        )

    for paper, count in paper_counts.items():

        print(
            f"- {paper}: {count}"
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts
    )

    print(
        "Embeddings generated:",
        len(embeddings)
    )

    collection = create_collection()

    print(
        "\nClearing old ChromaDB data..."
    )

    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    add_documents(
        collection,
        chunks,
        embeddings
    )

    print(
        "\nDocuments stored in ChromaDB!"
    )

    print(
        "Collection size:",
        collection.count()
    )


if __name__ == "__main__":

    main()
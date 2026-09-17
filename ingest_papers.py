import pymupdf
from sentence_transformers import SentenceTransformer

from embeddings.vector_store import create_collection, add_documents


PDF_PATH = "data/papers/attention.pdf"

CHUNK_SIZE = 500
OVERLAP = 50


def load_chunks():

    document = pymupdf.open(PDF_PATH)

    all_chunks = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if not text:
            continue

        words = text.split()

        step = CHUNK_SIZE - OVERLAP

        for i in range(0, len(words), step):

            chunk = words[i:i + CHUNK_SIZE]

            if chunk:

                all_chunks.append({
                    "paper": "NeMo Guardrails",
                    "page": page_number + 1,
                    "chunk_id": len(all_chunks),
                    "text": " ".join(chunk)
                })

    return all_chunks


def main():

    print("Loading paper...")

    chunks = load_chunks()

    print("Total chunks:", len(chunks))

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    print("Embeddings generated:", len(embeddings))

    collection = create_collection()

    add_documents(
        collection,
        chunks,
        embeddings
    )

    print("Documents stored in ChromaDB!")
    print("Collection size:", collection.count())


if __name__ == "__main__":
    main()
import pymupdf
from rank_bm25 import BM25Okapi
import re


PDF_PATH = "data/papers/attention.pdf"

CHUNK_SIZE = 500
OVERLAP = 50


def tokenize(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    return text.split()


def load_chunks():

    document = pymupdf.open(PDF_PATH)

    chunks = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if not text:
            continue

        words = text.split()

        step = CHUNK_SIZE - OVERLAP

        for i in range(0, len(words), step):

            chunk = words[i:i + CHUNK_SIZE]

            if chunk:

                chunks.append({
                    "paper": "NeMo Guardrails",
                    "page": page_number + 1,
                    "chunk_id": len(chunks),
                    "text": " ".join(chunk)
                })

    return chunks


# Load paper chunks
chunks = load_chunks()

print("Total chunks:", len(chunks))


# Tokenize chunks
tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]


# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)


# Query
query = "What is Colang?"

query_tokens = tokenize(query)


# Calculate scores
scores = bm25.get_scores(query_tokens)


# Get top 5 results
top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]


print("\nQuery:")
print(query)

print("\nTop 5 BM25 results:")


for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print("\n" + "=" * 60)

    print("Rank:", rank)
    print("BM25 score:", round(scores[index], 4))
    print("Page:", chunk["page"])
    print("Chunk ID:", chunk["chunk_id"])

    print("\nText:")
    print(chunk["text"][:500])
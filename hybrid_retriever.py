import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PDF_PATH = "data/papers/attention.pdf"

CHUNK_SIZE = 500
OVERLAP = 50


def tokenize(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    return text.split()


def load_chunks():

    import pymupdf

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


# Load chunks
chunks = load_chunks()

print("Total chunks:", len(chunks))


# Load semantic search
client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Build BM25 index
tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# Query
query = "What is Colang?"


# --------------------------------
# Semantic retrieval
# --------------------------------

query_embedding = embedding_model.encode(query)

semantic_results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

print("\n" + "=" * 70)
print("SEMANTIC RESULTS")

for i in range(5):

    metadata = semantic_results["metadatas"][0][i]

    print(
        f"Rank {i + 1} | "
        f"Page {metadata['page']} | "
        f"Chunk {metadata['chunk_id']} | "
        f"Distance {semantic_results['distances'][0][i]:.4f}"
    )


# --------------------------------
# BM25 retrieval
# --------------------------------

query_tokens = tokenize(query)

bm25_scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]

print("\n" + "=" * 70)
print("BM25 RESULTS")

for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print(
        f"Rank {rank} | "
        f"Page {chunk['page']} | "
        f"Chunk {chunk['chunk_id']} | "
        f"BM25 Score {bm25_scores[index]:.4f}"
    )
print("\n" + "=" * 70)
print("RANKING COMPARISON")

semantic_ranks = {}

for rank in range(5):

    metadata = semantic_results["metadatas"][0][rank]

    chunk_id = metadata["chunk_id"]

    semantic_ranks[chunk_id] = rank + 1


bm25_ranks = {}

for rank, index in enumerate(top_indices, start=1):

    chunk_id = chunks[index]["chunk_id"]

    bm25_ranks[chunk_id] = rank


all_chunk_ids = set(semantic_ranks) | set(bm25_ranks)

for chunk_id in sorted(all_chunk_ids):

    semantic_rank = semantic_ranks.get(chunk_id, "-")
    bm25_rank = bm25_ranks.get(chunk_id, "-")

    print(
        f"Chunk {chunk_id}: "
        f"Semantic Rank = {semantic_rank}, "
        f"BM25 Rank = {bm25_rank}"
    )
# --------------------------------
# Reciprocal Rank Fusion
# --------------------------------

RRF_K = 60

rrf_scores = {}

for chunk_id in all_chunk_ids:

    semantic_rank = semantic_ranks.get(chunk_id)
    bm25_rank = bm25_ranks.get(chunk_id)

    score = 0

    if semantic_rank is not None:
        score += 1 / (RRF_K + semantic_rank)

    if bm25_rank is not None:
        score += 1 / (RRF_K + bm25_rank)

    rrf_scores[chunk_id] = score


# Sort chunks by RRF score

hybrid_results = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)


print("\n" + "=" * 70)
print("HYBRID RESULTS — RRF")

for rank, (chunk_id, score) in enumerate(
    hybrid_results[:5],
    start=1
):

    chunk = chunks[chunk_id]

    print(
        f"Rank {rank} | "
        f"Page {chunk['page']} | "
        f"Chunk {chunk_id} | "
        f"RRF Score {score:.6f}"
    )
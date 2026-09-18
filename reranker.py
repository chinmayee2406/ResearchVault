import re

import chromadb
import pymupdf

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


PDF_PATH = "data/papers/attention.pdf"

CHUNK_SIZE = 500
OVERLAP = 50

RRF_K = 60


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


# --------------------------------
# Load chunks
# --------------------------------

chunks = load_chunks()

print("Total chunks:", len(chunks))


# --------------------------------
# Semantic retrieval
# --------------------------------

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# BM25
# --------------------------------

tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# --------------------------------
# Cross-Encoder
# --------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------
# Query
# --------------------------------

query = "What is Colang?"


# --------------------------------
# Semantic retrieval
# --------------------------------

query_embedding = embedding_model.encode(query)

semantic_results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10
)

semantic_ranks = {}

for rank in range(10):

    metadata = semantic_results["metadatas"][0][rank]

    chunk_id = metadata["chunk_id"]

    semantic_ranks[chunk_id] = rank + 1


# --------------------------------
# BM25 retrieval
# --------------------------------

query_tokens = tokenize(query)

bm25_scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:10]

bm25_ranks = {}

for rank, index in enumerate(top_indices, start=1):

    chunk_id = chunks[index]["chunk_id"]

    bm25_ranks[chunk_id] = rank


# --------------------------------
# RRF hybrid retrieval
# --------------------------------

all_chunk_ids = set(
    semantic_ranks
) | set(
    bm25_ranks
)

rrf_scores = {}

for chunk_id in all_chunk_ids:

    score = 0

    semantic_rank = semantic_ranks.get(chunk_id)
    bm25_rank = bm25_ranks.get(chunk_id)

    if semantic_rank is not None:

        score += 1 / (RRF_K + semantic_rank)

    if bm25_rank is not None:

        score += 1 / (RRF_K + bm25_rank)

    rrf_scores[chunk_id] = score


hybrid_candidates = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]


print("\n" + "=" * 70)
print("HYBRID CANDIDATES")

for rank, (chunk_id, score) in enumerate(
    hybrid_candidates,
    start=1
):

    chunk = chunks[chunk_id]

    print(
        f"Rank {rank} | "
        f"Page {chunk['page']} | "
        f"Chunk {chunk_id} | "
        f"RRF {score:.6f}"
    )


# --------------------------------
# Cross-Encoder reranking
# --------------------------------

candidate_pairs = []

for chunk_id, score in hybrid_candidates:

    candidate_pairs.append([
        query,
        chunks[chunk_id]["text"]
    ])


reranker_scores = reranker.predict(
    candidate_pairs
)


reranked_results = sorted(
    zip(
        hybrid_candidates,
        reranker_scores
    ),
    key=lambda item: float(item[1]),
    reverse=True
)


print("\n" + "=" * 70)
print("RERANKED RESULTS")

for rank, ((chunk_id, rrf_score), reranker_score) in enumerate(
    reranked_results[:5],
    start=1
):

    chunk = chunks[chunk_id]

    print(
        f"Rank {rank} | "
        f"Page {chunk['page']} | "
        f"Chunk {chunk_id} | "
        f"Reranker Score {float(reranker_score):.4f}"
    )
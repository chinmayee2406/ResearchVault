import re

import chromadb
import pymupdf

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from evaluation_dataset import evaluation_questions


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
# Semantic search
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
# Evaluation
# --------------------------------

hybrid_reciprocal_ranks = []
reranked_reciprocal_ranks = []


for item in evaluation_questions:

    query = item["question"]
    relevant_page = item["relevant_page"]

    print("\n" + "=" * 70)
    print("QUESTION:", query)
    print("EXPECTED PAGE:", relevant_page)


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


    hybrid_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]


    # --------------------------------
    # Hybrid MRR
    # --------------------------------

    hybrid_pages = [
        chunks[chunk_id]["page"]
        for chunk_id, score in hybrid_results[:5]
    ]


    hybrid_rank = None

    for rank, page in enumerate(
        hybrid_pages,
        start=1
    ):

        if page == relevant_page:

            hybrid_rank = rank
            break


    if hybrid_rank is not None:

        hybrid_rr = 1 / hybrid_rank

    else:

        hybrid_rr = 0


    hybrid_reciprocal_ranks.append(
        hybrid_rr
    )


    # --------------------------------
    # Cross-Encoder reranking
    # --------------------------------

    candidate_pairs = [
        [
            query,
            chunks[chunk_id]["text"]
        ]
        for chunk_id, score in hybrid_results
    ]


    reranker_scores = reranker.predict(
        candidate_pairs
    )


    reranked_results = sorted(
        zip(
            hybrid_results,
            reranker_scores
        ),
        key=lambda item: float(item[1]),
        reverse=True
    )


    reranked_pages = [
        chunks[chunk_id]["page"]
        for (
            (chunk_id, rrf_score),
            reranker_score
        ) in reranked_results[:5]
    ]


    # --------------------------------
    # Reranked MRR
    # --------------------------------

    reranked_rank = None

    for rank, page in enumerate(
        reranked_pages,
        start=1
    ):

        if page == relevant_page:

            reranked_rank = rank
            break


    if reranked_rank is not None:

        reranked_rr = 1 / reranked_rank

    else:

        reranked_rr = 0


    reranked_reciprocal_ranks.append(
        reranked_rr
    )


    print("Hybrid pages:", hybrid_pages)
    print("Hybrid relevant rank:", hybrid_rank)

    print("Reranked pages:", reranked_pages)
    print("Reranked relevant rank:", reranked_rank)


# --------------------------------
# Final metrics
# --------------------------------

hybrid_mrr = (
    sum(hybrid_reciprocal_ranks)
    / len(hybrid_reciprocal_ranks)
)


reranked_mrr = (
    sum(reranked_reciprocal_ranks)
    / len(reranked_reciprocal_ranks)
)


hybrid_recall = sum(
    1
    for rr in hybrid_reciprocal_ranks
    if rr > 0
) / len(hybrid_reciprocal_ranks)


reranked_recall = sum(
    1
    for rr in reranked_reciprocal_ranks
    if rr > 0
) / len(reranked_reciprocal_ranks)


print("\n" + "=" * 70)

print(
    "HYBRID RECALL@5:",
    round(hybrid_recall, 2)
)

print(
    "RERANKED RECALL@5:",
    round(reranked_recall, 2)
)

print(
    "HYBRID MRR:",
    round(hybrid_mrr, 2)
)

print(
    "RERANKED MRR:",
    round(reranked_mrr, 2)
)
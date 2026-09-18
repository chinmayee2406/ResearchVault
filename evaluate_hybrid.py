import re

import chromadb
import pymupdf

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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
# Load data
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
# Evaluation
# --------------------------------

reciprocal_ranks = []

for item in evaluation_questions:

    query = item["question"]
    relevant_page = item["relevant_page"]

    print("\n" + "=" * 70)
    print("QUESTION:", query)
    print("EXPECTED PAGE:", relevant_page)

    # Semantic retrieval

    query_embedding = embedding_model.encode(query)

    semantic_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    semantic_ranks = {}

    for rank in range(5):

        metadata = semantic_results["metadatas"][0][rank]

        chunk_id = metadata["chunk_id"]

        semantic_ranks[chunk_id] = rank + 1


    # BM25 retrieval

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

    bm25_ranks = {}

    for rank, index in enumerate(top_indices, start=1):

        chunk_id = chunks[index]["chunk_id"]

        bm25_ranks[chunk_id] = rank


    # RRF

    all_chunk_ids = set(semantic_ranks) | set(bm25_ranks)

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
    )


    retrieved_pages = [
        chunks[chunk_id]["page"]
        for chunk_id, score in hybrid_results[:5]
    ]


    # Check whether relevant page was retrieved

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


    print("HYBRID PAGES:", retrieved_pages)
    print("RELEVANT PAGE RANK:", rank)


# --------------------------------
# Final metrics
# --------------------------------

recall_at_5 = sum(
    1
    for item, rr in zip(
        evaluation_questions,
        reciprocal_ranks
    )
    if rr > 0
) / len(evaluation_questions)


mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


print("\n" + "=" * 70)
print("HYBRID RECALL@5:", round(recall_at_5, 2))
print("HYBRID MRR:", round(mrr, 2))
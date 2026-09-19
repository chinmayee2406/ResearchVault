import re

import chromadb

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from evaluation_dataset import evaluation_questions


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)


# --------------------------------------------------
# Models
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------------------------
# BM25 tokenizer
# --------------------------------------------------

def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.split()


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k=60
):

    scores = {}

    for rank, chunk_id in enumerate(
        semantic_results,
        start=1
    ):

        scores[chunk_id] = scores.get(
            chunk_id,
            0
        ) + 1 / (k + rank)

    for rank, chunk_id in enumerate(
        bm25_results,
        start=1
    ):

        scores[chunk_id] = scores.get(
            chunk_id,
            0
        ) + 1 / (k + rank)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk_id
        for chunk_id, score in ranked
    ]


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def calculate_recall_at_5(
    results,
    gold_chunks
):

    top_results = results[:5]

    for result in top_results:

        if result["chunk_id"] in gold_chunks:

            return 1

    return 0


def calculate_mrr(
    results,
    gold_chunks
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["chunk_id"] in gold_chunks:

            return 1 / rank

    return 0


# --------------------------------------------------
# Retrieval for one question
# --------------------------------------------------

def retrieve(
    query,
    paper
):

    paper_data = collection.get(
        where={
            "paper": paper
        }
    )

    documents = paper_data["documents"]

    metadatas = paper_data["metadatas"]

    ids = paper_data["ids"]


    # ----------------------------------------------
    # Semantic retrieval
    # ----------------------------------------------

    query_embedding = embedding_model.encode(
        query
    )

    document_embeddings = embedding_model.encode(
        documents
    )

    semantic_scores = []

    for chunk_id, embedding in zip(
        ids,
        document_embeddings
    ):

        score = embedding_model.similarity(
            query_embedding,
            embedding
        )

        semantic_scores.append(
            (
                chunk_id,
                float(score)
            )
        )

    semantic_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    semantic_ids = [
        chunk_id
        for chunk_id, score in semantic_scores[:10]
    ]


    # ----------------------------------------------
    # BM25 retrieval
    # ----------------------------------------------

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    bm25_scores = bm25.get_scores(
        tokenize(query)
    )

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    bm25_ids = [
        ids[i]
        for i in bm25_indices
    ]


    # ----------------------------------------------
    # Hybrid retrieval
    # ----------------------------------------------

    hybrid_ids = reciprocal_rank_fusion(
        semantic_ids,
        bm25_ids
    )[:10]


    # ----------------------------------------------
    # Cross-encoder reranking
    # ----------------------------------------------

    id_to_document = dict(
        zip(
            ids,
            documents
        )
    )

    id_to_metadata = dict(
        zip(
            ids,
            metadatas
        )
    )

    candidates = []

    for chunk_id in hybrid_ids:

        candidates.append({
            "id": chunk_id,
            "text": id_to_document[chunk_id],
            "metadata": id_to_metadata[chunk_id]
        })


    pairs = [
        (
            query,
            candidate["text"]
        )
        for candidate in candidates
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    reranked = sorted(
        zip(
            candidates,
            rerank_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )


    # ----------------------------------------------
    # Build semantic results
    # ----------------------------------------------

    semantic_results = []

    for chunk_id, score in semantic_scores:

        semantic_results.append({
            "chunk_id": chunk_id,
            "page": id_to_metadata[chunk_id]["page"]
        })


    # ----------------------------------------------
    # Build BM25 results
    # ----------------------------------------------

    bm25_results = []

    for chunk_id in bm25_ids:

        bm25_results.append({
            "chunk_id": chunk_id,
            "page": id_to_metadata[chunk_id]["page"]
        })


    # ----------------------------------------------
    # Build hybrid results
    # ----------------------------------------------

    hybrid_results = []

    for chunk_id in hybrid_ids:

        hybrid_results.append({
            "chunk_id": chunk_id,
            "page": id_to_metadata[chunk_id]["page"]
        })


    # ----------------------------------------------
    # Build reranked results
    # ----------------------------------------------

    reranked_results = []

    for candidate, score in reranked:

        reranked_results.append({
            "chunk_id": candidate["id"],
            "page": candidate["metadata"]["page"]
        })


    return {
        "semantic": semantic_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results,
        "reranked": reranked_results
    }


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

metrics = {
    "semantic": {
        "recall": [],
        "mrr": []
    },
    "bm25": {
        "recall": [],
        "mrr": []
    },
    "hybrid": {
        "recall": [],
        "mrr": []
    },
    "reranked": {
        "recall": [],
        "mrr": []
    }
}


print(
    "\nRunning ResearchVault retrieval evaluation..."
)


for item in evaluation_questions:

    question = item["question"]

    paper = item["paper"]

    gold_chunks = item["gold_chunks"]


    results = retrieve(
        question,
        paper
    )


    print(
        f"\nQuestion: {question}"
    )

    print(
        f"Paper: {paper}"
    )

    print(
        f"Gold chunks: {gold_chunks}"
    )


    # ----------------------------------------------
    # Calculate and store metrics
    # ----------------------------------------------

    for method in metrics:

        recall = calculate_recall_at_5(
            results[method],
            gold_chunks
        )

        mrr = calculate_mrr(
            results[method],
            gold_chunks
        )


        print(
            f"{method.upper():10} "
            f"Recall@5={recall:.0f} "
            f"MRR={mrr:.3f}"
        )


        # Store metrics for overall evaluation

        metrics[method]["recall"].append(
            recall
        )

        metrics[method]["mrr"].append(
            mrr
        )


# --------------------------------------------------
# Print overall results
# --------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "RESEARCHVAULT RETRIEVAL EVALUATION"
)

print(
    "=" * 70
)


for method in metrics:

    recall = sum(
        metrics[method]["recall"]
    ) / len(
        metrics[method]["recall"]
    )

    mrr = sum(
        metrics[method]["mrr"]
    ) / len(
        metrics[method]["mrr"]
    )


    print(
        f"\n{method.upper()}"
    )

    print(
        f"Recall@5: {recall:.3f}"
    )

    print(
        f"MRR:      {mrr:.3f}"
    )
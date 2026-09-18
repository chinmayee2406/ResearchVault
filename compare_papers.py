import re

import chromadb
import ollama

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


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
# Query
# --------------------------------------------------

query = input(
    "\nWhat would you like to compare? "
)


# --------------------------------------------------
# Query embedding
# --------------------------------------------------

query_embedding = embedding_model.encode(
    query
)


# --------------------------------------------------
# Papers
# --------------------------------------------------

papers = [
    "RAG",
    "attention"
]


final_results = []


# --------------------------------------------------
# Retrieve evidence from each paper
# --------------------------------------------------

for paper in papers:

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

    tokenized_query = tokenize(
        query
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    bm25_ids = [
        ids[i]
        for i in bm25_ranked_indices
    ]


    # ----------------------------------------------
    # Hybrid retrieval
    # ----------------------------------------------

    hybrid_ids = reciprocal_rank_fusion(
        semantic_ids,
        bm25_ids
    )

    hybrid_ids = hybrid_ids[:10]


    # ----------------------------------------------
    # Candidate documents
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


    # ----------------------------------------------
    # Cross-encoder reranking
    # ----------------------------------------------

    pairs = [
        (
            query,
            candidate["text"]
        )
        for candidate in candidates
    ]

    if not pairs:
        continue

    rerank_scores = reranker.predict(
        pairs
    )

    ranked_candidates = sorted(
        zip(
            candidates,
            rerank_scores
        ),
        key=lambda x: x[1],
        reverse=True
    )


    # ----------------------------------------------
    # Keep top 5 evidence chunks
    # ----------------------------------------------

    for candidate, score in ranked_candidates[:5]:

        final_results.append({
            "paper": candidate["metadata"]["paper"],
            "page": candidate["metadata"]["page"],
            "chunk_id": candidate["metadata"]["chunk_id"],
            "score": float(score),
            "text": candidate["text"]
        })


# --------------------------------------------------
# Build comparison evidence
# --------------------------------------------------

evidence_context = ""

evidence_number = 1

for result in final_results:

    result["evidence_id"] = evidence_number

    evidence_context += (
        f"\n[EVIDENCE {evidence_number}]\n"
        f"Paper: {result['paper']}\n"
        f"Page: {result['page']}\n"
        f"Evidence text:\n"
        f"{result['text']}\n"
    )

    evidence_number += 1


# --------------------------------------------------
# Generate structured comparison
# --------------------------------------------------

def generate_paper_summary(paper_name, evidence):

    prompt = f"""
You are a research assistant analyzing ONE research paper.

Paper: {paper_name}

Use ONLY the evidence below.

Evidence:
{evidence}

Answer the following five aspects:

Problem addressed:
Main approach:
Key mechanism:
Benefits:
Limitations:

Rules:
- Use only information explicitly supported by the evidence.
- Do not use outside knowledge.
- Do not mention other papers.
- Keep every answer to one concise sentence.
- If the evidence does not support an aspect, write:
  Not covered in retrieved evidence.

Return ONLY these five lines in exactly this format:

Problem addressed: ...
Main approach: ...
Key mechanism: ...
Benefits: ...
Limitations: ...
"""

    response = ollama.chat(
        model="lfm2.5:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Separate evidence by paper
# --------------------------------------------------

rag_evidence = ""

nemo_evidence = ""

for result in final_results:

    if result["paper"] == "RAG":

        rag_evidence += (
            f"\nPage {result['page']}:\n"
            f"{result['text']}\n"
        )

    elif result["paper"] == "attention":

        nemo_evidence += (
            f"\nPage {result['page']}:\n"
            f"{result['text']}\n"
        )


# --------------------------------------------------
# Generate independent summaries
# --------------------------------------------------

rag_summary = generate_paper_summary(
    "RAG",
    rag_evidence
)

nemo_summary = generate_paper_summary(
    "NeMo Guardrails",
    nemo_evidence
)


# --------------------------------------------------
# Parse summaries
# --------------------------------------------------

def parse_summary(summary):

    result = {
        "Problem addressed": "Not covered in retrieved evidence.",
        "Main approach": "Not covered in retrieved evidence.",
        "Key mechanism": "Not covered in retrieved evidence.",
        "Benefits": "Not covered in retrieved evidence.",
        "Limitations": "Not covered in retrieved evidence."
    }

    for line in summary.splitlines():

        line = line.strip()

        for key in result:

            prefix = key + ":"

            if line.startswith(prefix):

                value = line[len(prefix):].strip()

                if value:

                    result[key] = value

    return result


rag = parse_summary(
    rag_summary
)

nemo = parse_summary(
    nemo_summary
)


# --------------------------------------------------
# Build comparison table in Python
# --------------------------------------------------

comparison = f"""
| Aspect | RAG | NeMo Guardrails |
|---|---|---|
| Problem addressed | {rag["Problem addressed"]} | {nemo["Problem addressed"]} |
| Main approach | {rag["Main approach"]} | {nemo["Main approach"]} |
| Key mechanism | {rag["Key mechanism"]} | {nemo["Key mechanism"]} |
| Benefits | {rag["Benefits"]} | {nemo["Benefits"]} |
| Limitations | {rag["Limitations"]} | {nemo["Limitations"]} |
""".strip()


# --------------------------------------------------
# Display comparison
# --------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "MULTI-PAPER COMPARISON:"
)

print(
    comparison
)


# --------------------------------------------------
# Display evidence
# --------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "EVIDENCE USED:"
)

for result in final_results:

    print(
        f"\n[EVIDENCE {result['evidence_id']}] "
        f"{result['paper']} - "
        f"Page {result['page']}"
    )

    print(
        "Chunk ID:",
        result["chunk_id"]
    )

    print(
        "Reranker Score:",
        round(
            result["score"],
            4
        )
    )
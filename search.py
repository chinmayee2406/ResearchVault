import re

import chromadb
import ollama

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.split()


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


def get_available_papers():

    data = collection.get()

    papers = set()

    for metadata in data["metadatas"]:

        papers.add(
            metadata["paper"]
        )

    return sorted(papers)


def retrieve_evidence(
    query,
    selected_paper=None
):

    query_embedding = embedding_model.encode(
        query
    )

    if selected_paper:

        papers = [selected_paper]

    else:

        papers = get_available_papers()

    final_results = []

    for paper in papers:

        paper_data = collection.get(
            where={
                "paper": paper
            }
        )

        documents = paper_data["documents"]
        metadatas = paper_data["metadatas"]
        ids = paper_data["ids"]

        if not documents:
            continue

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
            for chunk_id, score
            in semantic_scores[:10]
        ]

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

        hybrid_ids = reciprocal_rank_fusion(
            semantic_ids,
            bm25_ids
        )

        hybrid_ids = hybrid_ids[:10]

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

        if not candidates:
            continue

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

        ranked_candidates = sorted(
            zip(
                candidates,
                rerank_scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        for candidate, score in ranked_candidates[:5]:

            final_results.append({
                "paper": candidate["metadata"]["paper"],
                "page": candidate["metadata"]["page"],
                "chunk_id": candidate["metadata"]["chunk_id"],
                "score": float(score),
                "text": candidate["text"]
            })

    return final_results


def build_evidence_context(results):

    grouped_results = {}

    for result in results:

        paper = result["paper"]

        if paper not in grouped_results:

            grouped_results[paper] = []

        grouped_results[paper].append(
            result
        )

    evidence_context = ""

    evidence_items = []

    evidence_number = 1

    for paper, paper_results in grouped_results.items():

        evidence_context += (
            f"\n\nPAPER: {paper}\n"
        )

        for result in paper_results:

            result["evidence_id"] = (
                evidence_number
            )

            evidence_context += (
                f"\n[EVIDENCE {evidence_number} | "
                f"{paper}, Page {result['page']}]\n"
            )

            evidence_context += result["text"]

            evidence_context += "\n"

            evidence_items.append({
                "evidence_id": evidence_number,
                "paper": result["paper"],
                "page": result["page"],
                "chunk_id": result["chunk_id"],
                "score": result["score"],
                "text": result["text"]
            })

            evidence_number += 1

    return evidence_context, evidence_items


def generate_answer(
    query,
    evidence_context
):

    prompt = f"""
You are a research assistant.

Answer the user's question using ONLY the evidence
provided below.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Compare the papers when appropriate.

4. Explain how the papers address the challenges
   mentioned in the question.

5. Keep the answer concise and structured.

6. Organize the answer clearly by paper when useful.

7. Do NOT generate citations.

8. Do NOT generate page numbers.

9. Do NOT generate evidence IDs.

10. Focus only on producing an accurate answer
    grounded in the provided evidence.

Evidence:

{evidence_context}

Question:

{query}

Answer:
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


def add_citations(
    answer,
    evidence_items
):

    if not evidence_items:

        return answer

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer
    )

    evidence_texts = [
        item["text"]
        for item in evidence_items
    ]

    evidence_embeddings = embedding_model.encode(
        evidence_texts
    )

    cited_answer = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_embedding = (
            embedding_model.encode(
                sentence
            )
        )

        similarities = []

        for index, evidence_embedding in enumerate(
            evidence_embeddings
        ):

            similarity = embedding_model.similarity(
                sentence_embedding,
                evidence_embedding
            )

            similarities.append(
                (
                    index,
                    float(similarity)
                )
            )

        similarities.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_evidence = similarities[:2]

        citations = []

        for index, similarity in top_evidence:

            evidence = evidence_items[index]

            citation = (
                f"[{evidence['paper']}, "
                f"Page {evidence['page']}]"
            )

            if citation not in citations:

                citations.append(
                    citation
                )

        citation_text = " ".join(
            citations
        )

        cited_answer += (
            f"{sentence} {citation_text}\n\n"
        )

    return cited_answer.strip()


def answer_question(
    query,
    selected_paper=None
):

    results = retrieve_evidence(
        query,
        selected_paper
    )

    if not results:

        return {
            "answer": (
                "I could not find relevant "
                "evidence in the selected "
                "research papers."
            ),
            "sources": []
        }

    evidence_context, evidence_items = (
        build_evidence_context(
            results
        )
    )

    answer = generate_answer(
        query,
        evidence_context
    )

    cited_answer = add_citations(
        answer,
        evidence_items
    )

    sources = []

    for evidence in evidence_items:

        sources.append({
            "paper": evidence["paper"],
            "page": evidence["page"],
            "chunk_id": evidence["chunk_id"]
        })

    return {
        "answer": cited_answer,
        "sources": sources
    }


if __name__ == "__main__":

    query = input(
        "\nAsk a question: "
    )

    result = answer_question(
        query
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANSWER:"
    )

    print(
        result["answer"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SOURCES:"
    )

    for source in result["sources"]:

        print(
            f"- {source['paper']} | "
            f"Page {source['page']} | "
            f"{source['chunk_id']}"
        )
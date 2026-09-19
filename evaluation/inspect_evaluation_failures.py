from evaluate_retrieval import retrieve, calculate_recall_at_5
from evaluation_dataset import evaluation_questions


print("\n" + "=" * 80)
print("RESEARCHVAULT RETRIEVAL FAILURE ANALYSIS")
print("=" * 80)


for item in evaluation_questions:

    question = item["question"]
    paper = item["paper"]
    gold_chunks = item["gold_chunks"]

    results = retrieve(
        question,
        paper
    )

    print("\n" + "-" * 80)
    print(f"Question: {question}")
    print(f"Paper: {paper}")
    print(f"Gold chunks: {gold_chunks}")

    for method in [
        "semantic",
        "bm25",
        "hybrid",
        "reranked"
    ]:

        recall = calculate_recall_at_5(
            results[method],
            gold_chunks
        )

        top_chunks = [
            result["chunk_id"]
            for result in results[method][:5]
        ]

        status = "PASS" if recall == 1 else "FAIL"

        print(
            f"{method.upper():10} "
            f"{status:4} "
            f"Top-5: {top_chunks}"
        )
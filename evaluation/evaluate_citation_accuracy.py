import json

import chromadb


# ---------------------------------------------------------
# Load generated answers
# ---------------------------------------------------------

with open(
    "evaluation/generated_answers.json",
    "r",
    encoding="utf-8"
) as file:
    generated_answers = json.load(file)


# ---------------------------------------------------------
# Connect to ChromaDB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_collection(
    name="research_papers"
)


# ---------------------------------------------------------
# Validate one evidence reference
# ---------------------------------------------------------

def validate_evidence_reference(evidence_item):

    chunk_id = evidence_item["chunk_id"]
    expected_page = evidence_item["page"]

    result = collection.get(
        ids=[chunk_id]
    )

    if not result["documents"]:
        return {
            "valid": False,
            "reason": "Chunk not found"
        }

    metadata = result["metadatas"][0]

    actual_page = metadata["page"]
    actual_paper = metadata["paper"]

    if actual_page != expected_page:
        return {
            "valid": False,
            "reason": (
                f"Page mismatch: "
                f"stored={expected_page}, "
                f"actual={actual_page}"
            )
        }

    return {
        "valid": True,
        "paper": actual_paper,
        "page": actual_page
    }


# ---------------------------------------------------------
# Evaluate citations
# ---------------------------------------------------------

print()
print("=" * 70)
print("RESEARCHVAULT CITATION ACCURACY EVALUATION")
print("=" * 70)


answer_scores = []
evaluation_results = []


for item in generated_answers:

    question = item["question"]
    paper = item["paper"]
    evidence = item["evidence"]

    print()
    print("-" * 70)
    print(f"Question: {question}")
    print(f"Paper: {paper}")

    if not evidence:

        print("No evidence references found.")

        answer_scores.append(0.0)

        evaluation_results.append({
            "question": question,
            "paper": paper,
            "citation_accuracy": 0.0,
            "evidence_references": 0
        })

        continue

    valid_references = 0

    for evidence_item in evidence:

        chunk_id = evidence_item["chunk_id"]
        page = evidence_item["page"]

        validation = validate_evidence_reference(
            evidence_item
        )

        if validation["valid"]:

            valid_references += 1

            print(
                f"[{paper}, Page {page}] "
                f"→ CORRECT "
                f"(Chunk {chunk_id})"
            )

        else:

            print(
                f"[{paper}, Page {page}] "
                f"→ INCORRECT "
                f"(Chunk {chunk_id})"
            )

            print(
                f"Reason: "
                f"{validation['reason']}"
            )

    citation_accuracy = (
        valid_references
        / len(evidence)
    )

    answer_scores.append(
        citation_accuracy
    )

    evaluation_results.append({
        "question": question,
        "paper": paper,
        "citation_accuracy": round(
            citation_accuracy,
            3
        ),
        "valid_references": valid_references,
        "total_references": len(evidence)
    })

    print()
    print(
        f"Citation Accuracy: "
        f"{citation_accuracy:.3f}"
    )


# ---------------------------------------------------------
# Overall score
# ---------------------------------------------------------

if answer_scores:

    overall_accuracy = (
        sum(answer_scores)
        / len(answer_scores)
    )

else:

    overall_accuracy = 0.0


print()
print("=" * 70)
print("OVERALL CITATION ACCURACY")
print("=" * 70)

print(
    f"Average Citation Accuracy: "
    f"{overall_accuracy:.3f}"
)

print(
    f"Answers evaluated: "
    f"{len(answer_scores)}"
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_output = {

    "metric": "Citation Accuracy",

    "description": (
        "Measures whether evidence references "
        "in generated answers correctly map "
        "to existing ChromaDB chunks and "
        "their stored page metadata."
    ),

    "warning": (
        "This metric validates citation metadata "
        "and evidence references. It does not "
        "determine whether the cited evidence "
        "semantically supports the specific claim."
    ),

    "overall_score": round(
        overall_accuracy,
        3
    ),

    "answers_evaluated": len(
        answer_scores
    ),

    "results": evaluation_results
}


output_path = (
    "evaluation/citation_accuracy_results.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results_output,
        file,
        indent=4,
        ensure_ascii=False
    )


print()
print(
    f"Results saved to: "
    f"{output_path}"
)
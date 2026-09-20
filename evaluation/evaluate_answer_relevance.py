
import json

from evaluation.answer_evaluation_dataset import (
    answer_evaluation_questions
)


# ==================================================
# Load generated answers
# ==================================================

with open(
    "evaluation/generated_answers.json",
    "r",
    encoding="utf-8"
) as file:

    generated_answers = json.load(file)


# ==================================================
# Concept coverage
# ==================================================

def calculate_concept_coverage(
    answer,
    expected_concepts
):

    answer_lower = answer.lower()

    matched_concepts = []

    for concept in expected_concepts:

        if concept.lower() in answer_lower:

            matched_concepts.append(
                concept
            )

    if not expected_concepts:

        return 0.0, matched_concepts

    coverage = (
        len(matched_concepts)
        / len(expected_concepts)
    )

    return coverage, matched_concepts


# ==================================================
# Match questions with generated answers
# ==================================================

answer_lookup = {}

for item in generated_answers:

    question = item["question"]

    answer_lookup[question] = item["answer"]


# ==================================================
# Run evaluation
# ==================================================

print()
print("=" * 70)
print("RESEARCHVAULT ANSWER RELEVANCE EVALUATION")
print("=" * 70)


scores = []


for item in answer_evaluation_questions:

    question = item["question"]

    expected_concepts = item[
        "expected_concepts"
    ]


    answer = answer_lookup.get(
        question
    )


    if answer is None:

        print()
        print("-" * 70)
        print(f"Question: {question}")
        print("ERROR: Generated answer not found.")

        continue


    coverage, matched_concepts = (
        calculate_concept_coverage(
            answer,
            expected_concepts
        )
    )


    scores.append(
        coverage
    )


    print()
    print("-" * 70)

    print(
        f"Question: {question}"
    )

    print(
        f"Expected concepts: "
        f"{expected_concepts}"
    )

    print(
        f"Matched concepts: "
        f"{matched_concepts}"
    )

    print(
        f"Concept Coverage: "
        f"{coverage:.3f}"
    )


# ==================================================
# Overall score
# ==================================================

if scores:

    overall_score = (
        sum(scores)
        / len(scores)
    )

else:

    overall_score = 0.0


print()
print("=" * 70)
print("OVERALL ANSWER RELEVANCE")
print("=" * 70)

print(
    f"Average Concept Coverage: "
    f"{overall_score:.3f}"
)

print(
    f"Questions evaluated: "
    f"{len(scores)}"
)

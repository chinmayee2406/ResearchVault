import json
import re

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
# Basic stopword list
# ---------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "has", "have", "how", "in", "is",
    "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with",
    "what", "which", "who", "these", "those", "can",
    "does", "do", "using", "used", "into", "through",
    "than", "then", "also", "such", "they", "them",
    "there", "being", "but", "not", "only", "more",
    "other", "both", "each", "where", "when"
}


# ---------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z-]*\b",
        text
    )

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return set(words)


# ---------------------------------------------------------
# Split answer into meaningful sentences
# ---------------------------------------------------------

def split_sentences(answer):

    # Remove numbered-list markers such as:
    # 1. First item
    # 2. Second item

    answer = re.sub(
        r"(?<!\w)\d+\.\s+",
        "\n",
        answer
    )

    # Remove parenthesized markers such as:
    # (1), (2), (3)

    answer = re.sub(
        r"\(\d+\)",
        " ",
        answer
    )

    # Split on sentence-ending punctuation
    # or new lines.

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        answer.strip()
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Remove leftover numbering.

        sentence = re.sub(
            r"^\d+\.\s*",
            "",
            sentence
        )

        # Ignore meaningless fragments.

        if len(tokenize(sentence)) < 2:
            continue

        cleaned_sentences.append(
            sentence
        )

    return cleaned_sentences


# ---------------------------------------------------------
# Retrieve evidence text
# ---------------------------------------------------------

def get_evidence_text(evidence):

    evidence_ids = [
        item["chunk_id"]
        for item in evidence
    ]

    result = collection.get(
        ids=evidence_ids
    )

    evidence_lookup = {}

    for chunk_id, document in zip(
        result["ids"],
        result["documents"]
    ):
        evidence_lookup[chunk_id] = document

    evidence_texts = []

    for chunk_id in evidence_ids:

        text = evidence_lookup.get(
            chunk_id
        )

        if text:
            evidence_texts.append(
                text
            )

    return evidence_texts


# ---------------------------------------------------------
# Calculate lexical evidence support
# ---------------------------------------------------------

def calculate_sentence_support(
    sentence,
    evidence_text
):

    sentence_words = tokenize(
        sentence
    )

    evidence_words = tokenize(
        evidence_text
    )

    if not sentence_words:
        return 0.0

    matched_words = (
        sentence_words.intersection(
            evidence_words
        )
    )

    return (
        len(matched_words)
        / len(sentence_words)
    )


# ---------------------------------------------------------
# Evaluate one answer
# ---------------------------------------------------------

def evaluate_answer(
    answer,
    evidence
):

    sentences = split_sentences(
        answer
    )

    evidence_texts = get_evidence_text(
        evidence
    )

    combined_evidence = " ".join(
        evidence_texts
    )

    sentence_scores = []

    for sentence in sentences:

        score = calculate_sentence_support(
            sentence,
            combined_evidence
        )

        sentence_scores.append({
            "sentence": sentence,
            "score": round(score, 3)
        })

    if sentence_scores:

        overall_score = (
            sum(
                item["score"]
                for item in sentence_scores
            )
            / len(sentence_scores)
        )

    else:

        overall_score = 0.0

    return (
        overall_score,
        sentence_scores
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

print()
print("=" * 70)
print("RESEARCHVAULT FAITHFULNESS EVALUATION")
print("=" * 70)

overall_scores = []
evaluation_results = []

for item in generated_answers:

    question = item["question"]
    answer = item["answer"]
    evidence = item["evidence"]

    overall_score, sentence_scores = (
        evaluate_answer(
            answer,
            evidence
        )
    )

    overall_scores.append(
        overall_score
    )

    evaluation_results.append({
        "question": question,
        "paper": item["paper"],
        "answer_evidence_support": round(
            overall_score,
            3
        ),
        "sentence_scores": sentence_scores
    })

    print()
    print("-" * 70)
    print(f"Question: {question}")

    for index, result in enumerate(
        sentence_scores,
        start=1
    ):

        print()

        print(
            f"Sentence {index} "
            f"Support: {result['score']:.3f}"
        )

        print(
            f"  {result['sentence']}"
        )

    print()

    print(
        f"Answer Evidence Support: "
        f"{overall_score:.3f}"
    )


# ---------------------------------------------------------
# Overall score
# ---------------------------------------------------------

if overall_scores:

    overall_faithfulness = (
        sum(overall_scores)
        / len(overall_scores)
    )

else:

    overall_faithfulness = 0.0


# ---------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------

results_output = {
    "metric": (
        "Lexical Evidence Support"
    ),
    "description": (
        "Baseline measuring the proportion "
        "of meaningful answer vocabulary "
        "that overlaps with retrieved evidence."
    ),
    "warning": (
        "This is a lexical evidence-support "
        "baseline and is not a semantic "
        "faithfulness metric."
    ),
    "overall_score": round(
        overall_faithfulness,
        3
    ),
    "answers_evaluated": len(
        overall_scores
    ),
    "results": evaluation_results
}


output_path = (
    "evaluation/faithfulness_results.json"
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


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print()
print("=" * 70)
print("OVERALL FAITHFULNESS BASELINE")
print("=" * 70)

print(
    f"Average Evidence Support: "
    f"{overall_faithfulness:.3f}"
)

print(
    f"Answers evaluated: "
    f"{len(overall_scores)}"
)

print()
print(
    f"Results saved to: "
    f"{output_path}"
)

print()
print(
    "Note: This is a lexical evidence-support "
    "baseline, not a semantic faithfulness metric."
)
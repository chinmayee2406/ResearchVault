from rank_bm25 import BM25Okapi
import re


documents = [
    "NeMo Guardrails controls the behavior of language models.",
    "Colang is a modeling language used with NeMo Guardrails.",
    "BM25 is a keyword based information retrieval algorithm.",
    "Large language models can generate natural language responses."
]


def tokenize(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    return text.split()


# Tokenize documents
tokenized_documents = [
    tokenize(document)
    for document in documents
]


# Create BM25 index
bm25 = BM25Okapi(tokenized_documents)


# User query
query = "What is Colang?"


# Tokenize query
tokenized_query = tokenize(query)


# Calculate BM25 scores
scores = bm25.get_scores(tokenized_query)


# Display results
for i, score in enumerate(scores):

    print("\n" + "=" * 50)

    print("Document:", i)
    print("BM25 score:", round(score, 4))
    print("Text:", documents[i])
evaluation_questions = [
    {
        "question": "What is NeMo Guardrails?",
        "relevant_page": 1
    },
    {
        "question": "What are the five reference Guardrails applications?",
        "relevant_page": 10
    },
    {
        "question": "What is Colang?",
        "relevant_page": 11
    },
    {
        "question": "What are the main concepts in Colang?",
        "relevant_page": 11
    }
]


for item in evaluation_questions:

    print("Question:", item["question"])
    print("Relevant page:", item["relevant_page"])
    print()
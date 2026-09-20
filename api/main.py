from fastapi import FastAPI

from api.schemas import AskRequest, AskResponse
from search import answer_question


app = FastAPI(
    title="ResearchVault API",
    description="Multi-Paper Research Intelligence RAG System",
    version="1.0.0"
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "ResearchVault API"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(
    request: AskRequest
):

    result = answer_question(
        request.question
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }
from fastapi import FastAPI

from api.schemas import AskRequest, AskResponse
from api.upload import router as upload_router
from search import answer_question, get_available_papers


app = FastAPI(
    title="ResearchVault API",
    description="Multi-Paper Research Intelligence RAG System",
    version="1.0.0"
)


app.include_router(upload_router)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "ResearchVault API"
    }


@app.get("/papers")
def get_papers():

    return get_available_papers()


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask_question(
    request: AskRequest
):

    result = answer_question(
        request.question,
        request.paper
    )

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }
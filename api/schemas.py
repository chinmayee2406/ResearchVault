from pydantic import BaseModel


class AskRequest(BaseModel):

    question: str

    paper: str | None = None


class Source(BaseModel):

    paper: str

    page: int

    chunk_id: int


class AskResponse(BaseModel):

    question: str

    answer: str

    sources: list[Source]